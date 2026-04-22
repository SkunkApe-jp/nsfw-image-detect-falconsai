from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_MODEL_ID = "Falconsai/nsfw_image_detection"


@dataclass(frozen=True)
class Classification:
    label: str  # "nsfw" or "normal" (aka sfw)
    nsfw_score: Optional[float]
    normal_score: Optional[float]


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.resolve().relative_to(other.resolve())
        return True
    except (ValueError, OSError):
        return False


def _iter_images(root: Path, recursive: bool, out_dir: Optional[Path]) -> Iterable[Path]:
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    iterator = root.rglob("*") if recursive else root.iterdir()
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed_suffixes:
            continue
        if out_dir is not None and _is_relative_to(p, out_dir):
            continue
        yield p


def _load_classifier(model_id: str, device: int):
    from transformers import pipeline

    return pipeline(
        "image-classification",
        model=model_id,
        device=device,
    )


def _classify_image(classifier, image_path: Path, threshold: float) -> Classification:
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        results = classifier(img, top_k=2)

    # HF pipelines typically return: [{"label":"nsfw","score":...}, {"label":"normal","score":...}]
    nsfw_score: Optional[float] = None
    normal_score: Optional[float] = None
    top_label = None
    for item in results:
        label = str(item.get("label", "")).strip().lower()
        score = float(item.get("score", 0.0))
        if top_label is None:
            top_label = label
        if label == "nsfw":
            nsfw_score = score
        elif label in ("normal", "sfw", "safe"):
            normal_score = score

    if nsfw_score is None:
        predicted = "nsfw" if (top_label and "nsfw" in top_label) else "normal"
        return Classification(label=predicted, nsfw_score=None, normal_score=None)

    predicted = "nsfw" if nsfw_score >= threshold else "normal"
    return Classification(label=predicted, nsfw_score=nsfw_score, normal_score=normal_score)


def _safe_dest_path(dest_dir: Path, src: Path, *, create_dir: bool = True) -> Path:
    if create_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / src.name
    if not candidate.exists():
        return candidate

    stem, suffix = src.stem, src.suffix
    for i in range(1, 10_000):
        candidate = dest_dir / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find a free destination name for: {src}")


def _copy_or_move(src: Path, dest: Path, action: str) -> None:
    if action == "copy":
        shutil.copy2(src, dest)
    elif action == "move":
        shutil.move(src, dest)
    else:
        raise ValueError(f"Unknown action: {action}")


def _generate_gallery_html(results: list[dict], out_path: Path, threshold: float) -> None:
    """Generate an interactive D3-powered thumbnail gallery (no inner scrollbars)."""
    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NSFW Scan Gallery</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #121826;
      --card: #0f1522;
      --border: rgba(255,255,255,0.10);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --nsfw: #ff6b6b;
      --sfw: #51cf66;
      --accent: #339af0;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      overflow-x: hidden;
    }}
    a {{ color: var(--accent); }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 16px; }}
    .top {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      padding: 12px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 12px;
    }}
    .title {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 650; letter-spacing: 0.2px; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    @media (min-width: 900px) {{
      .controls {{ grid-template-columns: 2fr 2fr 1fr 1fr; align-items: end; }}
    }}
    label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    input[type="text"], select {{
      width: 100%;
      padding: 10px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
      color: var(--text);
      outline: none;
    }}
    input[type="range"] {{ width: 100%; }}
    .toggles {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
    }}
    .toggles .t {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      color: var(--text);
      user-select: none;
    }}
    .grid {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 12px;
    }}
    .card {{
      appearance: none;
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0;
      background: var(--card);
      color: var(--text);
      text-align: left;
      overflow: hidden;
      cursor: pointer;
    }}
    .thumb {{
      position: relative;
      width: 100%;
      height: 140px;
      background: rgba(255,255,255,0.03);
    }}
    .thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .badge {{
      position: absolute;
      left: 10px;
      top: 10px;
      font-size: 12px;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.55);
      backdrop-filter: blur(6px);
    }}
    .badge.nsfw {{ border-color: rgba(255,107,107,0.4); color: var(--nsfw); }}
    .badge.sfw {{ border-color: rgba(81,207,102,0.4); color: var(--sfw); }}
    .info {{ padding: 10px 10px 12px; }}
    .name {{
      font-size: 12px;
      color: var(--text);
      line-height: 1.25;
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      word-break: break-word;
      min-height: 30px;
    }}
    .score {{
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
    .pager {{
      margin-top: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }}
    .btn {{
      appearance: none;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 10px;
      cursor: pointer;
    }}
    .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}

    /* Modal */
    .modal {{
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.80);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      z-index: 9999;
    }}
    .modal.open {{ display: flex; }}
    .modal-card {{
      width: min(1200px, 96vw);
      max-height: 92vh;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(15, 21, 34, 0.95);
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }}
    .modal-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 12px;
      border-bottom: 1px solid var(--border);
    }}
    .modal-title {{
      font-size: 13px;
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 80vw;
    }}
    .modal-body {{
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      overflow: hidden;
    }}
    @media (min-width: 900px) {{
      .modal-body {{ grid-template-columns: 2fr 1fr; }}
    }}
    .modal-img {{
      width: 100%;
      height: min(70vh, 680px);
      background: rgba(0,0,0,0.35);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .modal-img img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }}
    .modal-meta {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      background: rgba(255,255,255,0.03);
      overflow: auto;
    }}
    .kv {{ font-size: 12px; color: var(--muted); margin: 6px 0; word-break: break-all; }}
    .kv strong {{ color: var(--text); font-weight: 600; }}
    .modal-foot {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 12px;
      border-top: 1px solid var(--border);
      flex-wrap: wrap;
    }}

    /* Hover Preview */
    .hover-preview {{
      position: fixed;
      pointer-events: none;
      z-index: 10000;
      background: rgba(15, 21, 34, 0.98);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 8px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.6);
      display: none;
      max-width: 400px;
      max-height: 400px;
    }}
    .hover-preview img {{
      max-width: 380px;
      max-height: 380px;
      object-fit: contain;
      border-radius: 8px;
      display: block;
    }}
    .hover-preview .hp-path {{
      font-size: 11px;
      color: var(--muted);
      margin-top: 6px;
      max-width: 380px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="title">
        <h1>NSFW Scan Gallery</h1>
        <div class="meta">Threshold: {threshold} • Total: {len(results)} images</div>
      </div>

      <div class="controls">
        <div>
          <label for="q">Search (filename / path)</label>
          <input id="q" type="text" placeholder="e.g. beach, IMG_1234, folder/name" autocomplete="off">
        </div>

        <div>
          <label for="sort">Sort</label>
          <select id="sort">
            <option value="score_desc">NSFW score (high → low)</option>
            <option value="score_asc">NSFW score (low → high)</option>
            <option value="name_asc">Name (A → Z)</option>
            <option value="name_desc">Name (Z → A)</option>
          </select>
        </div>

        <div>
          <label for="minScore">Min NSFW score</label>
          <input id="minScore" type="range" min="0" max="1" step="0.01" value="0">
          <div class="meta"><span id="minScoreVal">0.00</span></div>
        </div>

        <div>
          <label for="pageSize">Page size</label>
          <select id="pageSize">
            <option value="60">60</option>
            <option value="120" selected>120</option>
            <option value="240">240</option>
            <option value="480">480</option>
          </select>
        </div>
      </div>

      <div class="toggles">
        <label class="t"><input id="showSfw" type="checkbox" checked> Show SFW</label>
        <label class="t"><input id="showNsfw" type="checkbox" checked> Show NSFW</label>
        <label class="t"><input id="blurNsfw" type="checkbox" checked> Blur NSFW thumbnails</label>
        <span class="meta" id="counts"></span>
      </div>
    </div>

    <div class="grid" id="grid"></div>
    <div class="pager">
      <div><span id="pageInfo"></span></div>
      <div style="display:flex; gap:8px; align-items:center;">
        <button class="btn" id="prev">Prev</button>
        <button class="btn" id="next">Next</button>
      </div>
    </div>
    <div class="note">
      Tip: click a card to preview. Keyboard: <strong>Esc</strong> close, <strong>←/→</strong> previous/next.
      If images don't load, your browser may block local file access; try a different browser or serve the folder via a local web server.
    </div>
  </div>

  <!-- Hover Preview -->
  <div class="hover-preview" id="hoverPreview">
    <img id="hoverImg" src="" alt="">
    <div class="hp-path" id="hoverPath"></div>
  </div>

  <div class="modal" id="modal" aria-hidden="true">
    <div class="modal-card" role="dialog" aria-modal="true">
      <div class="modal-head">
        <div class="modal-title" id="modalTitle"></div>
        <button class="btn" id="close">Close</button>
      </div>
      <div class="modal-body">
        <div class="modal-img"><img id="modalImg" alt=""></div>
        <div class="modal-meta" id="modalMeta"></div>
      </div>
      <div class="modal-foot">
        <div class="meta" id="modalIdx"></div>
        <div style="display:flex; gap:8px; align-items:center;">
          <button class="btn" id="mPrev">Prev</button>
          <button class="btn" id="mNext">Next</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {json.dumps(results, ensure_ascii=False)};
    const state = {{
      q: "",
      sort: "score_desc",
      minScore: 0.0,
      showSfw: true,
      showNsfw: true,
      blurNsfw: true,
      page: 1,
      pageSize: 120,
      activeIndex: -1,
    }};

    const $ = (id) => document.getElementById(id);
    const grid = d3.select("#grid");

    function asNumber(v) {{
      const n = Number(v);
      return Number.isFinite(n) ? n : 0;
    }}

    function normalize(s) {{
      return (s || "").toString().toLowerCase();
    }}

    function getFilteredSorted() {{
      const q = normalize(state.q).trim();
      const minScore = state.minScore;
      let items = DATA.filter(d => {{
        const label = (d.label || "").toLowerCase();
        if (!state.showSfw && label !== "nsfw") return false;
        if (!state.showNsfw && label === "nsfw") return false;

        const score = asNumber(d.nsfw_score);
        if (score < minScore) return false;

        if (!q) return true;
        const hay = normalize(d.rel_path || d.name || "");
        return hay.includes(q);
      }});

      const cmpName = (a, b) => normalize(a.rel_path).localeCompare(normalize(b.rel_path));
      const cmpScore = (a, b) => asNumber(a.nsfw_score) - asNumber(b.nsfw_score);

      if (state.sort === "name_asc") items.sort(cmpName);
      else if (state.sort === "name_desc") items.sort((a,b) => -cmpName(a,b));
      else if (state.sort === "score_asc") items.sort(cmpScore);
      else items.sort((a,b) => -cmpScore(a,b));

      return items;
    }}

    function paginate(items) {{
      const total = items.length;
      const totalPages = Math.max(1, Math.ceil(total / state.pageSize));
      state.page = Math.min(Math.max(1, state.page), totalPages);
      const start = (state.page - 1) * state.pageSize;
      const end = Math.min(total, start + state.pageSize);
      return {{ total, totalPages, start, end, pageItems: items.slice(start, end) }};
    }}

    function labelBadge(label) {{
      return (label || "").toLowerCase() === "nsfw" ? "nsfw" : "sfw";
    }}

    function fmtScore(v) {{
      if (v === null || v === undefined || v === "") return "—";
      const n = Number(v);
      return Number.isFinite(n) ? n.toFixed(4) : "—";
    }}

    function render() {{
      const items = getFilteredSorted();
      const {{ total, totalPages, start, end, pageItems }} = paginate(items);

      const totalNsfw = items.filter(d => (d.label || "").toLowerCase() === "nsfw").length;
      const totalSfw = items.length - totalNsfw;
      $("counts").textContent = `Showing ${{items.length}} • SFW ${{totalSfw}} • NSFW ${{totalNsfw}}`;

      $("pageInfo").textContent = `Page ${{state.page}} / ${{totalPages}} • Items ${{total ? (start+1) : 0}}-${{end}} of ${{total}}`;
      $("prev").disabled = state.page <= 1;
      $("next").disabled = state.page >= totalPages;

      const cards = grid.selectAll("button.card").data(pageItems, d => d.rel_path);
      cards.exit().remove();

      const enter = cards.enter().append("button").attr("class", "card");
      enter.append("div").attr("class", "thumb");
      enter.append("div").attr("class", "info");

      const merged = enter.merge(cards);

      merged.attr("title", d => d.rel_path || "");
      merged.on("click", (event, d) => openModal(items, d));

      // Hover preview
      merged.on("mouseenter", (event, d) => {{
        const hp = $("hoverPreview");
        const img = $("hoverImg");
        const path = $("hoverPath");
        img.src = d.image_href || "";
        path.textContent = d.rel_path || "";
        hp.style.display = "block";
      }});
      merged.on("mouseleave", () => {{
        $("hoverPreview").style.display = "none";
        $("hoverImg").src = "";
      }});
      merged.on("mousemove", (event) => {{
        const hp = $("hoverPreview");
        const x = event.clientX + 15;
        const y = event.clientY + 15;
        const maxX = window.innerWidth - 420;
        const maxY = window.innerHeight - 420;
        hp.style.left = Math.min(x, maxX) + "px";
        hp.style.top = Math.min(y, maxY) + "px";
      }});

      merged.select(".thumb").each(function(d) {{
        const el = d3.select(this);
        const badge = labelBadge(d.label);
        let img = el.select("img");
        if (img.empty()) {{
          img = el.append("img")
            .attr("loading", "lazy")
            .attr("alt", "");
        }}
        img.attr("src", d.image_href || "");
        img.style("filter", (state.blurNsfw && badge === "nsfw") ? "blur(18px)" : "none");
        img.on("error", () => img.style("opacity", "0.25"));

        let b = el.select(".badge");
        if (b.empty()) b = el.append("div").attr("class", "badge");
        b.attr("class", `badge ${{badge}}`).text(badge.toUpperCase());
      }});

      merged.select(".info").each(function(d) {{
        const el = d3.select(this);
        let name = el.select(".name");
        if (name.empty()) name = el.append("div").attr("class", "name");
        name.text(d.rel_path || "");

        let score = el.select(".score");
        if (score.empty()) {{
          score = el.append("div").attr("class", "score");
          score.append("div").attr("class", "s");
          score.append("div").attr("class", "n");
        }}
        score.select(".s").text(`NSFW: ${{fmtScore(d.nsfw_score)}}`);
        score.select(".n").text(`SFW: ${{fmtScore(d.normal_score)}}`);
      }});
    }}

    function openModal(allItems, item) {{
      const idx = allItems.findIndex(d => d.rel_path === item.rel_path);
      state.activeIndex = idx;
      updateModal(allItems);
      const modal = $("modal");
      modal.classList.add("open");
      modal.setAttribute("aria-hidden", "false");
    }}

    function closeModal() {{
      const modal = $("modal");
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
      state.activeIndex = -1;
    }}

    function updateModal(allItems) {{
      const idx = state.activeIndex;
      if (idx < 0 || idx >= allItems.length) return;
      const d = allItems[idx];
      $("modalTitle").textContent = d.rel_path || "";
      $("modalImg").src = d.image_href || "";
      $("modalImg").style.filter = "none";
      $("modalMeta").innerHTML = `
        <div class="kv"><strong>Label:</strong> ${{(d.label || "").toUpperCase()}}</div>
        <div class="kv"><strong>NSFW score:</strong> ${{fmtScore(d.nsfw_score)}}</div>
        <div class="kv"><strong>SFW score:</strong> ${{fmtScore(d.normal_score)}}</div>
        <div class="kv"><strong>Path:</strong> ${{d.rel_path || ""}}</div>
        <div class="kv"><strong>Source:</strong> <a href="${{d.image_href || "#"}}"
          target="_blank" rel="noreferrer">open</a></div>
      `;
      $("modalIdx").textContent = `${{idx+1}} / ${{allItems.length}}`;
      $("mPrev").disabled = idx <= 0;
      $("mNext").disabled = idx >= allItems.length - 1;
    }}

    function stepModal(delta) {{
      const items = getFilteredSorted();
      const idx = state.activeIndex;
      if (idx < 0) return;
      const next = Math.min(Math.max(0, idx + delta), items.length - 1);
      state.activeIndex = next;
      updateModal(items);
    }}

    // Wire controls
    $("q").addEventListener("input", (e) => {{ state.q = e.target.value; state.page = 1; render(); }});
    $("sort").addEventListener("change", (e) => {{ state.sort = e.target.value; state.page = 1; render(); }});
    $("pageSize").addEventListener("change", (e) => {{ state.pageSize = Number(e.target.value) || 120; state.page = 1; render(); }});
    $("minScore").addEventListener("input", (e) => {{
      state.minScore = Number(e.target.value) || 0;
      $("minScoreVal").textContent = state.minScore.toFixed(2);
      state.page = 1;
      render();
    }});
    $("showSfw").addEventListener("change", (e) => {{ state.showSfw = !!e.target.checked; state.page = 1; render(); }});
    $("showNsfw").addEventListener("change", (e) => {{ state.showNsfw = !!e.target.checked; state.page = 1; render(); }});
    $("blurNsfw").addEventListener("change", (e) => {{ state.blurNsfw = !!e.target.checked; render(); }});
    $("prev").addEventListener("click", () => {{ state.page -= 1; render(); }});
    $("next").addEventListener("click", () => {{ state.page += 1; render(); }});

    $("close").addEventListener("click", closeModal);
    $("modal").addEventListener("click", (e) => {{
      if (e.target && e.target.id === "modal") closeModal();
    }});
    $("mPrev").addEventListener("click", () => stepModal(-1));
    $("mNext").addEventListener("click", () => stepModal(+1));

    document.addEventListener("keydown", (e) => {{
      if (!$("modal").classList.contains("open")) return;
      if (e.key === "Escape") closeModal();
      else if (e.key === "ArrowLeft") stepModal(-1);
      else if (e.key === "ArrowRight") stepModal(+1);
    }});

    // Initial render
    $("minScoreVal").textContent = state.minScore.toFixed(2);
    render();
  </script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Scan a folder for NSFW/SFW images using Falconsai/nsfw_image_detection and group results."
    )
    parser.add_argument("folder", type=Path, help="Folder to scan")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classify as NSFW when nsfw score >= threshold (default: 0.5)",
    )
    parser.add_argument("--recursive", action="store_true", default=True, help="Scan subfolders recursively")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help='Output folder (default: "<folder>/_nsfw_scan")',
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize images into NSFW/SFW folders (default: only scan and generate gallery)",
    )
    parser.add_argument(
        "--action",
        choices=("copy", "move"),
        default="copy",
        help="Whether to copy or move files when organizing (default: copy)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying/moving")
    parser.add_argument(
        "--device",
        type=int,
        default=int(os.environ.get("NSFW_SCAN_DEVICE", "-1")),
        help="HF pipeline device: -1 for CPU, 0 for first CUDA GPU (default: -1)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop after processing N images (0 = no limit)",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write a CSV report to the output folder",
    )
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Skip generating the HTML gallery view",
    )

    args = parser.parse_args(argv)
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be between 0.0 and 1.0")

    root = args.folder.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Folder does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    out_dir = (args.out.expanduser().resolve() if args.out else (root / "_nsfw_scan"))
    nsfw_dir = out_dir / "nsfw"
    sfw_dir = out_dir / "sfw"

    # Only create output folder when organizing or writing CSV
    if args.organize or args.write_csv:
        out_dir.mkdir(parents=True, exist_ok=True)

    images = list(_iter_images(root, recursive=args.recursive, out_dir=out_dir))
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    if not images:
        print("No images found.")
        return 0

    print(f"Found {len(images)} images. Loading model: {args.model}")
    classifier = _load_classifier(args.model, device=args.device)

    nsfw_count = 0
    sfw_count = 0
    errors = 0

    csv_rows = []
    gallery_results = []

    for idx, image_path in enumerate(images, start=1):
        rel = image_path.relative_to(root)
        try:
            result = _classify_image(classifier, image_path=image_path, threshold=args.threshold)
            dest: Optional[Path] = None
            image_href: str = image_path.resolve().as_uri()

            if args.organize:
                bucket_dir = nsfw_dir if result.label == "nsfw" else sfw_dir
                dest = _safe_dest_path(bucket_dir, image_path, create_dir=not args.dry_run)

                if args.dry_run:
                    try:
                        dest_display = dest.relative_to(root)
                    except ValueError:
                        dest_display = dest
                    print(f"[{idx}/{len(images)}] {rel} -> {dest_display} ({result.label})")
                else:
                    _copy_or_move(image_path, dest, action=args.action)
                    try:
                        image_href = dest.resolve().relative_to(out_dir.resolve()).as_posix()
                    except Exception:
                        image_href = dest.resolve().as_uri()

            # Store result for gallery view
            gallery_results.append(
                {
                    "rel_path": rel.as_posix(),
                    "label": result.label,
                    "nsfw_score": result.nsfw_score,
                    "normal_score": result.normal_score,
                    "image_href": image_href,
                }
            )

            # Only organize files if requested
            if result.label == "nsfw":
                nsfw_count += 1
            else:
                sfw_count += 1

            if args.write_csv:
                csv_rows.append(
                    {
                        "source": str(image_path),
                        "destination": "" if dest is None else str(dest),
                        "label": result.label,
                        "nsfw_score": "" if result.nsfw_score is None else f"{result.nsfw_score:.6f}",
                        "normal_score": "" if result.normal_score is None else f"{result.normal_score:.6f}",
                    }
                )
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            errors += 1
            print(f"[{idx}/{len(images)}] ERROR {rel}: {e}", file=sys.stderr)

    # Generate gallery HTML in root folder (default behavior)
    if not args.no_gallery and gallery_results:
        gallery_path = root / "gallery.html"
        _generate_gallery_html(gallery_results, gallery_path, args.threshold)
        print(f"Gallery: {gallery_path}")

    if args.write_csv and csv_rows and not args.dry_run:
        report_path = out_dir / "report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source", "destination", "label", "nsfw_score", "normal_score"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Report: {report_path}")

    print(f"Done. SFW: {sfw_count}, NSFW: {nsfw_count}, Errors: {errors}")
    if args.organize and not args.dry_run:
        print(f"Output folders: {sfw_dir} , {nsfw_dir}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
