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
    from transformers import pipeline, AutoImageProcessor

    # Loading the processor separately with use_fast=False to avoid the "breaking change" warning
    # that occurs when using the pipeline default.
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    except Exception:
        processor = None

    return pipeline(
        "image-classification",
        model=model_id,
        image_processor=processor,
        device=device,
    )


def _classify_image(classifier, image_path: Path, threshold: float) -> Classification:
    from PIL import Image

    with Image.open(image_path) as img:
        # Skip tiny images (like tracking pixels) which cause ambiguous channel warnings or crashes
        if img.width < 10 or img.height < 10:
            return Classification(label="normal", nsfw_score=0.0, normal_score=1.0)

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
    """Generate a collapsible D3 tree view with root at top, NSFW/SFW branches."""
    # Build hierarchical data: root -> [NSFW, SFW] -> images
    nsfw_items = [r for r in results if r.get("label") == "nsfw"]
    sfw_items = [r for r in results if r.get("label") != "nsfw"]
    
    tree_data = {
        "name": f"Scan Results ({len(results)})",
        "type": "root",
        "children": [
            {
                "name": f"NSFW ({len(nsfw_items)})",
                "type": "nsfw",
                "children": [
                    {"name": r["rel_path"], "type": "image", "score": r.get("nsfw_score", 0), "href": r.get("image_href", "")}
                    for r in sorted(nsfw_items, key=lambda x: x.get("nsfw_score", 0), reverse=True)
                ]
            },
            {
                "name": f"SFW ({len(sfw_items)})",
                "type": "sfw", 
                "children": [
                    {"name": r["rel_path"], "type": "image", "score": r.get("nsfw_score", 0), "href": r.get("image_href", "")}
                    for r in sorted(sfw_items, key=lambda x: x.get("nsfw_score", 0), reverse=True)
                ]
            }
        ]
    }
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NSFW Scan Tree</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {{
      margin: 0;
      background: #0a0a0a;
      color: rgba(255,255,255,0.92);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      overflow: hidden;
    }}
    #tree {{
      width: 100vw;
      height: 100vh;
      overflow: auto;
      cursor: grab;
    }}
    #tree:active {{ cursor: grabbing; }}
    .node circle {{
      fill: #fff;
      stroke-width: 2px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .node:hover circle {{
      stroke-width: 3px;
      r: 8;
    }}
    .node.root circle {{ stroke: #339af0; fill: #339af0; r: 10; }}
    .node.nsfw circle {{ stroke: #ff6b6b; fill: #ff6b6b; r: 8; }}
    .node.sfw circle {{ stroke: #51cf66; fill: #51cf66; r: 8; }}
    .node.image circle {{ stroke: #888; fill: #fff; r: 4; }}
    .node.collapsed circle {{ fill: #666 !important; }}
    .node text {{
      font-size: 12px;
      fill: rgba(255,255,255,0.85);
      cursor: pointer;
    }}
    .node.root text {{ font-size: 14px; font-weight: 600; fill: #fff; }}
    .node.nsfw text {{ font-size: 13px; font-weight: 500; fill: #ff6b6b; }}
    .node.sfw text {{ font-size: 13px; font-weight: 500; fill: #51cf66; }}
    .node.image text {{ font-size: 11px; fill: rgba(255,255,255,0.65); }}
    .link {{
      fill: none;
      stroke: rgba(255,255,255,0.2);
      stroke-width: 1.5px;
    }}
    .tooltip {{
      position: absolute;
      padding: 10px;
      background: rgba(15, 21, 34, 0.98);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px;
      pointer-events: none;
      font-size: 12px;
      max-width: 400px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.5);
      display: none;
      z-index: 1000;
    }}
    .tooltip img {{
      max-width: 380px;
      max-height: 300px;
      border-radius: 6px;
      display: block;
      margin-bottom: 8px;
    }}
    .tooltip .path {{
      color: rgba(255,255,255,0.7);
      word-break: break-all;
    }}
    .tooltip .score {{
      color: #ff6b6b;
      font-weight: 500;
      margin-top: 4px;
    }}
    .controls {{
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(15, 21, 34, 0.95);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px;
      padding: 12px;
      font-size: 12px;
      z-index: 100;
    }}
    .controls button {{
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.15);
      color: #fff;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      margin-right: 6px;
      font-size: 12px;
    }}
    .controls button:hover {{ background: rgba(255,255,255,0.15); }}
  </style>
</head>
<body>
  <div id="tree"></div>
  <div class="tooltip" id="tooltip"></div>

  <script>
    const data = {json.dumps(tree_data, indent=2)};
    const margin = {{top: 60, right: 120, bottom: 20, left: 120}};
    const nodeWidth = 300;
    const nodeHeight = 30;
    
    // Setup SVG with zoom
    const svg = d3.select("#tree").append("svg")
      .attr("width", "100%")
      .attr("height", "100%")
      .style("min-height", "100vh");
    
    const g = svg.append("g")
      .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
    
    // Zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);
    
    let i = 0;
    let root;
    
    const tooltip = d3.select("#tooltip");
    
    // Initialize
    root = d3.hierarchy(data);
    root.x0 = 0;
    root.y0 = 0;
    
    // Collapse function
    function collapse(d) {{
      if (d.children) {{
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
      }}
    }}
    
    // Expand function  
    function expand(d) {{
      if (d._children) {{
        d.children = d._children;
        d.children.forEach(expand);
        d._children = null;
      }}
    }}
    
    // Toggle on click
    function toggle(d) {{
      if (d.children) {{
        d._children = d.children;
        d.children = null;
      }} else {{
        d.children = d._children;
        d._children = null;
      }}
      update(d);
    }}
    
    function expandAll() {{
      expand(root);
      update(root);
    }}
    
    function collapseAll() {{
      root.children.forEach(collapse);
      update(root);
    }}
    
    function toggleNsfw() {{
      const nsfwNode = root.children.find(c => c.data.type === "nsfw");
      if (nsfwNode) {{
        if (nsfwNode.children) {{
          collapse(nsfwNode);
        }} else {{
          expand(nsfwNode);
        }}
        update(root);
      }}
    }}
    
    function toggleSfw() {{
      const sfwNode = root.children.find(c => c.data.type === "sfw");
      if (sfwNode) {{
        if (sfwNode.children) {{
          collapse(sfwNode);
        }} else {{
          expand(sfwNode);
        }}
        update(root);
      }}
    }}
    
    // Tree layout - top down
    const tree = d3.tree().nodeSize([nodeHeight, nodeWidth]);
    
    function update(source) {{
      const nodes = root.descendants();
      const links = root.links();
      
      // Compute new tree layout
      tree(root);
      
      // Normalize for fixed depth
      nodes.forEach(d => {{ d.y = d.depth * 250; }});
      
      // Update nodes
      const node = g.selectAll("g.node")
        .data(nodes, d => d.id || (d.id = ++i));
      
      // Enter new nodes
      const nodeEnter = node.enter().append("g")
        .attr("class", d => `node ${{d.data.type}} ${{d._children ? "collapsed" : ""}}`)
        .attr("transform", d => `translate(${{source.y0}},${{source.x0}})`)
        .on("click", (event, d) => toggle(d));
      
      nodeEnter.append("circle")
        .attr("r", 1e-6)
        .style("fill", d => d._children ? "#666" : null);
      
      nodeEnter.append("text")
        .attr("dy", ".35em")
        .attr("x", d => d.children || d._children ? -12 : 12)
        .style("text-anchor", d => d.children || d._children ? "end" : "start")
        .text(d => d.data.name.length > 50 ? d.data.name.slice(0, 47) + "..." : d.data.name)
        .on("mouseover", (event, d) => {{
          if (d.data.type === "image" && d.data.href) {{
            tooltip.style("display", "block")
              .html(`<img src="${{d.data.href}}" onerror="this.style.display='none'" /><div class="path">${{d.data.name}}</div>${{d.data.score !== undefined ? `<div class="score">NSFW: ${{d.data.score.toFixed(4)}}</div>` : ""}}`)
              .style("left", (event.pageX + 15) + "px")
              .style("top", (event.pageY + 15) + "px");
          }} else {{
            tooltip.style("display", "block")
              .html(`<div class="path">${{d.data.name}}</div>`)
              .style("left", (event.pageX + 15) + "px")
              .style("top", (event.pageY + 15) + "px");
          }}
        }})
        .on("mouseout", () => {{
          tooltip.style("display", "none").html("");
        }});
      
      // Transition nodes
      const nodeUpdate = node.merge(nodeEnter).transition().duration(500)
        .attr("transform", d => `translate(${{d.y}},${{d.x}})`);
      
      nodeUpdate.select("circle")
        .attr("r", d => d.data.type === "root" ? 10 : (d.data.type === "image" ? 4 : 8))
        .style("fill", d => d._children ? "#666" : null);
      
      nodeUpdate.attr("class", d => `node ${{d.data.type}} ${{d._children ? "collapsed" : ""}}`);
      
      // Exit nodes
      const nodeExit = node.exit().transition().duration(500)
        .attr("transform", d => `translate(${{source.y}},${{source.x}})`)
        .remove();
      
      nodeExit.select("circle").attr("r", 1e-6);
      nodeExit.select("text").style("fill-opacity", 1e-6);
      
      // Update links
      const link = g.selectAll("path.link")
        .data(links, d => d.target.id);
      
      const linkEnter = link.enter().insert("path", "g")
        .attr("class", "link")
        .attr("d", d => {{
          const o = {{x: source.x0, y: source.y0}};
          return diagonal(o, o);
        }});
      
      const linkUpdate = link.merge(linkEnter).transition().duration(500)
        .attr("d", d => diagonal(d.source, d.target));
      
      link.exit().transition().duration(500)
        .attr("d", d => {{
          const o = {{x: source.x, y: source.y}};
          return diagonal(o, o);
        }})
        .remove();
      
      // Store positions
      nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
      
      // Auto-resize SVG height
      const maxX = d3.max(nodes, d => d.x) || 0;
      const minX = d3.min(nodes, d => d.x) || 0;
      const height = Math.max(window.innerHeight, maxX - minX + margin.top + margin.bottom);
      svg.attr("height", height);
    }}
    
    function diagonal(s, d) {{
      return `M ${{s.y}} ${{s.x}}
              C ${{(s.y + d.y) / 2}} ${{s.x}},
                ${{(s.y + d.y) / 2}} ${{d.x}},
                ${{d.y}} ${{d.x}}`;
    }}
    
    // Initial render
    update(root);
    
    // Center the tree in viewport
    const treeWidth = (root.height + 1) * 250;
    const treeHeight = (root.leaves().length + 1) * 30;
    const svgWidth = svg.node().clientWidth || window.innerWidth;
    const svgHeight = svg.node().clientHeight || window.innerHeight;
    const scale = Math.min(1, svgWidth / (treeWidth + 200), svgHeight / (treeHeight + 100));
    const tx = (svgWidth - treeWidth * scale) / 2 + 60;
    const ty = (svgHeight - treeHeight * scale) / 2 + 30;
    const initialTransform = d3.zoomIdentity.translate(tx, ty).scale(scale);
    svg.call(zoom.transform, initialTransform);
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

    # Generate gallery HTML in output folder
    if not args.no_gallery and gallery_results:
        gallery_path = out_dir / "gallery.html"
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
