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


def _safe_dest_path(dest_dir: Path, src: Path) -> Path:
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


def _generate_tree_html(results: list[dict], out_path: Path, threshold: float) -> None:
    """Generate D3 tree visualization HTML."""
    # Build tree structure: root -> [NSFW branch, SFW branch] -> files
    nsfw_children = []
    sfw_children = []

    for r in results:
        node = {"name": r["rel_path"], "score": r.get("nsfw_score", 0)}
        if r["label"] == "nsfw":
            nsfw_children.append(node)
        else:
            sfw_children.append(node)

    tree_data = {
        "name": "Scan Results",
        "children": [
            {"name": f"NSFW ({len(nsfw_children)})", "type": "nsfw", "children": nsfw_children},
            {"name": f"SFW ({len(sfw_children)})", "type": "sfw", "children": sfw_children},
        ],
    }

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NSFW Scan Results</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }}
        #tree {{ width: 100%; height: 90vh; overflow: auto; }}
        .node circle {{ fill: #fff; stroke-width: 2px; cursor: pointer; }}
        .node.nsfw circle {{ stroke: #ff6b6b; fill: #ff6b6b; }}
        .node.sfw circle {{ stroke: #51cf66; fill: #51cf66; }}
        .node.root circle {{ stroke: #339af0; fill: #339af0; }}
        .node text {{ font-size: 12px; fill: #fff; }}
        .link {{ fill: none; stroke: #555; stroke-width: 1.5px; }}
        .tooltip {{ position: absolute; padding: 8px; background: rgba(0,0,0,0.9); border-radius: 4px; pointer-events: none; font-size: 12px; max-width: 400px; word-break: break-all; }}
        h1 {{ margin: 0 0 10px 0; font-size: 18px; }}
        .stats {{ color: #aaa; margin-bottom: 10px; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>NSFW Scan Results</h1>
    <div class="stats">Threshold: {threshold} | Total: {len(results)} images</div>
    <div id="tree"></div>
    <div class="tooltip" style="display:none;"></div>
    <script>
        const data = {json.dumps(tree_data, indent=2)};
        const margin = {{top: 40, right: 120, bottom: 20, left: 120}};
        const width = window.innerWidth - margin.left - margin.right;
        const height = Math.max(600, {len(results) * 25}) - margin.top - margin.bottom;

        const svg = d3.select("#tree").append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

        const tree = d3.tree().size([height, width]);
        const root = d3.hierarchy(data);
        tree(root);

        // Links
        svg.selectAll(".link")
            .data(root.links())
            .enter().append("path")
            .attr("class", "link")
            .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

        // Nodes
        const node = svg.selectAll(".node")
            .data(root.descendants())
            .enter().append("g")
            .attr("class", d => "node " + (d.data.type || (d.depth === 0 ? "root" : "")))
            .attr("transform", d => `translate(${{d.y}},${{d.x}})`);

        node.append("circle")
            .attr("r", d => d.depth === 0 ? 8 : (d.children ? 6 : 4));

        node.append("text")
            .attr("dy", ".35em")
            .attr("x", d => d.children ? -12 : 12)
            .style("text-anchor", d => d.children ? "end" : "start")
            .text(d => d.data.name);

        // Tooltip
        const tooltip = d3.select(".tooltip");
        node.on("mouseover", function(event, d) {{
            if (!d.children) {{
                tooltip.style("display", "block")
                    .html(`<strong>${{d.data.name}}</strong>${{d.data.score ? `<br>NSFW Score: ${{d.data.score.toFixed(4)}}` : ""}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            }}
        }}).on("mouseout", () => tooltip.style("display", "none"));
    </script>
</body>
</html>'''

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

    # Only create output dirs if organizing
    if args.organize:
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
    tree_results = []

    for idx, image_path in enumerate(images, start=1):
        rel = image_path.relative_to(root)
        try:
            result = _classify_image(classifier, image_path=image_path, threshold=args.threshold)
            bucket_dir = nsfw_dir if result.label == "nsfw" else sfw_dir
            dest = _safe_dest_path(bucket_dir, image_path)

            # Store result for tree view
            tree_results.append({
                "rel_path": str(rel),
                "label": result.label,
                "nsfw_score": result.nsfw_score,
                "normal_score": result.normal_score,
            })

            # Only organize files if requested
            if args.organize:
                if args.dry_run:
                    try:
                        dest_display = dest.relative_to(root)
                    except ValueError:
                        dest_display = dest
                    print(f"[{idx}/{len(images)}] {rel} -> {dest_display} ({result.label})")
                else:
                    _copy_or_move(image_path, dest, action=args.action)

            if result.label == "nsfw":
                nsfw_count += 1
            else:
                sfw_count += 1

            if args.write_csv:
                csv_rows.append(
                    {
                        "source": str(image_path),
                        "destination": str(dest),
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

    # Generate tree view HTML (default behavior)
    if not args.no_gallery and tree_results:
        gallery_path = out_dir / "gallery.html"
        _generate_tree_html(tree_results, gallery_path, args.threshold)
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
