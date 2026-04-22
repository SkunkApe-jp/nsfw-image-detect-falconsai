# NSFW Folder Scanner (Falconsai)

Scans a folder of images using the Hugging Face model `Falconsai/nsfw_image_detection` and generates an interactive D3 tree view gallery showing NSFW/SFW classification results. Optionally organize files into folders.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `torch` can be large; install the right build for your machine (CPU vs CUDA).

## Run

**Default** (scan + generate gallery, recursive by default):

```powershell
python .\scan_nsfw_folder.py "C:\path\to\images"
```

This creates `C:\path\to\images\_nsfw_scan\gallery.html` with an interactive tree view.

### Optional Flags

- `--organize` — Copy/move images into `sfw/` and `nsfw/` folders (default: gallery only)
- `--action move` — Move instead of copy when organizing
- `--threshold 0.7` — Be stricter about NSFW (default: 0.5)
- `--device 0` — Use first CUDA GPU
- `--write-csv` — Also write `_nsfw_scan\report.csv`
- `--no-gallery` — Skip generating the HTML gallery

