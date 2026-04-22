# NSFW Folder Scanner (Falconsai)

This repo contains a simple script that scans a folder of images using the Hugging Face model `Falconsai/nsfw_image_detection`, then groups images into `sfw/` and `nsfw/` subfolders.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `torch` can be large; install the right build for your machine (CPU vs CUDA).

## Run

Scan a folder and **copy** images into grouped subfolders (recursive by default):

```powershell
python .\scan_nsfw_folder.py "C:\path\to\images"
```

This creates:

- `C:\path\to\images\_nsfw_scan\sfw\`
- `C:\path\to\images\_nsfw_scan\nsfw\`

To **move** (instead of copy):

```powershell
python .\scan_nsfw_folder.py "C:\path\to\images" --recursive --action move
```

Optional:

- `--threshold 0.7` (be stricter about NSFW)
- `--dry-run` (no file writes)
- `--device 0` (use first CUDA GPU, if available)
- `--write-csv` (writes `_nsfw_scan\report.csv`)

