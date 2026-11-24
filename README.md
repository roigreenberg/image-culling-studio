# ImageCulling Studio — GPU Wildlife & Landscape Photography Culling Web UI

A fast, local, Lightroom & Photo Mechanic-inspired **Web UI** powered by a GPU-accelerated AI culling pipeline for RAW (`.CR3`, `.CR2`, `.NEF`, `.ARW`, `.DNG`, `.RAF`, `.ORF`, `.RW2`, and other `rawpy`-supported formats) and `.JPG`/`.JPEG` photography shoots (tested on NVIDIA RTX 4060 8 GB VRAM with 8,600+ RAW shoots).

Review thousands of burst frames at **full camera resolution**, compare **Winner vs. Challenger** side-by-side with synced focus zoom, sync **star ratings & color labels to Lightroom / Photo Mechanic `.xmp` sidecars** in real time (without touching Develop edits), and train a **personal LightGBM ranker** from your own culling decisions.

---

## Quick Start

### 1. Prerequisites
- **Python 3.10+** (Windows 10/11, Linux, or macOS)
- **Hardware:**
  - **NVIDIA GPU with CUDA (8 GB+ VRAM recommended):** Fast AI culling (~6–10 img/s on RTX 4060).
  - **CPU-only / macOS:** Full Web UI browsing, zoom inspection, burst comparison, and XMP sync work smoothly; AI culling also runs on CPU (slower).
- **Disk space & first-run download:** ~2.2 GB total for AI weights (~480 MB Project Kestrel ONNX weights via `setup_models.py` + ~1.7 GB SigLIP 2 from Hugging Face on first AI cull).

### 2. Install & Download AI Models

#### Option A — One-Click Installers (Easiest — No Git or Python needed)
- **Windows:** Double-click `Install_Windows.bat`, then double-click `Start_Windows.bat` to launch.
- **macOS:** Double-click `Install_MacOS.command`, then double-click `Start_MacOS.command` to launch.

#### Option B — Manual Install (Windows PowerShell)
```powershell
git clone https://github.com/roigreenberg/image-culling-studio.git
cd image-culling-studio

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# NVIDIA GPU (CUDA 12.4 — recommended):
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# (Or CPU-only instead: .\.venv\Scripts\python.exe -m pip install torch torchvision)

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Download Project Kestrel AI model weights once (~480 MB):
.\.venv\Scripts\python.exe setup_models.py
```

#### macOS / Linux (bash / zsh)
```bash
git clone https://github.com/roigreenberg/image-culling-studio.git
cd image-culling-studio

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# Linux with NVIDIA GPU (CUDA 12.4):
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# macOS or CPU-only Linux (use this instead of the line above):
# .venv/bin/python -m pip install torch torchvision

.venv/bin/python -m pip install -r requirements.txt

# Download Project Kestrel AI model weights once (~480 MB):
.venv/bin/python setup_models.py
```

> **Tip:** Run `python setup_models.py --check` anytime to verify all required model files are present, or pass `--preload-siglip` to pre-download the ~1.7 GB Hugging Face SigLIP 2 weights ahead of time.

### 3. Launch ImageCulling Studio (Web UI)

Windows:
```powershell
.\.venv\Scripts\python.exe run_webapp.py --open
```

macOS / Linux:
```bash
.venv/bin/python run_webapp.py --open
```

Or omit `--open` and visit **[http://127.0.0.1:8765](http://127.0.0.1:8765)** manually. All CLI flags:
```powershell
.\.venv\Scripts\python.exe run_webapp.py --host 127.0.0.1 --port 8765 --open
```

### 4. First-Time Workflow in the Studio
1. Click **📂 Open Photo Folder** on the welcome screen (or **📂 Open Folder** in the top bar) to select your photo folder. Subfolders (e.g., `Day1`, `Day2`, or camera card folders) appear automatically in the sidebar. Original photos are strictly read-only.
2. Click **⚡ Run AI Cull ▾ → Run Unculled Only** (or run the CLI pipeline first) to score your images with Project Kestrel + SigLIP 2 and group them into bursts/scenarios.
3. Review and refine picks using **Grid (`G`)**, **Bursts (`M`)**, **Loupe (`E`)**, and **Winner vs. Challenger (`C`)**.
4. Rate (`0`–`5`) and color-label (`6`–`9`, `P`) photos — changes save immediately to SQLite and optionally write live to `.xmp` sidecars.
5. Click **🧠 Insights** to view AI vs. user agreement, train your local LightGBM personal ranker from your edits, and run **🔁 Fast Reselect** in seconds.

---

## Studio Views & Features

### 4 Purpose-Built Culling Views
| View | Shortcut | Best For |
| --- | :---: | --- |
| **Grid** | `G` | High-speed overview of filtered photos, multi-select batch rating/labeling, sorting, and facet filtering |
| **Bursts** | `M` | Reviewing shoots chronologically burst-by-burst (one row per scenario/burst, expandable inline) |
| **Loupe** | `E` | Single-image full-resolution inspection with AF point overlay, highlight zebra, zoom presets, and EXIF/AI inspector |
| **Winner vs. Challenger** | `C` | Side-by-side burst culling: **Pinned Winner** on the left, step through **Challengers** on the right with synced pan/zoom |

### Pro Inspection Tools
- **Zero-Reencode Full-Resolution Zoom:** Automatically upgrades from fast HQ preview to the full-resolution image (or embedded full-size RAW JPEG preview) on zoom (`Fit`, `25%`, `50%`, `100%`, `200%`, or mouse wheel).
- **Camera AF Focus Zoom (`Z` / `Space` / `A`):** Parses embedded Canon `.CR3` autofocus point metadata when available to zoom directly onto the active AF point (falling back to the detected subject/center on other cameras) and displays a toggleable transparent red AF box overlay (`F`).
- **Burned Highlights Overlay (`H`):** Highlights clipped specular/blown areas without obscuring subject detail.
- **Hold `Q` to Peek Full-Res:** Instantly inspect full resolution while holding `Q`.
- **Winner vs. Challenger Synced Zoom & Pan:** Drag-pan and wheel-zoom move both Winner and Challenger in lockstep. Press `S` to **Swap Winner**, `K` to **Keep Both**, `↑`/`↓` to step through bursts (including single-frame photos), and `✂ Split Here` / `🔗 Merge Back` to manually split or merge bursts (persisted across re-selects).
- **Live Color State vs. Original AI Toggle:** By default, sidebar buckets (**Layer 2 · To Edit**, **Layer 1 · Keep (All)**, **Not Selected**, **To Delete**) reflect your live color labels so changes move photos immediately. Check the **AI** box next to *Culling Layer* in the sidebar to inspect the AI's original decisions at any time.
- **Persistent Session:** Refreshing the browser keeps your exact view, burst, Winner, Challenger, and scroll position.

---

## Color Labels & Lightroom / Photo Mechanic XMP Sync

ImageCulling Studio uses standard Lightroom Classic & Photo Mechanic color mappings:

| Key | Color | Studio Meaning | Lightroom `xmp:Label` | Photo Mechanic `ColorClass` |
| :---: | --- | --- | --- | :---: |
| `8` | 🟢 **Green** | **Edit** (Layer 2 — Portfolio / Hero Picks) | `Green` | `3` |
| `7` | 🟡 **Yellow** | **Keep** (Layer 1 — Strong Keepers) | `Yellow` | `2` |
| `6` | 🔴 **Red** | **To Delete** (Blurry / Bad / No Subject) | `Red` | `1` |
| `9` | 🔵 **Blue** | Custom Blue | `Blue` | `4` |
| `P` | 🟣 **Purple** | Custom Purple | `Purple` | `5` |
| *(repeat key)* | ∅ **None** | **Not Selected** (Toggles color off) | `""` | `0` |

### Safe XMP Sidecar Sync
- **Preserves Develop Edits:** Only updates `xmp:Rating`, `xmp:Label`, and `photomechanic:ColorClass` via atomic file writes. All Lightroom Develop sliders, crops, masks, and keywords remain untouched.
- **Star Ratings Protected:** Batch AI color-mapping defaults to **Keep** for star ratings so existing camera/Lightroom ratings are never overwritten unless you explicitly configure them or rate a photo.
- **Full Undo (`Ctrl+Z`):** Every rating/color edit is logged with a pre-edit XMP snapshot so `Ctrl+Z` (or **📜 Logs & Restore**) cleanly reverts both the database and `.xmp` sidecar without ever deleting existing Develop edits.
- **Zero-XMP Option:** Disable Real-Time XMP Sync in **⚙ Settings** if you prefer keeping edits strictly inside SQLite without modifying `.xmp` sidecars.

---

## Keyboard Shortcuts

All shortcuts work across lowercase, uppercase, CapsLock, Hebrew keyboard layouts, top number row, and Numpad.

| Keys | Action |
| --- | --- |
| `G` / `M` / `E` / `C` | Switch view: **Grid** / **Bursts** / **Loupe** / **Winner vs. Challenger** |
| `0`–`5` | Set star rating (`0` clears rating) |
| `6` / `7` / `8` / `9` / `P` | Toggle color label: **Red** / **Yellow** / **Green** / **Blue** / **Purple** |
| `←` / `→` | Previous / Next image (Loupe & Grid) or Challenger frame (Winner vs. Challenger) |
| `↑` / `↓` | Previous / Next Burst (Bursts & Winner vs. Challenger) |
| `Z` or `Space` | Toggle Fit ↔ 100% Zoom (centered on AF focus point) |
| `Mouse Wheel` / `Ctrl`+`+`/`-`/`0` | Smooth zoom in/out / Fit |
| `Drag` | Pan zoomed image (synced across both panes in Winner vs. Challenger) |
| `A` | Jump zoomed view to camera AF focus point (Loupe) |
| `Hold Q` | Peek at full-resolution image while holding `Q` (Loupe) |
| `F` / `H` / `B` | Toggle **AF Focus Box** / **Burned Highlights** / **AI Detection Box** (Loupe) |
| `S` | **Loupe:** Split burst at current frame · **Winner vs. Challenger:** Swap Winner |
| `K` | **Winner vs. Challenger:** Keep Both (apply Winner's color to Challenger) |
| `X` / `U` | Reject flag / Clear flag |
| `Ctrl+Z` | Undo last rating / color / XMP edit |
| `?` | Open Keyboard Shortcuts & Help modal |
| `Esc` | Close modal / exit Loupe or Compare view |

> **Tip (Browser Extensions):** If you use single-key navigation extensions such as **Vimium** or **Surfingkeys**, add `http://127.0.0.1:8765/*` to the extension's excluded URLs so single-key shortcuts (`0`–`9`, `G`, `E`, `C`, `S`, `K`) go straight to ImageCulling Studio.

---

## AI Culling Pipeline Architecture

1. **Step 2A — Project Kestrel Core Extraction (`CUDAExecutionProvider` with CPU fallback):**
   - Strictly read-only, Unicode-safe I/O via Python binary file handles + `rawpy.extract_thumb()` (extracts embedded RAW JPEG previews without slow RAW demosaicing, with `rawpy.postprocess` fallback when a RAW lacks an embedded preview).
   - **MegaDetector (`mdv1000-cedar`) + SAM-HQ (ViT-Tiny ONNX):** Detects wildlife and segments exact silhouettes.
   - **BioCLIP 2 + AnimalGate:** Rejects false positives (branches, rocks, posts) and identifies bird/wildlife species.
   - **Segmented Sharpness:** Scores subject and head sharpness separately so background bokeh never skews sharpness metrics.
2. **Step 2B — SigLIP 2 Visual Scoring (`google/siglip2-so400m-patch14-384`, FP16 GPU / CPU fallback):**
   - Scores 100% of full frames for wildlife action/pose (flight, wing spread, hunting, bokeh) and sunrise/sunset/golden-hour landscapes.
   - Scores tight head crops for eye sharpness, catchlight, eye contact, and occlusion.
3. **Step 2C & 2D — Two-Track Routing & Scenario-Driven Selection:**
   - Groups consecutive shots into scenarios/bursts using timestamp gaps, focal-length changes (`>= 10%`), and semantic similarity.
   - Routes into **Track A (Wildlife)** and **Track B (Landscape/Sun)** with a **+15% Hybrid Light bonus** for wildlife in dramatic golden-hour light.
   - Two-tier rejection separates **Not Selected** (usable runner-ups) from **To Delete** (blurry/bad/no subject, while protecting intentional slow-shutter motion blur).
4. **Personalized Learning (LightGBM):**
   - Train `./models/personal_ranker.pkl` directly from the Web UI (**🧠 Insights**) or via CLI using your ratings/labels and cached `embeddings.npz`.

---

## CLI Usage (Optional / Headless)

You can also run the pipeline directly from the command line; ImageCulling Studio automatically imports `cull_output/results.json` into its persistent database (`studio_data/cull_studio.db`) while preserving your manual edits and burst splits.

```powershell
# Full AI culling run on a photo folder
.\.venv\Scripts\python.exe run_cull.py --input-dir "D:\Photos\WildlifeShoot" --output-dir "./cull_output"

# Fast re-selection from cached scores & embeddings
.\.venv\Scripts\python.exe run_cull.py --output-dir "./cull_output" --reselect-only

# Train personal LightGBM ranker from exported picks
.\.venv\Scripts\python.exe train_ranker.py --picks "path\to\my_picks.json"
```

---

## Project Structure

```text
ImageCulling/
├── run_webapp.py                  # FastAPI + Uvicorn launcher for ImageCulling Studio Web UI
├── run_cull.py                    # CLI entry point for AI culling pipeline
├── setup_models.py                # One-command Project Kestrel model downloader & verifier
├── train_ranker.py                # CLI trainer for LightGBM personal preference ranker
├── update.py                      # Built-in self-updater (preserves local DB/models)
├── Install_Windows.bat            # One-click Windows installer (auto-installs Python, deps & models)
├── Start_Windows.bat              # One-click Windows launcher
├── Install_MacOS.command          # One-click macOS installer
├── Start_MacOS.command            # One-click macOS launcher
├── webapp/
│   ├── server.py                  # REST API, WebSocket live updates, media & full-res preview endpoints
│   ├── db.py                      # Persistent SQLite database (studio_data/cull_studio.db)
│   ├── pipeline_runner.py         # Background AI pipeline, EXIF/AF scanner, preview & ranker tasks
│   ├── xmp_sync.py                # Atomic Lightroom + Photo Mechanic .xmp sidecar reader/writer
│   └── static/
│       ├── index.html             # Studio UI markup & Help/Settings/Logs modals
│       ├── app.js                 # Frontend controller (Grid, Bursts, Loupe, Winner vs. Challenger)
│       └── styles.css             # Dark studio theme & responsive layout
├── src/
│   ├── config.py                  # Supported RAW/JPG extensions, pipeline thresholds & model config
│   ├── io_utils.py                # Read-only RAW/JPG I/O, EXIF, embedded XMP & Canon AF point parser
│   ├── kestrel_bridge.py          # MegaDetector + SAM-HQ + BioCLIP 2 + QualityClassifier runner
│   ├── siglip2_scorer.py          # SigLIP 2 visual scorer (GPU FP16 / CPU fallback)
│   ├── scene_burst_and_select.py  # Scenario grouping, two-track selection & two-tier rejection
│   └── deliverables.py            # Export 2048px previews, results.json, CSV & embeddings.npz
├── vendor/ProjectKestrel/         # Vendored offline Project Kestrel analysis modules (AGPL-3.0)
├── studio_data/                   # (gitignored) Persistent SQLite DB & session state
├── cull_output/                   # (gitignored) Generated thumbnails, previews, results.json, embeddings.npz
└── models/                        # (gitignored) Local LightGBM personal ranker & training history
```

---

## License & Attribution

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see [LICENSE](LICENSE).

- Incorporates offline analysis modules from **[SanjaySoniLV/ProjectKestrel](https://github.com/SanjaySoniLV/ProjectKestrel)** under **GNU AGPL-3.0** (`./vendor/ProjectKestrel/LICENSE`).
- **No affiliation:** This project is an independent local studio/pipeline and is not affiliated with or endorsed by Project Kestrel LLC. All Project Kestrel cloud features, telemetry, and proprietary brand assets are excluded.
- See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for BioCLIP 2 (MIT), SigLIP 2 (Apache-2.0), MegaDetector/SAM-HQ, and IOC World Bird List (CC BY 3.0) attributions.

---

## Security & Zero-Secrets Policy

**Never commit secrets, API keys, `.env` files, personal filesystem paths, private photos/RAWs/XMPs, SQLite databases, or local culling outputs.**

See [SECURITY.md](SECURITY.md) for the full policy, `.gitignore` rules, and the automated pre-commit guard (`./scripts/pre-commit`).
