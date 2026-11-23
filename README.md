# ImageCulling — Local GPU Wildlife & Landscape Culling Pipeline

A strictly read-only, GPU-accelerated photography culling pipeline built for Canon `.CR3` RAW and `.JPG`/`.JPEG` shoots on consumer NVIDIA GPUs (tested on RTX 4060 8 GB VRAM). It evaluates **100% of all images** across both **Wildlife / Birds** and **Sun / Sunrise / Sunset / Landscape** tracks before performing scenario-driven selection.

---

## Architecture

1. **Step 2A — Project Kestrel Core Extraction (`CUDAExecutionProvider`):**
   - **Strictly Read-Only & Unicode-Safe I/O:** Reads files via Python `pathlib` + binary file handles (`open(path, "rb")`) and extracts embedded `.CR3` JPEG previews via `rawpy.extract_thumb()` (no slow full RAW demosaicing, safe for non-ASCII/Hebrew Windows paths).
   - **MegaDetector (`mdv1000-cedar`) + SAM-HQ (ViT-Tiny ONNX):** Proposes animal bounding boxes and segments the exact wildlife silhouette.
   - **BioCLIP 2 + AnimalGate:** Filters out false positives (branches, rocks, posts) and identifies bird/wildlife species.
   - **Segmented Sharpness:** Runs Kestrel's `QualityClassifier` on the segmented subject pixels plus masked upper/head sharpness so background bokeh never skews focus scores.

2. **Step 2B — SigLIP 2 Visual Scoring (`google/siglip2-so400m-patch14-384`, FP16 GPU):**
   - Evaluates **100% of full frames** for wildlife action/pose (flight, wing spread, hunting, open beak, clean bokeh) AND sunrise/sunset/golden-hour landscape quality (sun disk, dramatic clouds, reflections, non-blown highlights).
   - Evaluates **tight head/subject crops** on all verified wildlife detections for eye sharpness, catchlight, direct eye contact, and branch occlusion.

3. **Step 2C & 2D — Two-Track Routing & Scenario-Driven Selection (Zero Hardcoded Counts):**
   - **Zoom-Aware Scenario Grouping:** Splits consecutive shots into separate scenarios when EXIF `focal_length` changes by `>= 10%`, or when capture time / semantic similarity indicates a new scene.
   - **Track A (Wildlife) & Track B (Sun/Sunrise/Sunset/Landscape)** with a **+15% Hybrid Light bonus** when wildlife is photographed in dramatic golden-hour/sunrise/sunset light.
   - **Scenario Rules:**
     - **Long Action Sequences (e.g., birds in flight):** If `>= 60%` of the sequence passes the quality floor, **all frames** are kept in Layer 1, and Layer 2 promotes up to 1 peak frame per distinct pose/phase.
     - **Static / Repeated Groups:** Keeps the top 2–3 usable frames in Layer 1 and promotes the best 1–2 to Layer 2.
     - **Singletons:** Kept if above the quality floor; promoted to Layer 2 if standout.

4. **Personalized Learning (`train_ranker.py`):**
   - Review picks in `report.html` and export `my_picks.json`, then train a local LightGBM preference model (`./models/personal_ranker.pkl`) using the saved `embeddings.npz` (`SigLIP 2` + `BioCLIP 2` + technical features). Future runs automatically blend your learned preferences.

---

## Setup & CLI Usage

### 1. Create Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure Input/Output Paths (Optional `.env`)
Copy `.env.example` to `.env` (never committed) or pass paths directly on the CLI:
```powershell
.\.venv\Scripts\python.exe run_cull.py --input-dir "D:\Photos\WildlifeShoot" --output-dir "./cull_output"
```

### 3. Re-select From Cached Scores (Fast, ~1 min)
```powershell
.\.venv\Scripts\python.exe run_cull.py --output-dir "./cull_output" --reselect-only
```

### 4. Train Personal Ranker From Your Selections
```powershell
.\.venv\Scripts\python.exe train_ranker.py --picks "path\to\my_picks.json"
```

---

## Lightroom Classic Filename-Filter Workflow (Zero XMP Modification)

1. Open `./cull_output/report.html` in any browser.
2. Inspect full-frame previews side-by-side with 1:1 head/horizon detail crops.
3. Click **Copy LR: Layer 2** (or **Layer 1**, **Wildlife**, **Landscapes**, or **My Picks**) to copy a comma-separated list of filenames (`IMG_1234, IMG_1235, ...`).
4. In Adobe Lightroom Classic: **Library Filter → Text → Filename → Contains** and paste the list to flag/color-label your picks without modifying any `.xmp` sidecars or `.CR3` files.

---

## License & Attribution

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see [LICENSE](LICENSE).

- Incorporates offline analysis modules from **[SanjaySoniLV/ProjectKestrel](https://github.com/SanjaySoniLV/ProjectKestrel)** under **GNU AGPL-3.0** (`./vendor/ProjectKestrel/LICENSE`).
- **No affiliation:** This project is an independent local pipeline and is not affiliated with or endorsed by Project Kestrel LLC. All Project Kestrel cloud features, telemetry, and proprietary brand assets (`assets/`, logos, icons) are excluded.
- See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for BioCLIP 2 (MIT), SigLIP 2 (Apache-2.0), MegaDetector/SAM-HQ, and IOC World Bird List (CC BY 3.0) attributions.

---

## Security & Zero-Secrets Policy

**In no way may any secret, API key, token, `.env` file, personal filesystem path, private photo/RAW/XMP file, or local culling output ever be committed to this repository.**

See [SECURITY.md](SECURITY.md) for the full policy, `.gitignore` rules, and the automated pre-commit secret guard (`./scripts/pre-commit`).
