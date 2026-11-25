# Changelog

All notable changes to **ImageCulling Studio** are documented in this file.

## [1.1.0] - 2026-11-25
### Added
- **Auto Smart Straighten & Composition-Aware Smart Crop (`src/smart_crop.py`):**
  - Multi-candidate composition evaluator scoring Rule of Thirds (Left/Right with gaze lead room), Golden Ratio, Upper-Third Center, Centered Subject, and Full Frame (never clipping wings/tails), plus background edge-clutter avoidance.
  - Multi-horizon / waterline detector returning up to 5 optional horizon candidates with live cyan guide preview and one-click reselection (plus `0.0° Level / None`).
  - Bird-size-aware framing with a `2700px` minimum long-side resolution floor (computed from true full-resolution sensor dimensions) and locked aspect ratio by default (with free-ratio toggle and quick presets).
  - Non-destructive Lightroom Classic `crs:Crop*` & `crs:CropAngle` XMP sync with full `Ctrl+Z` Undo.
  - Automatic unapplied suggestion previews (faint gray border) for Layer 2 / Green images, plus single-image and batch **Smart Crop / Reset** controls, interactive drag/resize/angle editing, composition rule badge, and double-click reset in Loupe.

### Improved
- **Smarter Action-Burst Selection (`src/scene_burst_and_select.py`):** Keeping an entire burst (`action_sequence`) now requires genuine dynamic motion/shape evolution (such as takeoffs, landings, vigorous wingbeats, or pose shifts) rather than steady flat glides or static perched poses.
- **Minimum 1 Keep Per Burst:** Every multi-frame wildlife/landscape burst now retains at least its single highest-scoring frame in Layer 1 (`Keep`) unless all frames in the burst are genuinely bad (`to_delete`).

## [1.0.0] - 2026-11-24
### Added
- **Full-Dataset Two-Track AI Culling Pipeline:** Project Kestrel (MegaDetector + SAM-HQ ViT-Tiny + BioCLIP 2 + AnimalGate + QualityClassifier) paired with SigLIP 2 (`google/siglip2-so400m-patch14-384`) across 100% of images.
- **Wildlife & Landscape Routing:** Separate scoring tracks for Wildlife/Birds and Sun/Sunrise/Sunset/Landscape, with a `+15%` Hybrid Light bonus for environmental wildlife shots.
- **Scenario-Driven Selection:** `>= 10%` zoom-change scenario splitting, full-sequence retention for strong flying/action sequences, distinct pose/phase promotion for Layer 2 Edit picks, and zero hardcoded count quotas.
- **Personalized LightGBM Ranker (`train_ranker.py`):** Learns from your Keep/Edit/Reject overrides and blends personal preferences into future shoots.
- **Cross-Platform GPU Support:** NVIDIA CUDA & DirectML on Windows, Apple Silicon MPS & CoreML on macOS, plus CPU fallback.
- **One-Click Installers & Auto-Updater:** Zero-Git installers for Windows (`Install_Windows.bat`) and macOS (`Install_MacOS.command`), release ZIP packager (`scripts/build_release.py`), and automatic GitHub Releases update checker/installer (`src/updater.py` / `update.py`).
