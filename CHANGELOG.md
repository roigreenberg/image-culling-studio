# Changelog

All notable changes to **ImageCulling Studio** are documented in this file.

## [1.0.0] - 2026-11-24
### Added
- **Full-Dataset Two-Track AI Culling Pipeline:** Project Kestrel (MegaDetector + SAM-HQ ViT-Tiny + BioCLIP 2 + AnimalGate + QualityClassifier) paired with SigLIP 2 (`google/siglip2-so400m-patch14-384`) across 100% of images.
- **Wildlife & Landscape Routing:** Separate scoring tracks for Wildlife/Birds and Sun/Sunrise/Sunset/Landscape, with a `+15%` Hybrid Light bonus for environmental wildlife shots.
- **Scenario-Driven Selection:** `>= 10%` zoom-change scenario splitting, full-sequence retention for strong flying/action sequences, distinct pose/phase promotion for Layer 2 Edit picks, and zero hardcoded count quotas.
- **Personalized LightGBM Ranker (`train_ranker.py`):** Learns from your Keep/Edit/Reject overrides and blends personal preferences into future shoots.
- **Cross-Platform GPU Support:** NVIDIA CUDA & DirectML on Windows, Apple Silicon MPS & CoreML on macOS, plus CPU fallback.
- **One-Click Installers & Auto-Updater:** Zero-Git installers for Windows (`Install_Windows.bat`) and macOS (`Install_MacOS.command`), release ZIP packager (`scripts/build_release.py`), and automatic GitHub Releases update checker/installer (`src/updater.py` / `update.py`).
