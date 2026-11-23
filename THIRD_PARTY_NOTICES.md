# Third-Party Notices & Attributions

This project incorporates and builds upon open-source software, models, and taxonomic datasets. All original copyright notices and license headers are preserved in `./vendor/ProjectKestrel/`.

---

## 1. Project Kestrel (GNU AGPL-3.0)
- **Source:** https://github.com/SanjaySoniLV/ProjectKestrel
- **Copyright:** Copyright (C) 2026 Project Kestrel LLC
- **License:** GNU Affero General Public License v3.0 (AGPL-3.0) — see `./vendor/ProjectKestrel/LICENSE` and `./LICENSE`.
- **Scope:** This repository uses Project Kestrel's local, offline analysis modules (`kestrel_analyzer/ml/`, `similarity.py`, `bird_catalog.py`) in headless mode. Cloud compute, authentication, and proprietary brand assets (`assets/`, logos, icons) have been excluded. There is no affiliation with or endorsement by Project Kestrel LLC.

---

## 2. BioCLIP 2 (MIT License)
- **Source:** https://huggingface.co/imageomics/bioclip-2
- **Copyright:** Copyright (c) 2025 Imageomics Institute
- **License:** MIT License
- Used via Project Kestrel's ONNX image tower (`bioclip2_image_int8.onnx` / `bioclip2_image_wdq16.onnx`) and precomputed species/wildlife/gate embeddings for false-positive gating and zero-shot species identification.

---

## 3. SigLIP 2 (Apache-2.0)
- **Source:** `google/siglip2-so400m-patch14-384` / `timm/ViT-B-16-SigLIP2` (Google `big_vision`)
- **License:** Apache License 2.0 (`https://www.apache.org/licenses/LICENSE-2.0`)
- Used for full-frame wildlife action/pose scoring, landscape/sunrise/sunset scoring, eye-contact/catchlight evaluation, and semantic scene similarity.

---

## 4. MegaDetector & SAM-HQ (MIT / Apache-2.0)
- **MegaDetector (`mdv1000-cedar` / `mdv5a`):** Microsoft / CameraTrapAI wildlife & animal proposal detector (MIT).
- **SAM-HQ (ViT-Tiny ONNX):** High-Quality Segment Anything Model (`sam_hq_vit_tiny_encoder.onnx`, `sam_hq_vit_tiny_decoder.onnx`, Apache-2.0) used for subject silhouette segmentation.

---

## 5. Bird Taxonomy & Catalog Data
- **IOC World Bird List (v15.1):** Frank Gill, David Donsker & Pamela Rasmussen (Eds). 2025. doi:10.14344/IOC.ML.15.1 (`https://www.worldbirdnames.org/`), licensed under **Creative Commons Attribution 3.0 Unported (CC BY 3.0)**.
- **IBP-AOS Alpha Codes:** Pyle, P. and DeSante, D.F., The Institute for Bird Populations (`https://www.birdpop.org/`), 66th AOS Supplement (2025).
