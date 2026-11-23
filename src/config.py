import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set


IMAGE_EXTENSIONS: Set[str] = {".cr3", ".jpg", ".jpeg"}


def load_local_env(env_path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from an untracked local .env file if present."""
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


@dataclass
class PipelineConfig:
    source_dir: Optional[Path] = None
    output_dir: Path = Path("./cull_output")
    kestrel_dir: Path = Path("./vendor/ProjectKestrel")
    limit: int = 0  # 0 = process all images

    # Step 2A: Project Kestrel (MegaDetector + SAM-HQ + BioCLIP 2 + QualityClassifier)
    kestrel_detector: str = "mdv1000-cedar"  # Fast MegaDetector YOLOv9 ONNX; "mdv5a" also supported
    md_threshold: float = 0.28
    io_workers: int = 8

    # Step 2A.4: Scene / Burst grouping
    burst_gap_seconds: float = 2.0
    scene_gap_seconds: float = 12.0
    scene_sim_threshold: float = 0.84
    zoom_split_ratio: float = 0.10  # >= 10% focal length change starts a new scenario

    # Step 2B: SigLIP 2 (GPU fp16 across 100% of images)
    siglip2_model: str = "google/siglip2-so400m-patch14-384"
    siglip2_fallback_model: str = "google/siglip2-base-patch16-384"
    siglip2_batch_size: int = 16

    # Step 2C: Two-Track Routing weights
    # Track A: Wildlife / Birds
    wildlife_w_sharp: float = 0.40
    wildlife_w_eye: float = 0.30
    wildlife_w_action: float = 0.30
    hybrid_light_bonus: float = 1.15  # +15% bonus for wildlife + dramatic golden/sunrise/sunset light

    # Track B: Sun / Sunrise / Sunset / Landscape
    landscape_w_scenic: float = 0.60
    landscape_w_tech: float = 0.40

    # Step 2D: Scenario-driven quality floors (no hardcoded counts)
    keep_min_overall_wildlife: float = 54.0
    keep_min_sharp_wildlife: float = 45.0
    edit_min_overall_wildlife: float = 65.0
    edit_min_sharp_wildlife: float = 52.0
    keep_min_overall_landscape: float = 48.0
    edit_min_overall_landscape: float = 60.0
    edit_min_scenic_landscape: float = 66.0
    action_pose_sim_threshold: float = 0.945  # Distinct pose/phase threshold in flight sequences

    personal_ranker_path: Path = Path("./models/personal_ranker.pkl")
    personal_blend_weight: float = 0.30

    # Output image sizing
    gallery_thumb_max_dim: int = 720
    gallery_thumb_quality: int = 80
    crop_output_size: int = 384
    crop_quality: int = 86
    export_max_dim: int = 2048
    export_quality: int = 92

    resume: bool = True

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gallery_assets" / "thumbs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gallery_assets" / "crops").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gallery_assets" / "siglip_crops").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "layer1_keep").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "layer2_to_edit").mkdir(parents=True, exist_ok=True)
