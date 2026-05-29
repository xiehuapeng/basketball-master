from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PoseConfig:
    model_complexity: int = 1
    min_detection_confidence: float = 0.25
    min_tracking_confidence: float = 0.25
    smooth_window: int = 7
    min_box_area: float = 0.06
    min_visibility: float = 0.45
    draw_min_visibility: float = 0.30


@dataclass(slots=True)
class ClipConfig:
    pre_seconds: float = 2.0
    post_seconds: float = 2.5
    min_release_height: float = 0.06
    peak_prominence: float = 0.012
    elbow_accel_weight: float = 0.01


@dataclass(slots=True)
class ReleaseConfig:
    down_threshold: float = 0.0025
    down_consecutive: int = 3
    peak_left: int = 12
    peak_right: int = 24
    pre_window: int = 18
    release_backoff: int = 2


@dataclass(slots=True)
class RenderConfig:
    show_preview: bool = False
    overlay_text: bool = True
    highlight_release: bool = True
    line_thickness: int = 2
    point_radius: int = 4
    me_color: tuple[int, int, int] = (0, 255, 0)
    reference_color: tuple[int, int, int] = (255, 120, 0)
    mirror_reference: bool = False


@dataclass(slots=True)
class BallDetectionConfig:
    provider: str = "none"
    target_labels: tuple[str, ...] = ("basketball", "ball", "sports ball")
    release_window_before: int = 6
    release_window_after: int = 6
    sample_stride: int = 1
    confidence_threshold: float = 0.2
    roboflow_model_id: str = "basketball-game-detections/9"
    roboflow_api_key: str = ""
    huggingface_model_id: str = ""
    huggingface_api_key: str = ""


@dataclass(slots=True)
class AnalyzerConfig:
    pose: PoseConfig = field(default_factory=PoseConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    ball_detection: BallDetectionConfig = field(default_factory=BallDetectionConfig)
