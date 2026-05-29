from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


Landmark = list[float]
FrameLandmarks = list[Landmark]


@dataclass(slots=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    total_frames: int
    detected_frames: int


@dataclass(slots=True)
class ClipInfo:
    used_clip: bool
    reason: str
    seg_start: int
    seg_end: int
    peak_idx: int
    peak_video_frame: int
    seg_video_start: int
    seg_video_end: int
    seg_len: int


@dataclass(slots=True)
class ShotMetrics:
    release_video_frame: int
    elbow_angle_deg: float
    release_height: float
    follow_through_drop: float
    lateral_tilt_deg: float
    assist_wrist_sep: float
    elbow_channel: float
    arm: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BallDetection:
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    source: str


@dataclass(slots=True)
class FrameBallDetections:
    frame_index: int
    detections: list[BallDetection]


@dataclass(slots=True)
class BallDetectionRun:
    provider: str
    frame_count: int
    target_labels: list[str]
    sampled_frames: list[int]
    matched_frames: list[FrameBallDetections]
    release_frame: int | None = None
    matched_frame_count: int = 0
    detection_rate: float = 0.0
    release_frame_detected: bool = False
    nearest_detection_frame: int | None = None
    nearest_detection_delta: int | None = None
    best_frame: int | None = None
    best_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReleaseAnalysis:
    pose_release_frame: int
    final_release_frame: int
    source: str
    delta_from_pose: int
    rationale: str
    hand_separation_detected: bool = False
    contact_frame: int | None = None
    separation_frame: int | None = None
    separation_trend: str | None = None
    ball_candidate_frame: int | None = None
    ball_candidate_wrist_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BallDetectionEvaluationRow:
    provider: str
    model_id: str
    input_video: Path
    release_frame: int
    sampled_frame_count: int
    matched_frame_count: int
    detection_rate: float
    release_frame_detected: bool
    nearest_detection_frame: int | None
    nearest_detection_delta: int | None
    best_frame: int | None
    best_confidence: float | None
    output_json: Path
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_video"] = str(self.input_video)
        payload["output_json"] = str(self.output_json)
        return payload


@dataclass(slots=True)
class BallDetectionEvaluationReport:
    provider: str
    model_ids: list[str]
    videos: list[Path]
    rows: list[BallDetectionEvaluationRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_ids": self.model_ids,
            "videos": [str(video) for video in self.videos],
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(slots=True)
class PoseExtraction:
    frame_to_landmarks: dict[int, FrameLandmarks]
    detected_ids: list[int]
    detected_landmarks: list[FrameLandmarks]
    metadata: VideoMetadata


@dataclass(slots=True)
class SingleAnalysisArtifacts:
    output_dir: Path
    overlay_video: Path
    release_frame_image: Path


@dataclass(slots=True)
class SingleAnalysisResult:
    input_video: Path
    arm: str
    metadata: VideoMetadata
    clip_info: ClipInfo
    release_analysis: ReleaseAnalysis
    metrics: ShotMetrics
    tips: list[str]
    artifacts: SingleAnalysisArtifacts
    ball_detection: BallDetectionRun | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_video"] = str(self.input_video)
        payload["artifacts"]["output_dir"] = str(self.artifacts.output_dir)
        payload["artifacts"]["overlay_video"] = str(self.artifacts.overlay_video)
        payload["artifacts"]["release_frame_image"] = str(self.artifacts.release_frame_image)
        return payload


@dataclass(slots=True)
class ComparisonArtifacts:
    output_dir: Path
    comparison_video: Path


@dataclass(slots=True)
class ComparisonResult:
    input_video: Path
    reference_video: Path
    user_release_frame: int
    reference_release_frame: int
    artifacts: ComparisonArtifacts
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_video"] = str(self.input_video)
        payload["reference_video"] = str(self.reference_video)
        payload["artifacts"]["output_dir"] = str(self.artifacts.output_dir)
        payload["artifacts"]["comparison_video"] = str(self.artifacts.comparison_video)
        return payload
