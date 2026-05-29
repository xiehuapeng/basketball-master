from __future__ import annotations

from pathlib import Path

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.dependencies import require_cv2
from basketball_analyzer.detection.providers import build_ball_detector
from basketball_analyzer.models import BallDetectionRun, FrameBallDetections


def _sample_frames_around_release(seg_ids: list[int], release_video_frame: int, config: AnalyzerConfig) -> list[int]:
    ball_cfg = config.ball_detection
    if not seg_ids:
        return []

    frame_set = set(seg_ids)
    sampled: list[int] = []
    for frame in range(
        release_video_frame - ball_cfg.release_window_before,
        release_video_frame + ball_cfg.release_window_after + 1,
        max(1, ball_cfg.sample_stride),
    ):
        if frame in frame_set:
            sampled.append(frame)

    if release_video_frame in frame_set and release_video_frame not in sampled:
        sampled.append(release_video_frame)
    return sorted(set(sampled))


def detect_ball_around_release(
    video_path: Path,
    seg_ids: list[int],
    release_video_frame: int,
    config: AnalyzerConfig,
    window_before: int | None = None,
    window_after: int | None = None,
) -> BallDetectionRun | None:
    detector = build_ball_detector(config.ball_detection)
    if detector is None:
        return None

    original_before = config.ball_detection.release_window_before
    original_after = config.ball_detection.release_window_after
    if window_before is not None:
        config.ball_detection.release_window_before = window_before
    if window_after is not None:
        config.ball_detection.release_window_after = window_after
    try:
        sampled_frames = _sample_frames_around_release(seg_ids, release_video_frame, config)
    finally:
        config.ball_detection.release_window_before = original_before
        config.ball_detection.release_window_after = original_after
    if not sampled_frames:
        return BallDetectionRun(
            provider=config.ball_detection.provider,
            frame_count=0,
            target_labels=list(config.ball_detection.target_labels),
            sampled_frames=[],
            matched_frames=[],
            release_frame=release_video_frame,
        )

    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    matched_frames: list[FrameBallDetections] = []
    best_frame = None
    best_confidence = None

    for frame_index in sampled_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detector.detect(frame)
        if detections:
            matched_frames.append(FrameBallDetections(frame_index=frame_index, detections=detections))
            frame_best = max(detections, key=lambda item: item.confidence)
            if best_confidence is None or frame_best.confidence > best_confidence:
                best_confidence = frame_best.confidence
                best_frame = frame_index

    cap.release()
    matched_frame_count = len(matched_frames)
    detection_rate = float(matched_frame_count / len(sampled_frames)) if sampled_frames else 0.0
    release_frame_detected = any(frame.frame_index == release_video_frame for frame in matched_frames)
    nearest_detection_frame = None
    nearest_detection_delta = None
    if matched_frames:
        nearest_detection_frame = min(
            (frame.frame_index for frame in matched_frames),
            key=lambda frame_index: abs(frame_index - release_video_frame),
        )
        nearest_detection_delta = abs(nearest_detection_frame - release_video_frame)
    return BallDetectionRun(
        provider=config.ball_detection.provider,
        frame_count=len(sampled_frames),
        target_labels=list(config.ball_detection.target_labels),
        sampled_frames=sampled_frames,
        matched_frames=matched_frames,
        release_frame=release_video_frame,
        matched_frame_count=matched_frame_count,
        detection_rate=detection_rate,
        release_frame_detected=release_frame_detected,
        nearest_detection_frame=nearest_detection_frame,
        nearest_detection_delta=nearest_detection_delta,
        best_frame=best_frame,
        best_confidence=best_confidence,
    )
