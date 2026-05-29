from __future__ import annotations

from pathlib import Path

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.dependencies import require_cv2, require_mediapipe
from basketball_analyzer.models import PoseExtraction, VideoMetadata


def extract_pose_landmarks(video_path: Path, config: AnalyzerConfig) -> PoseExtraction:
    cv2 = require_cv2()
    mp = require_mediapipe()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=config.pose.model_complexity,
        min_detection_confidence=config.pose.min_detection_confidence,
        min_tracking_confidence=config.pose.min_tracking_confidence,
    )

    frame_to_landmarks: dict[int, list[list[float]]] = {}
    detected_ids: list[int] = []
    detected_landmarks: list[list[list[float]]] = []

    idx = 0
    detected = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            detected += 1
            landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
            frame_to_landmarks[idx] = landmarks
            detected_ids.append(idx)
            detected_landmarks.append(landmarks)

        idx += 1

    cap.release()
    pose.close()

    return PoseExtraction(
        frame_to_landmarks=frame_to_landmarks,
        detected_ids=detected_ids,
        detected_landmarks=detected_landmarks,
        metadata=VideoMetadata(
            fps=float(fps),
            width=width,
            height=height,
            total_frames=idx,
            detected_frames=detected,
        ),
    )

