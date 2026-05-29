from __future__ import annotations

from pathlib import Path

from basketball_analyzer.analysis import map_reference_to_target, maybe_mirror_landmarks, sample_landmarks_at
from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.dependencies import require_cv2, require_mediapipe
from basketball_analyzer.models import PoseExtraction
from basketball_analyzer.rendering.common import draw_skeleton, load_chinese_font, put_chinese_text


def render_comparison_video(
    user_video: Path,
    out_path: Path,
    user_pose: PoseExtraction,
    reference_pose: PoseExtraction,
    user_release_frame: int,
    reference_release_frame: int,
    config: AnalyzerConfig,
) -> None:
    cv2 = require_cv2()
    mp = require_mediapipe()
    _ = mp

    cap = cv2.VideoCapture(str(user_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{user_video}")

    width = user_pose.metadata.width
    height = user_pose.metadata.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), user_pose.metadata.fps, (width, height))
    font = load_chinese_font(26)

    ratio = reference_pose.metadata.fps / user_pose.metadata.fps if user_pose.metadata.fps > 0 else 1.0
    connections = list(require_mediapipe().solutions.pose.POSE_CONNECTIONS)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        user_landmarks = user_pose.frame_to_landmarks.get(idx)
        if user_landmarks is not None:
            draw_skeleton(
                frame,
                user_landmarks,
                connections,
                color=config.render.me_color,
                min_vis=config.pose.draw_min_visibility,
                line_thickness=config.render.line_thickness,
                point_radius=config.render.point_radius,
            )

        ref_time = reference_release_frame + (idx - user_release_frame) * ratio
        reference_landmarks = sample_landmarks_at(reference_pose.frame_to_landmarks, ref_time)
        if reference_landmarks is not None:
            if config.render.mirror_reference:
                reference_landmarks = maybe_mirror_landmarks(reference_landmarks)
            mapped = map_reference_to_target(user_landmarks, reference_landmarks) if user_landmarks else reference_landmarks
            if mapped is not None:
                draw_skeleton(
                    frame,
                    mapped,
                    connections,
                    color=config.render.reference_color,
                    min_vis=config.pose.draw_min_visibility,
                    line_thickness=config.render.line_thickness,
                    point_radius=config.render.point_radius,
                )

        lines = [
            "你(绿) vs 参考动作(蓝橙) - 离手对齐",
            f"你的离手帧: {user_release_frame} | 参考离手帧: {reference_release_frame}",
        ]
        frame = put_chinese_text(frame, lines, x=20, y=20, line_h=32, font=font)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

