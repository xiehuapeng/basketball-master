from __future__ import annotations

from pathlib import Path

import numpy as np

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.dependencies import require_cv2, require_mediapipe
from basketball_analyzer.models import ClipInfo, ShotMetrics
from basketball_analyzer.rendering.common import draw_skeleton, load_chinese_font, put_chinese_text


def save_release_frame(video_path: Path, release_video_frame: int, out_jpg: Path) -> bool:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, release_video_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), frame)
    return True


def overlay_pose_video(
    video_path: Path,
    out_path: Path,
    frame_to_landmarks: dict[int, list[list[float]]],
    fps: float,
    frame_size: tuple[int, int],
    config: AnalyzerConfig,
    metrics: ShotMetrics | None = None,
    tips_cn: list[str] | None = None,
    clip_info: ClipInfo | None = None,
) -> None:
    cv2 = require_cv2()
    mp = require_mediapipe()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    width, height = frame_size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    connections = list(mp.solutions.pose.POSE_CONNECTIONS)
    font = load_chinese_font(26)
    release_frame = metrics.release_video_frame if metrics else None

    text_lines: list[str] = []
    if clip_info and clip_info.used_clip:
        text_lines.append(f"自动截取片段：{clip_info.seg_video_start}~{clip_info.seg_video_end}（原视频帧）")
        text_lines.append(f"峰值候选帧：{clip_info.peak_video_frame} | {clip_info.reason}")

    if metrics is not None:
        text_lines.append(f"离手帧：{metrics.release_video_frame}")
        if not np.isnan(metrics.elbow_angle_deg):
            text_lines.append(f"出手肘角：{metrics.elbow_angle_deg:.1f}°")
        text_lines.append(f"出手高度(肩-腕)：{metrics.release_height:.3f}")
        if not np.isnan(metrics.follow_through_drop):
            text_lines.append(f"随球下压幅度：{metrics.follow_through_drop:.3f}")
        if not np.isnan(metrics.lateral_tilt_deg):
            text_lines.append(f"躯干侧倾角：{metrics.lateral_tilt_deg:.1f}°")
        text_lines.append(f"辅助手分离(腕高差)：{metrics.assist_wrist_sep:.3f}")
        text_lines.append(f"肘通道(归一化)：{metrics.elbow_channel:.3f}")

    if tips_cn:
        text_lines.append("建议（前3条）")
        text_lines.extend(tips_cn[:3])

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = frame_to_landmarks.get(idx)
        highlight = bool(config.render.highlight_release and release_frame is not None and idx == release_frame)

        if landmarks is not None:
            color = (0, 0, 255) if highlight else config.render.me_color
            thickness = 4 if highlight else config.render.line_thickness
            radius = 5 if highlight else config.render.point_radius
            draw_skeleton(
                frame,
                landmarks,
                connections,
                color=color,
                min_vis=config.pose.draw_min_visibility,
                line_thickness=thickness,
                point_radius=radius,
            )
            if highlight:
                cv2.putText(
                    frame,
                    "RELEASE",
                    (20, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

        if config.render.overlay_text and text_lines:
            frame = put_chinese_text(frame, text_lines, x=20, y=20, line_h=34, font=font)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

