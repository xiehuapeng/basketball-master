from __future__ import annotations

from typing import Any

import numpy as np

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.models import ClipInfo, FrameLandmarks, ShotMetrics


def smooth_1d(x: list[float] | np.ndarray, k: int = 7) -> np.ndarray:
    series = np.array(x, dtype=float)
    if len(series) < k or k <= 1:
        return series
    kernel = np.ones(k) / k
    return np.convolve(series, kernel, mode="same")


def calc_angle(a: list[float], b: list[float], c: list[float]) -> float:
    pa, pb, pc = np.array(a), np.array(b), np.array(c)
    ba = pa - pb
    bc = pc - pb
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def choose_shooting_arm(frames_landmarks: list[FrameLandmarks], smooth_window: int) -> str:
    left_wrist = 15
    right_wrist = 16
    left_y = smooth_1d([frame[left_wrist][1] for frame in frames_landmarks], smooth_window)
    right_y = smooth_1d([frame[right_wrist][1] for frame in frames_landmarks], smooth_window)

    left_velocity = np.abs(np.diff(left_y, prepend=left_y[0])).max()
    right_velocity = np.abs(np.diff(right_y, prepend=right_y[0])).max()
    return "left" if left_velocity > right_velocity else "right"


def _arm_joint_ids(arm: str) -> tuple[int, int, int]:
    return (12, 14, 16) if arm == "right" else (11, 13, 15)


def filter_frames(
    frames_landmarks: list[FrameLandmarks],
    frame_ids: list[int],
    arm: str,
    min_box_area: float,
    min_vis: float,
) -> tuple[list[FrameLandmarks], list[int]]:
    shoulder, elbow, wrist = _arm_joint_ids(arm)
    kept_frames: list[FrameLandmarks] = []
    kept_ids: list[int] = []

    for frame, frame_id in zip(frames_landmarks, frame_ids):
        xs = [point[0] for point in frame]
        ys = [point[1] for point in frame]
        box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        visibilities = [frame[shoulder][3], frame[elbow][3], frame[wrist][3]]
        if box_area >= min_box_area and min(visibilities) >= min_vis:
            kept_frames.append(frame)
            kept_ids.append(frame_id)

    return kept_frames, kept_ids


def auto_clip_shot_segment(
    filtered_frames: list[FrameLandmarks],
    filtered_ids: list[int],
    arm: str,
    fps: float,
    config: AnalyzerConfig,
) -> tuple[list[FrameLandmarks], list[int], ClipInfo]:
    shoulder, elbow, wrist = _arm_joint_ids(arm)
    clip_cfg = config.clip
    smooth_window = config.pose.smooth_window

    total = len(filtered_frames)
    if total < 30:
        return filtered_frames, filtered_ids, ClipInfo(
            used_clip=False,
            reason="有效帧过少，跳过自动截取",
            seg_start=0,
            seg_end=max(0, total - 1),
            peak_idx=max(0, total // 2),
            peak_video_frame=filtered_ids[max(0, total // 2)] if filtered_ids else 0,
            seg_video_start=filtered_ids[0] if filtered_ids else 0,
            seg_video_end=filtered_ids[-1] if filtered_ids else 0,
            seg_len=total,
        )

    wrist_y = np.array([frame[wrist][1] for frame in filtered_frames], dtype=float)
    shoulder_y = np.array([frame[shoulder][1] for frame in filtered_frames], dtype=float)
    elbow_angles = np.array(
        [calc_angle(frame[shoulder][:3], frame[elbow][:3], frame[wrist][:3]) for frame in filtered_frames],
        dtype=float,
    )

    wy = smooth_1d(wrist_y, smooth_window)
    sy = smooth_1d(shoulder_y, smooth_window)
    ea = smooth_1d(elbow_angles, smooth_window)
    release_height = sy - wy
    elbow_speed = np.diff(ea, prepend=ea[0])

    candidates: list[tuple[float, int]] = []
    for idx in range(2, total - 2):
        if not (wy[idx] <= wy[idx - 1] and wy[idx] <= wy[idx + 1]):
            continue
        if release_height[idx] < clip_cfg.min_release_height:
            continue
        left_mean = np.mean(wy[max(0, idx - 6) : idx])
        right_mean = np.mean(wy[idx + 1 : min(total, idx + 7)])
        prominence = (left_mean + right_mean) / 2 - wy[idx]
        if prominence < clip_cfg.peak_prominence:
            continue
        left = max(0, idx - 6)
        right = min(total - 1, idx + 6)
        speed = float(np.nanmax(elbow_speed[left : right + 1]))
        score = float(prominence) + clip_cfg.elbow_accel_weight * speed
        candidates.append((score, idx))

    if candidates:
        candidates.sort(reverse=True)
        peak_idx = int(candidates[0][1])
        reason = "使用投篮峰值候选（腕高+峰值显著+伸肘加速）截取"
    else:
        peak_idx = int(np.argmin(wy))
        reason = "未找到强投篮峰值候选，退化为全局手腕最高点截取"

    pre_frames = int(round(clip_cfg.pre_seconds * fps))
    post_frames = int(round(clip_cfg.post_seconds * fps))
    seg_start = max(0, peak_idx - pre_frames)
    seg_end = min(total - 1, peak_idx + post_frames)
    seg_frames = filtered_frames[seg_start : seg_end + 1]
    seg_ids = filtered_ids[seg_start : seg_end + 1]

    clip_info = ClipInfo(
        used_clip=True,
        reason=reason,
        seg_start=seg_start,
        seg_end=seg_end,
        peak_idx=peak_idx,
        peak_video_frame=int(filtered_ids[peak_idx]),
        seg_video_start=int(seg_ids[0]),
        seg_video_end=int(seg_ids[-1]),
        seg_len=int(len(seg_frames)),
    )
    return seg_frames, seg_ids, clip_info


def find_release_frame_early(
    wrist_y: list[float],
    elbow_angles: list[float],
    config: AnalyzerConfig,
) -> int:
    release_cfg = config.release
    smooth_window = config.pose.smooth_window
    wy = smooth_1d(wrist_y, smooth_window)
    ea = smooth_1d(elbow_angles, smooth_window)

    total = len(wy)
    if total < 10:
        return int(np.argmin(wy))

    peak = int(np.argmin(wy))
    left = max(1, peak - release_cfg.peak_left)
    right = min(total - 1, peak + release_cfg.peak_right)
    vy = np.diff(wy, prepend=wy[0])

    start_down = None
    for idx in range(left, right - release_cfg.down_consecutive + 1):
        if all(vy[idx + offset] > release_cfg.down_threshold for offset in range(release_cfg.down_consecutive)):
            start_down = idx
            break

    if start_down is None:
        return max(0, peak - 1)

    d_ea = np.diff(ea, prepend=ea[0])
    window_left = max(0, start_down - release_cfg.pre_window)
    window_right = start_down
    if window_right <= window_left + 1:
        return max(0, start_down - 2)

    best = window_left + int(np.argmax(d_ea[window_left:window_right]))
    return int(max(0, best - release_cfg.release_backoff))


def analyze_shot_segment(
    seg_frames: list[FrameLandmarks],
    seg_ids: list[int],
    arm: str,
    config: AnalyzerConfig,
) -> tuple[int, int, list[float], list[float]]:
    shoulder, elbow, wrist = _arm_joint_ids(arm)
    elbow_angles: list[float] = []
    wrist_y: list[float] = []

    for frame in seg_frames:
        elbow_angles.append(calc_angle(frame[shoulder][:3], frame[elbow][:3], frame[wrist][:3]))
        wrist_y.append(frame[wrist][1])

    if not wrist_y:
        raise ValueError("片段内没有可用帧")

    release_idx = find_release_frame_early(wrist_y, elbow_angles, config)
    release_video_frame = seg_ids[release_idx]
    return release_video_frame, release_idx, elbow_angles, wrist_y


def compute_metrics(
    frames: list[FrameLandmarks],
    ids_: list[int],
    arm: str,
    release_idx: int,
) -> ShotMetrics:
    left_shoulder, left_elbow, left_wrist = 11, 13, 15
    right_shoulder, right_elbow, right_wrist = 12, 14, 16
    left_hip, right_hip = 23, 24

    if arm == "right":
        shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
        other_shoulder, other_wrist = left_shoulder, left_wrist
    else:
        shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
        other_shoulder, other_wrist = right_shoulder, right_wrist

    clipped_idx = max(0, min(release_idx, len(frames) - 1))
    frame = frames[clipped_idx]

    shoulder_point = frame[shoulder]
    elbow_point = frame[elbow]
    wrist_point = frame[wrist]
    other_shoulder_point = frame[other_shoulder]
    other_wrist_point = frame[other_wrist]
    left_hip_point = frame[left_hip]
    right_hip_point = frame[right_hip]

    elbow_angle = calc_angle(shoulder_point[:3], elbow_point[:3], wrist_point[:3])
    release_height = shoulder_point[1] - wrist_point[1]

    post_frames = int(max(1, round(0.35 * 30)))
    post_wrist_y = [
        frames[max(0, min(clipped_idx + offset, len(frames) - 1))][wrist][1]
        for offset in range(1, post_frames + 1)
    ]
    follow_drop = (float(np.max(post_wrist_y)) - wrist_point[1]) if post_wrist_y else float("nan")

    hip_mid = [(left_hip_point[0] + right_hip_point[0]) / 2, (left_hip_point[1] + right_hip_point[1]) / 2]
    shoulder_mid = [
        (shoulder_point[0] + other_shoulder_point[0]) / 2,
        (shoulder_point[1] + other_shoulder_point[1]) / 2,
    ]
    vx = shoulder_mid[0] - hip_mid[0]
    vy = hip_mid[1] - shoulder_mid[1]
    lateral_tilt_deg = float(np.degrees(np.arctan2(abs(vx), vy))) if vy > 1e-6 else float("nan")

    assist_wrist_sep = other_wrist_point[1] - wrist_point[1]
    shoulder_width = abs(frame[right_shoulder][0] - frame[left_shoulder][0]) + 1e-6
    elbow_channel = abs(elbow_point[0] - shoulder_point[0]) / shoulder_width

    return ShotMetrics(
        release_video_frame=int(ids_[clipped_idx]),
        elbow_angle_deg=float(elbow_angle) if not np.isnan(elbow_angle) else float("nan"),
        release_height=float(release_height),
        follow_through_drop=float(follow_drop) if not np.isnan(follow_drop) else float("nan"),
        lateral_tilt_deg=float(lateral_tilt_deg) if not np.isnan(lateral_tilt_deg) else float("nan"),
        assist_wrist_sep=float(assist_wrist_sep),
        elbow_channel=float(elbow_channel),
        arm=arm,
    )


def generate_coach_tips_cn(metrics: ShotMetrics) -> list[str]:
    tips: list[str] = []

    if metrics.release_height >= 0.06:
        tips.append("出手点较高：继续保持手腕高于肩，出手更抗干扰。")
    elif metrics.release_height >= 0.04:
        tips.append("出手点中等：可把球带得更高一点，出手更稳定。")
    else:
        tips.append("出手点偏低：尝试提高出手点，让手腕明显高于肩。")

    if not np.isnan(metrics.follow_through_drop):
        if metrics.follow_through_drop >= 0.05:
            tips.append("随球下压明显：手腕有压筐感，旋转与稳定性较好。")
        elif metrics.follow_through_drop >= 0.03:
            tips.append("随球一般：出手后手腕可以更放松地下压。")
        else:
            tips.append("随球不足：容易推球或平球，建议练习自然拨腕。")

    if not np.isnan(metrics.elbow_angle_deg):
        if metrics.elbow_angle_deg >= 150:
            tips.append("伸肘充分：发力链条更完整。")
        elif metrics.elbow_angle_deg >= 125:
            tips.append("伸肘尚可：出手时可以再把肘和腕送出去一点。")
        else:
            tips.append("伸肘偏少：建议加强伸肘和拨腕衔接。")

    if metrics.assist_wrist_sep >= 0.04:
        tips.append("辅助手退得较好：扶球为主，干扰较小。")
    elif metrics.assist_wrist_sep >= 0.015:
        tips.append("辅助手略晚：可尝试更早离球，避免双手推球。")
    else:
        tips.append("辅助手可能干扰出手：建议更早撤离。")

    if metrics.elbow_channel <= 0.45:
        tips.append("肘通道较直：方向性更稳定。")
    elif metrics.elbow_channel <= 0.65:
        tips.append("肘通道略偏：注意肘对筐，减少外张。")
    else:
        tips.append("肘外张较明显：容易造成左右偏。")

    if not np.isnan(metrics.lateral_tilt_deg):
        if metrics.lateral_tilt_deg <= 12:
            tips.append("躯干稳定：侧倾较小，动作一致性更好。")
        elif metrics.lateral_tilt_deg <= 18:
            tips.append("有轻微侧倾：出手时可加强核心稳定。")
        else:
            tips.append("侧倾偏大：建议加强直上直下起跳和核心控制。")

    return tips

