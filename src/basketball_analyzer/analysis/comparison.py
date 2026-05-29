from __future__ import annotations

import math

from basketball_analyzer.models import FrameLandmarks


def body_scale(landmarks: FrameLandmarks) -> float:
    left_shoulder, right_shoulder, left_hip, right_hip = 11, 12, 23, 24

    def dist(a: int, b: int) -> float:
        ax, ay = landmarks[a][0], landmarks[a][1]
        bx, by = landmarks[b][0], landmarks[b][1]
        return float(math.hypot(ax - bx, ay - by))

    shoulder = dist(left_shoulder, right_shoulder)
    hip = dist(left_hip, right_hip)
    return max(0.5 * shoulder + 0.5 * hip, 1e-6)


def body_center(landmarks: FrameLandmarks) -> tuple[float, float]:
    left_hip, right_hip = 23, 24
    x = 0.5 * (landmarks[left_hip][0] + landmarks[right_hip][0])
    y = 0.5 * (landmarks[left_hip][1] + landmarks[right_hip][1])
    return float(x), float(y)


def maybe_mirror_landmarks(landmarks: FrameLandmarks) -> FrameLandmarks:
    mirrored = [point[:] for point in landmarks]
    for point in mirrored:
        point[0] = 1.0 - point[0]
    return mirrored


def interp_landmarks(a: FrameLandmarks, b: FrameLandmarks, alpha: float) -> FrameLandmarks:
    output: FrameLandmarks = []
    for pa, pb in zip(a, b):
        output.append(
            [
                float((1 - alpha) * pa[0] + alpha * pb[0]),
                float((1 - alpha) * pa[1] + alpha * pb[1]),
                float((1 - alpha) * pa[2] + alpha * pb[2]),
                float((1 - alpha) * pa[3] + alpha * pb[3]),
            ]
        )
    return output


def sample_landmarks_at(frame_to_landmarks: dict[int, FrameLandmarks], frame_index: float) -> FrameLandmarks | None:
    left = int(math.floor(frame_index))
    right = int(math.ceil(frame_index))
    if left == right:
        return frame_to_landmarks.get(left)

    left_landmarks = frame_to_landmarks.get(left)
    right_landmarks = frame_to_landmarks.get(right)

    if left_landmarks is None and right_landmarks is None:
        return None
    if left_landmarks is None:
        return right_landmarks
    if right_landmarks is None:
        return left_landmarks

    alpha = float(frame_index - left)
    return interp_landmarks(left_landmarks, right_landmarks, alpha)


def map_reference_to_target(target: FrameLandmarks, reference: FrameLandmarks) -> FrameLandmarks | None:
    if target is None or reference is None:
        return None

    target_scale = body_scale(target)
    reference_scale = body_scale(reference)
    scale = target_scale / reference_scale

    target_center_x, target_center_y = body_center(target)
    reference_center_x, reference_center_y = body_center(reference)

    mapped: FrameLandmarks = []
    for x, y, z, visibility in reference:
        mapped.append(
            [
                float((x - reference_center_x) * scale + target_center_x),
                float((y - reference_center_y) * scale + target_center_y),
                float(z),
                float(visibility),
            ]
        )
    return mapped

