import math
import unittest

from basketball_analyzer.analysis import (
    body_center,
    body_scale,
    calc_angle,
    find_release_frame_early,
    map_reference_to_target,
    maybe_mirror_landmarks,
    smooth_1d,
)
from basketball_analyzer.config import AnalyzerConfig


def make_landmarks(scale: float = 1.0, center_x: float = 0.5, center_y: float = 0.5):
    landmarks = [[center_x, center_y, 0.0, 1.0] for _ in range(33)]
    landmarks[11] = [center_x - 0.1 * scale, center_y - 0.15 * scale, 0.0, 1.0]
    landmarks[12] = [center_x + 0.1 * scale, center_y - 0.15 * scale, 0.0, 1.0]
    landmarks[23] = [center_x - 0.08 * scale, center_y + 0.1 * scale, 0.0, 1.0]
    landmarks[24] = [center_x + 0.08 * scale, center_y + 0.1 * scale, 0.0, 1.0]
    return landmarks


class AnalysisCoreTests(unittest.TestCase):
    def test_calc_angle_right_angle(self):
        angle = calc_angle([1, 0, 0], [0, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(angle, 90.0, places=4)

    def test_smooth_short_series_is_stable(self):
        data = [1.0, 2.0, 3.0]
        self.assertEqual(list(smooth_1d(data, 7)), data)

    def test_find_release_frame_returns_peak_related_frame(self):
        config = AnalyzerConfig()
        wrist_y = [0.60, 0.55, 0.50, 0.45, 0.40, 0.38, 0.37, 0.39, 0.42, 0.46, 0.50, 0.54]
        elbow_angles = [95, 100, 108, 116, 128, 140, 150, 154, 150, 145, 138, 130]
        release_idx = find_release_frame_early(wrist_y, elbow_angles, config)
        self.assertTrue(0 <= release_idx < len(wrist_y))

    def test_mirror_landmarks_flips_x_axis(self):
        landmarks = make_landmarks()
        mirrored = maybe_mirror_landmarks(landmarks)
        self.assertAlmostEqual(mirrored[11][0], 1.0 - landmarks[11][0], places=6)

    def test_map_reference_to_target_aligns_body_center(self):
        target = make_landmarks(scale=1.0, center_x=0.5, center_y=0.5)
        reference = make_landmarks(scale=0.5, center_x=0.2, center_y=0.3)
        mapped = map_reference_to_target(target, reference)
        target_center = body_center(target)
        mapped_center = body_center(mapped)
        self.assertAlmostEqual(target_center[0], mapped_center[0], places=6)
        self.assertAlmostEqual(target_center[1], mapped_center[1], places=6)

        self.assertAlmostEqual(body_scale(target), body_scale(mapped), places=6)


if __name__ == "__main__":
    unittest.main()
