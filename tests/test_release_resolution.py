import unittest

from basketball_analyzer.models import BallDetection, BallDetectionRun, FrameBallDetections
from basketball_analyzer.service import AnalysisService


def make_frame_to_landmarks(frames: list[int], wrist_x: float = 0.5, wrist_y: float = 0.5):
    frame_to_landmarks = {}
    for frame in frames:
        landmarks = [[0.5, 0.5, 0.0, 1.0] for _ in range(33)]
        landmarks[11] = [0.4, 0.4, 0.0, 1.0]
        landmarks[12] = [0.6, 0.4, 0.0, 1.0]
        landmarks[16] = [wrist_x, wrist_y, 0.0, 1.0]
        landmarks[15] = [1.0 - wrist_x, wrist_y, 0.0, 1.0]
        frame_to_landmarks[frame] = landmarks
    return frame_to_landmarks


def make_run(detection_frames: list[tuple[int, float]]) -> BallDetectionRun:
    matched_frames = []
    for frame_index, x in detection_frames:
        matched_frames.append(
            FrameBallDetections(
                frame_index=frame_index,
                detections=[
                    BallDetection(
                        label="ball",
                        confidence=0.9,
                        x=x,
                        y=50.0,
                        width=10.0,
                        height=10.0,
                        source="test",
                    )
                ],
            )
        )
    return BallDetectionRun(
        provider="roboflow",
        frame_count=len(detection_frames),
        target_labels=["ball"],
        sampled_frames=[frame for frame, _ in detection_frames],
        matched_frames=matched_frames,
        release_frame=175,
        matched_frame_count=len(detection_frames),
        detection_rate=1.0 if detection_frames else 0.0,
        release_frame_detected=any(frame == 175 for frame, _ in detection_frames),
        nearest_detection_frame=min(
            (frame for frame, _ in detection_frames),
            key=lambda frame: abs(frame - 175),
        )
        if detection_frames
        else None,
        nearest_detection_delta=min(abs(frame - 175) for frame, _ in detection_frames) if detection_frames else None,
        best_frame=detection_frames[0][0] if detection_frames else None,
        best_confidence=0.9 if detection_frames else None,
    )


class ReleaseResolutionTests(unittest.TestCase):
    def setUp(self):
        self.service = AnalysisService()
        self.valid_ids = [148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 175, 176, 177, 178]
        self.frame_to_landmarks = make_frame_to_landmarks(self.valid_ids)

    def test_returns_pose_when_no_ball_detection(self):
        result = self.service._resolve_release_frame_with_ball(
            pose_release_frame=175,
            arm="right",
            valid_frame_ids=self.valid_ids,
            segment_start_frame=175,
            frame_width=100,
            frame_height=100,
            frame_to_landmarks=self.frame_to_landmarks,
            ball_detection=None,
        )

        self.assertEqual(result.final_release_frame, 175)
        self.assertEqual(result.source, "pose")
        self.assertFalse(result.hand_separation_detected)

    def test_confirms_pose_when_ball_stays_close_on_pose_frame(self):
        run = make_run([(175, 54.0), (176, 56.0), (177, 58.0)])
        result = self.service._resolve_release_frame_with_ball(
            pose_release_frame=175,
            arm="right",
            valid_frame_ids=self.valid_ids,
            segment_start_frame=160,
            frame_width=100,
            frame_height=100,
            frame_to_landmarks=self.frame_to_landmarks,
            ball_detection=run,
        )

        self.assertEqual(result.final_release_frame, 175)
        self.assertEqual(result.source, "pose+ball")
        # contact_frame now tracks the LAST hand-on-ball frame of the run.
        self.assertEqual(result.contact_frame, 177)
        self.assertFalse(result.hand_separation_detected)
        self.assertIsNone(result.separation_frame)

    def test_pushes_release_forward_when_ball_still_on_hand_at_pose_frame(self):
        # Ball keeps touching the hand through the pose frame (175..177) and
        # only shows flight afterwards: release must move to the next valid
        # frame after the last touch instead of confirming the pose frame.
        run = make_run([(153, 52.0), (154, 52.0), (155, 53.0), (156, 54.0), (157, 55.0), (175, 56.0), (176, 57.0), (177, 58.0), (178, 120.0)])
        result = self.service._resolve_release_frame_with_ball(
            pose_release_frame=155,
            arm="right",
            valid_frame_ids=self.valid_ids,
            segment_start_frame=148,
            frame_width=100,
            frame_height=100,
            frame_to_landmarks=self.frame_to_landmarks,
            ball_detection=run,
        )

        self.assertEqual(result.source, "ball-contact-transition")
        self.assertEqual(result.contact_frame, 157)
        self.assertEqual(result.final_release_frame, 158)

    def test_corrects_release_to_frame_after_last_contact(self):
        run = make_run([(148, 52.0), (149, 53.0), (156, 120.0), (157, 130.0)])
        result = self.service._resolve_release_frame_with_ball(
            pose_release_frame=175,
            arm="right",
            valid_frame_ids=self.valid_ids,
            segment_start_frame=175,
            frame_width=100,
            frame_height=100,
            frame_to_landmarks=self.frame_to_landmarks,
            ball_detection=run,
        )

        self.assertEqual(result.final_release_frame, 150)
        self.assertEqual(result.source, "ball-contact-transition")
        self.assertEqual(result.contact_frame, 149)
        self.assertEqual(result.separation_frame, 150)
        self.assertTrue(result.hand_separation_detected)


if __name__ == "__main__":
    unittest.main()
