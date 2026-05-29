from __future__ import annotations

import json
from pathlib import Path

from basketball_analyzer.analysis import (
    analyze_shot_segment,
    auto_clip_shot_segment,
    choose_shooting_arm,
    compute_metrics,
    filter_frames,
    generate_coach_tips_cn,
)
from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.detection import detect_ball_around_release
from basketball_analyzer.models import ComparisonArtifacts, ComparisonResult, SingleAnalysisArtifacts, SingleAnalysisResult
from basketball_analyzer.pose import extract_pose_landmarks
from basketball_analyzer.rendering import overlay_pose_video, render_comparison_video, save_release_frame


class AnalysisService:
    def __init__(self, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig()

    def analyze_single_video(self, input_video: str | Path, output_dir: str | Path) -> SingleAnalysisResult:
        input_path = Path(input_video).resolve()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        pose = extract_pose_landmarks(input_path, self.config)
        if len(pose.detected_landmarks) < 10:
            raise RuntimeError("检测到的姿态帧太少，无法完成分析。")

        arm = choose_shooting_arm(pose.detected_landmarks, self.config.pose.smooth_window)
        filtered_frames, filtered_ids = filter_frames(
            pose.detected_landmarks,
            pose.detected_ids,
            arm,
            self.config.pose.min_box_area,
            self.config.pose.min_visibility,
        )
        if not filtered_frames:
            raise RuntimeError("过滤后无有效姿态帧，无法锁定主体。")

        seg_frames, seg_ids, clip_info = auto_clip_shot_segment(filtered_frames, filtered_ids, arm, pose.metadata.fps, self.config)
        release_frame, release_idx, _, _ = analyze_shot_segment(seg_frames, seg_ids, arm, self.config)
        metrics = compute_metrics(seg_frames, seg_ids, arm, release_idx)
        tips = generate_coach_tips_cn(metrics)
        ball_detection = None
        if self.config.ball_detection.provider and self.config.ball_detection.provider.lower() != "none":
            ball_detection = detect_ball_around_release(input_path, seg_ids, release_frame, self.config)

        overlay_video = output_path / "single_overlay.mp4"
        release_image = output_path / "release_frame.jpg"

        save_release_frame(input_path, release_frame, release_image)
        overlay_pose_video(
            input_path,
            overlay_video,
            pose.frame_to_landmarks,
            pose.metadata.fps,
            (pose.metadata.width, pose.metadata.height),
            self.config,
            metrics=metrics,
            tips_cn=tips,
            clip_info=clip_info,
        )

        result = SingleAnalysisResult(
            input_video=input_path,
            arm=arm,
            metadata=pose.metadata,
            clip_info=clip_info,
            metrics=metrics,
            tips=tips,
            artifacts=SingleAnalysisArtifacts(
                output_dir=output_path,
                overlay_video=overlay_video,
                release_frame_image=release_image,
            ),
            ball_detection=ball_detection,
            debug={
                "detected_pose_frames": len(pose.detected_landmarks),
                "filtered_pose_frames": len(filtered_frames),
            },
        )
        self._write_json(output_path / "analysis_result.json", result.to_dict())
        if ball_detection is not None:
            self._write_json(output_path / "ball_detection_result.json", ball_detection.to_dict())
        return result

    def compare_videos(
        self,
        input_video: str | Path,
        reference_video: str | Path,
        output_dir: str | Path,
    ) -> ComparisonResult:
        input_path = Path(input_video).resolve()
        reference_path = Path(reference_video).resolve()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        user_pose, user_arm, user_release, user_clip = self._analyze_pose_only(input_path)
        reference_pose, reference_arm, reference_release, reference_clip = self._analyze_pose_only(reference_path)
        _ = (user_arm, reference_arm)

        comparison_video = output_path / "comparison_overlay.mp4"
        render_comparison_video(
            input_path,
            comparison_video,
            user_pose,
            reference_pose,
            user_release,
            reference_release,
            self.config,
        )

        result = ComparisonResult(
            input_video=input_path,
            reference_video=reference_path,
            user_release_frame=user_release,
            reference_release_frame=reference_release,
            artifacts=ComparisonArtifacts(output_dir=output_path, comparison_video=comparison_video),
            debug={
                "user_clip_start": user_clip.seg_video_start,
                "user_clip_end": user_clip.seg_video_end,
                "reference_clip_start": reference_clip.seg_video_start,
                "reference_clip_end": reference_clip.seg_video_end,
            },
        )
        self._write_json(output_path / "comparison_result.json", result.to_dict())
        return result

    def _analyze_pose_only(self, video_path: Path):
        pose = extract_pose_landmarks(video_path, self.config)
        if len(pose.detected_landmarks) < 10:
            raise RuntimeError(f"姿态帧过少，无法分析视频：{video_path}")

        arm = choose_shooting_arm(pose.detected_landmarks, self.config.pose.smooth_window)
        filtered_frames, filtered_ids = filter_frames(
            pose.detected_landmarks,
            pose.detected_ids,
            arm,
            self.config.pose.min_box_area,
            self.config.pose.min_visibility,
        )
        if not filtered_frames:
            raise RuntimeError(f"过滤后无有效姿态帧：{video_path}")

        seg_frames, seg_ids, clip_info = auto_clip_shot_segment(filtered_frames, filtered_ids, arm, pose.metadata.fps, self.config)
        release_frame, _, _, _ = analyze_shot_segment(seg_frames, seg_ids, arm, self.config)
        return pose, arm, release_frame, clip_info

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
