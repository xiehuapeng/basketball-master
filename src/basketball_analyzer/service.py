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
from basketball_analyzer.models import (
    BallDetectionEvaluationReport,
    BallDetectionEvaluationRow,
    BallDetectionRun,
    ComparisonArtifacts,
    ComparisonResult,
    ReleaseAnalysis,
    SingleAnalysisArtifacts,
    SingleAnalysisResult,
)
from basketball_analyzer.pose import extract_pose_landmarks
from basketball_analyzer.rendering import overlay_pose_video, render_comparison_video, save_release_frame


class AnalysisService:
    # Empirical thresholds for MVP phase. We prefer "ball just left hand"
    # over the older "pose frame can still see the ball anywhere nearby" rule.
    CONTACT_DISTANCE_THRESHOLD = 1.5
    FAR_DISTANCE_THRESHOLD = 3.0
    EXPANDED_BALL_SEARCH_BEFORE = 30

    def __init__(self, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig()

    def analyze_single_video(self, input_video: str | Path, output_dir: str | Path) -> SingleAnalysisResult:
        input_path = Path(input_video).resolve()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        context = self._prepare_release_context(input_path)
        pose = context["pose"]
        arm = context["arm"]
        seg_ids = context["seg_ids"]
        filtered_ids = context["filtered_ids"]
        clip_info = context["clip_info"]
        release_frame = context["release_frame"]
        metrics = context["metrics"]
        tips = context["tips"]
        release_analysis = ReleaseAnalysis(
            pose_release_frame=release_frame,
            final_release_frame=release_frame,
            source="pose",
            delta_from_pose=0,
            rationale="使用姿态启发式离手帧作为初始结果",
            hand_separation_detected=False,
        )

        ball_detection = None
        if self.config.ball_detection.provider and self.config.ball_detection.provider.lower() != "none":
            ball_detection = detect_ball_around_release(input_path, filtered_ids, release_frame, self.config)
            release_analysis = self._resolve_release_frame_with_ball(
                pose_release_frame=release_frame,
                arm=arm,
                valid_frame_ids=filtered_ids,
                segment_start_frame=seg_ids[0] if seg_ids else None,
                frame_width=pose.metadata.width,
                frame_height=pose.metadata.height,
                frame_to_landmarks=pose.frame_to_landmarks,
                ball_detection=ball_detection,
            )

            if self._should_expand_ball_search(release_analysis, seg_ids):
                expanded_detection = detect_ball_around_release(
                    input_path,
                    filtered_ids,
                    release_frame,
                    self.config,
                    window_before=max(
                        self.config.ball_detection.release_window_before,
                        self.EXPANDED_BALL_SEARCH_BEFORE,
                    ),
                    window_after=self.config.ball_detection.release_window_after,
                )
                expanded_analysis = self._resolve_release_frame_with_ball(
                    pose_release_frame=release_frame,
                    arm=arm,
                    valid_frame_ids=filtered_ids,
                    segment_start_frame=seg_ids[0] if seg_ids else None,
                    frame_width=pose.metadata.width,
                    frame_height=pose.metadata.height,
                    frame_to_landmarks=pose.frame_to_landmarks,
                    ball_detection=expanded_detection,
                )
                if self._is_better_release_analysis(release_analysis, expanded_analysis):
                    ball_detection = expanded_detection
                    release_analysis = expanded_analysis

            if release_analysis.final_release_frame != release_frame:
                metrics, tips = self._compute_metrics_for_frame(context, arm, release_analysis.final_release_frame)

        overlay_video = output_path / "single_overlay.mp4"
        release_image = output_path / "release_frame.jpg"

        save_release_frame(input_path, release_analysis.final_release_frame, release_image)
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
            release_analysis=release_analysis,
            ball_detection=ball_detection,
        )

        result = SingleAnalysisResult(
            input_video=input_path,
            arm=arm,
            metadata=pose.metadata,
            clip_info=clip_info,
            release_analysis=release_analysis,
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
                "filtered_pose_frames": len(context["filtered_frames"]),
                "pose_release_frame": release_frame,
                "final_release_frame": release_analysis.final_release_frame,
            },
        )
        self._write_json(output_path / "analysis_result.json", result.to_dict())
        if ball_detection is not None:
            self._write_json(output_path / "ball_detection_result.json", ball_detection.to_dict())
        return result

    def evaluate_ball_detection_models(
        self,
        input_videos: list[str | Path],
        output_dir: str | Path,
        provider: str,
        model_ids: list[str],
    ) -> BallDetectionEvaluationReport:
        if provider.lower() not in {"roboflow", "huggingface", "hf"}:
            raise ValueError("ball evaluation currently supports provider=roboflow or provider=huggingface/hf")
        if not model_ids:
            raise ValueError("At least one model_id is required for ball evaluation.")

        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        contexts: dict[Path, dict] = {}
        for video in input_videos:
            video_path = Path(video).resolve()
            contexts[video_path] = self._prepare_release_context(video_path)

        rows: list[BallDetectionEvaluationRow] = []
        original_provider = self.config.ball_detection.provider
        original_rf_model = self.config.ball_detection.roboflow_model_id
        original_hf_model = self.config.ball_detection.huggingface_model_id

        try:
            self.config.ball_detection.provider = provider
            for model_id in model_ids:
                if provider.lower() == "roboflow":
                    self.config.ball_detection.roboflow_model_id = model_id
                else:
                    self.config.ball_detection.huggingface_model_id = model_id

                for video_path, context in contexts.items():
                    safe_model_id = self._safe_name(model_id)
                    video_output_dir = output_path / safe_model_id / video_path.stem
                    video_output_dir.mkdir(parents=True, exist_ok=True)

                    result_json = video_output_dir / "ball_detection_result.json"
                    try:
                        run = detect_ball_around_release(
                            video_path,
                            context["filtered_ids"],
                            context["release_frame"],
                            self.config,
                        )
                        if run is None:
                            continue
                        self._write_json(result_json, run.to_dict())
                        row = BallDetectionEvaluationRow(
                            provider=provider,
                            model_id=model_id,
                            input_video=video_path,
                            release_frame=context["release_frame"],
                            sampled_frame_count=run.frame_count,
                            matched_frame_count=run.matched_frame_count,
                            detection_rate=run.detection_rate,
                            release_frame_detected=run.release_frame_detected,
                            nearest_detection_frame=run.nearest_detection_frame,
                            nearest_detection_delta=run.nearest_detection_delta,
                            best_frame=run.best_frame,
                            best_confidence=run.best_confidence,
                            output_json=result_json,
                        )
                    except Exception as exc:
                        row = BallDetectionEvaluationRow(
                            provider=provider,
                            model_id=model_id,
                            input_video=video_path,
                            release_frame=context["release_frame"],
                            sampled_frame_count=0,
                            matched_frame_count=0,
                            detection_rate=0.0,
                            release_frame_detected=False,
                            nearest_detection_frame=None,
                            nearest_detection_delta=None,
                            best_frame=None,
                            best_confidence=None,
                            output_json=result_json,
                            error=str(exc),
                        )
                        self._write_json(
                            result_json,
                            {
                                "provider": provider,
                                "model_id": model_id,
                                "input_video": str(video_path),
                                "error": str(exc),
                            },
                        )
                    rows.append(row)
        finally:
            self.config.ball_detection.provider = original_provider
            self.config.ball_detection.roboflow_model_id = original_rf_model
            self.config.ball_detection.huggingface_model_id = original_hf_model

        report = BallDetectionEvaluationReport(
            provider=provider,
            model_ids=model_ids,
            videos=list(contexts.keys()),
            rows=rows,
        )
        self._write_json(output_path / "ball_evaluation_summary.json", report.to_dict())
        return report

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
        context = self._prepare_release_context(video_path)
        return context["pose"], context["arm"], context["release_frame"], context["clip_info"]

    def _prepare_release_context(self, video_path: Path) -> dict:
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
            raise RuntimeError(f"过滤后没有有效姿态帧：{video_path}")

        seg_frames, seg_ids, clip_info = auto_clip_shot_segment(
            filtered_frames,
            filtered_ids,
            arm,
            pose.metadata.fps,
            self.config,
        )
        release_frame, release_idx, _, _ = analyze_shot_segment(seg_frames, seg_ids, arm, self.config)
        metrics = compute_metrics(seg_frames, seg_ids, arm, release_idx)
        tips = generate_coach_tips_cn(metrics)
        return {
            "pose": pose,
            "arm": arm,
            "filtered_frames": filtered_frames,
            "filtered_ids": filtered_ids,
            "seg_frames": seg_frames,
            "seg_ids": seg_ids,
            "clip_info": clip_info,
            "release_frame": release_frame,
            "release_idx": release_idx,
            "metrics": metrics,
            "tips": tips,
        }

    def _resolve_release_frame_with_ball(
        self,
        pose_release_frame: int,
        arm: str,
        valid_frame_ids: list[int],
        segment_start_frame: int | None,
        frame_width: int,
        frame_height: int,
        frame_to_landmarks: dict[int, list[list[float]]],
        ball_detection: BallDetectionRun | None,
    ) -> ReleaseAnalysis:
        if ball_detection is None or not ball_detection.matched_frames:
            return ReleaseAnalysis(
                pose_release_frame=pose_release_frame,
                final_release_frame=pose_release_frame,
                source="pose",
                delta_from_pose=0,
                rationale="未检测到篮球，保留姿态启发式离手帧",
                hand_separation_detected=False,
            )

        wrist_index = 16 if arm == "right" else 15
        left_shoulder = 11
        right_shoulder = 12
        candidates: list[tuple[int, float]] = []
        for frame_run in ball_detection.matched_frames:
            landmarks = frame_to_landmarks.get(frame_run.frame_index)
            if not landmarks:
                continue

            wrist = landmarks[wrist_index]
            wrist_x = wrist[0] * frame_width
            wrist_y = wrist[1] * frame_height
            shoulder_width = abs(landmarks[right_shoulder][0] - landmarks[left_shoulder][0]) * frame_width + 1e-6

            frame_best_distance = None
            for detection in frame_run.detections:
                dx = detection.x - wrist_x
                dy = detection.y - wrist_y
                distance = ((dx * dx + dy * dy) ** 0.5) / shoulder_width
                if frame_best_distance is None or distance < frame_best_distance:
                    frame_best_distance = distance

            if frame_best_distance is not None:
                candidates.append((frame_run.frame_index, float(frame_best_distance)))

        if not candidates:
            return ReleaseAnalysis(
                pose_release_frame=pose_release_frame,
                final_release_frame=pose_release_frame,
                source="pose",
                delta_from_pose=0,
                rationale="检测到篮球，但无法和投篮手关键点建立稳定对应，保留姿态离手帧",
                hand_separation_detected=False,
            )

        candidates.sort(key=lambda item: item[0])
        valid_frame_set = set(valid_frame_ids)
        exact_pose_match = next((item for item in candidates if item[0] == pose_release_frame), None)
        nearest_candidate_frame, nearest_candidate_distance = min(
            candidates,
            key=lambda item: (abs(item[0] - pose_release_frame), item[1]),
        )

        contact_frames = [frame_index for frame_index, distance in candidates if distance <= self.CONTACT_DISTANCE_THRESHOLD]
        contact_frame = None
        if contact_frames:
            contact_before_pose = [frame_index for frame_index in contact_frames if frame_index <= pose_release_frame]
            contact_frame = contact_before_pose[-1] if contact_before_pose else contact_frames[-1]

        separation_frame = None
        separation_trend = None
        hand_separation_detected = False
        if contact_frame is not None:
            later_far_frames = [
                frame_index
                for frame_index, distance in candidates
                if frame_index > contact_frame and distance >= self.FAR_DISTANCE_THRESHOLD
            ]
            next_valid_frame = next((frame_index for frame_index in valid_frame_ids if frame_index > contact_frame), None)
            if later_far_frames and next_valid_frame is not None:
                separation_frame = next_valid_frame
                separation_trend = "contact-to-flight-transition"
                hand_separation_detected = True

        suspicious_pose = bool(
            segment_start_frame is not None
            and pose_release_frame <= segment_start_frame + 1
        ) or bool(exact_pose_match is not None and exact_pose_match[1] >= self.FAR_DISTANCE_THRESHOLD)

        if contact_frame is not None and separation_frame is not None:
            if suspicious_pose or contact_frame < pose_release_frame:
                return ReleaseAnalysis(
                    pose_release_frame=pose_release_frame,
                    final_release_frame=separation_frame,
                    source="ball-contact-transition",
                    delta_from_pose=separation_frame - pose_release_frame,
                    rationale="找到篮球最后贴手的关键帧，使用下一帧作为最终离手帧",
                    hand_separation_detected=True,
                    contact_frame=contact_frame,
                    separation_frame=separation_frame,
                    separation_trend=separation_trend,
                    ball_candidate_frame=separation_frame,
                    ball_candidate_wrist_distance=next(
                        distance for frame_index, distance in candidates if frame_index == contact_frame
                    ),
                )

        if exact_pose_match is not None and exact_pose_match[1] < self.FAR_DISTANCE_THRESHOLD:
            return ReleaseAnalysis(
                pose_release_frame=pose_release_frame,
                final_release_frame=pose_release_frame,
                source="pose+ball",
                delta_from_pose=0,
                rationale="姿态离手帧附近检测到篮球，且球手距离仍在合理范围内，保留姿态离手帧",
                hand_separation_detected=hand_separation_detected,
                contact_frame=contact_frame,
                separation_frame=separation_frame or pose_release_frame,
                separation_trend=separation_trend,
                ball_candidate_frame=pose_release_frame,
                ball_candidate_wrist_distance=exact_pose_match[1],
            )

        if nearest_candidate_frame not in valid_frame_set:
            return ReleaseAnalysis(
                pose_release_frame=pose_release_frame,
                final_release_frame=pose_release_frame,
                source="pose",
                delta_from_pose=0,
                rationale="篮球候选帧不在有效姿态帧内，保留姿态离手帧",
                hand_separation_detected=hand_separation_detected,
                contact_frame=contact_frame,
                separation_frame=separation_frame,
                separation_trend=separation_trend,
            )

        return ReleaseAnalysis(
            pose_release_frame=pose_release_frame,
            final_release_frame=nearest_candidate_frame,
            source="ball-assisted",
            delta_from_pose=nearest_candidate_frame - pose_release_frame,
            rationale="使用距离姿态离手帧最近的篮球候选帧修正最终离手帧",
            hand_separation_detected=hand_separation_detected,
            contact_frame=contact_frame,
            separation_frame=separation_frame or nearest_candidate_frame,
            separation_trend=separation_trend,
            ball_candidate_frame=nearest_candidate_frame,
            ball_candidate_wrist_distance=nearest_candidate_distance,
        )

    def _compute_metrics_for_frame(self, context: dict, arm: str, release_frame: int):
        if release_frame in context["seg_ids"]:
            release_idx = context["seg_ids"].index(release_frame)
            metrics = compute_metrics(context["seg_frames"], context["seg_ids"], arm, release_idx)
        elif release_frame in context["filtered_ids"]:
            release_idx = context["filtered_ids"].index(release_frame)
            metrics = compute_metrics(context["filtered_frames"], context["filtered_ids"], arm, release_idx)
        else:
            raise RuntimeError(f"修正后的离手帧不在有效姿态帧内：{release_frame}")
        tips = generate_coach_tips_cn(metrics)
        return metrics, tips

    def _should_expand_ball_search(self, release_analysis: ReleaseAnalysis, seg_ids: list[int]) -> bool:
        if not seg_ids:
            return False
        if release_analysis.pose_release_frame <= seg_ids[0] + 1:
            return True
        if release_analysis.ball_candidate_wrist_distance is not None and (
            release_analysis.ball_candidate_wrist_distance >= self.FAR_DISTANCE_THRESHOLD
        ):
            return True
        return False

    @staticmethod
    def _is_better_release_analysis(current: ReleaseAnalysis, candidate: ReleaseAnalysis) -> bool:
        if candidate.source == "ball-contact-transition" and current.source != "ball-contact-transition":
            return True
        if candidate.contact_frame is not None and current.contact_frame is None:
            return True
        if candidate.final_release_frame < current.final_release_frame:
            return True
        return False

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = value.replace("\\", "_").replace("/", "_").replace(":", "_").replace(" ", "_")
        return safe or "model"
