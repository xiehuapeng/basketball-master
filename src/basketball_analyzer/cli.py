from __future__ import annotations

import argparse
import json
from pathlib import Path

from basketball_analyzer.service import AnalysisService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Basketball shooting analysis toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Analyze one shooting video")
    single.add_argument("--input", required=True, help="Path to the input video")
    single.add_argument("--output", required=True, help="Directory for outputs")
    _add_ball_detection_args(single)

    compare = subparsers.add_parser("compare", help="Compare shooter video with a reference video")
    compare.add_argument("--input", required=True, help="Path to the user video")
    compare.add_argument("--reference", required=True, help="Path to the reference video")
    compare.add_argument("--output", required=True, help="Directory for outputs")

    ball_eval = subparsers.add_parser("ball-eval", help="Batch evaluate remote basketball detection models")
    ball_eval.add_argument("--provider", choices=["roboflow", "huggingface", "hf"], required=True, help="Remote provider")
    ball_eval.add_argument("--videos", nargs="+", required=True, help="One or more local video paths")
    ball_eval.add_argument("--models", nargs="+", required=True, help="One or more remote model ids")
    ball_eval.add_argument("--output", required=True, help="Directory for evaluation outputs")

    return parser


def _add_ball_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ball-provider",
        choices=["none", "roboflow", "huggingface", "hf"],
        default="none",
        help="Optional remote basketball detection provider",
    )
    parser.add_argument("--roboflow-model-id", default=None, help="Roboflow hosted model ID")
    parser.add_argument("--huggingface-model-id", default=None, help="Hugging Face object detection model ID")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = AnalysisService()

    if args.command == "single":
        service.config.ball_detection.provider = args.ball_provider
        if args.roboflow_model_id:
            service.config.ball_detection.roboflow_model_id = args.roboflow_model_id
        if args.huggingface_model_id:
            service.config.ball_detection.huggingface_model_id = args.huggingface_model_id
        result = service.analyze_single_video(args.input, args.output)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "compare":
        result = service.compare_videos(args.input, args.reference, args.output)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "ball-eval":
        report = service.evaluate_ball_detection_models(
            input_videos=args.videos,
            output_dir=args.output,
            provider=args.provider,
            model_ids=args.models,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
