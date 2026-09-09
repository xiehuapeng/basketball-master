from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from basketball_analyzer.config import BallDetectionConfig
from basketball_analyzer.models import BallDetection


class BallDetectionProvider(ABC):
    def __init__(self, config: BallDetectionConfig):
        self.config = config
        self.target_labels = {label.lower() for label in config.target_labels}

    @abstractmethod
    def detect(self, image: Any) -> list[BallDetection]:
        raise NotImplementedError

    def _filter_detections(self, detections: list[BallDetection]) -> list[BallDetection]:
        return [
            detection
            for detection in detections
            if detection.label.lower() in self.target_labels and detection.confidence >= self.config.confidence_threshold
        ]


class RoboflowBallDetector(BallDetectionProvider):
    def __init__(self, config: BallDetectionConfig):
        super().__init__(config)
        try:
            from inference_sdk import InferenceHTTPClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Roboflow remote detection requires `pip install -e .[remote]`."
            ) from exc

        api_key = config.roboflow_api_key or os.getenv("ROBOFLOW_API_KEY", "")
        model_id = config.roboflow_model_id or os.getenv("ROBOFLOW_MODEL_ID", "")
        if not api_key or not model_id:
            raise RuntimeError("Roboflow provider requires ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID.")

        self.model_id = model_id
        self.client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)

    def detect(self, image: Any) -> list[BallDetection]:
        result = self.client.infer(image, model_id=self.model_id)
        predictions = result.get("predictions", [])
        detections = [
            BallDetection(
                label=str(item.get("class", "")),
                confidence=float(item.get("confidence", 0.0)),
                x=float(item.get("x", 0.0)),
                y=float(item.get("y", 0.0)),
                width=float(item.get("width", 0.0)),
                height=float(item.get("height", 0.0)),
                source="roboflow",
            )
            for item in predictions
        ]
        return self._filter_detections(detections)


class HuggingFaceBallDetector(BallDetectionProvider):
    def __init__(self, config: BallDetectionConfig):
        super().__init__(config)
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face remote detection requires `pip install -e .[remote]`."
            ) from exc

        api_key = config.huggingface_api_key or os.getenv("HF_TOKEN", "")
        model_id = config.huggingface_model_id or os.getenv("HF_MODEL_ID", "")
        if not api_key or not model_id:
            raise RuntimeError("Hugging Face provider requires HF_TOKEN and HF_MODEL_ID.")

        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_id = model_id

    def detect(self, image: Any) -> list[BallDetection]:
        result = self.client.object_detection(image, model=self.model_id)
        detections: list[BallDetection] = []
        for item in result:
            box = item.get("box", {}) if isinstance(item, dict) else getattr(item, "box", None)
            label = item.get("label", "") if isinstance(item, dict) else getattr(item, "label", "")
            score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
            if hasattr(box, "xmin"):
                xmin = float(box.xmin)
                ymin = float(box.ymin)
                xmax = float(box.xmax)
                ymax = float(box.ymax)
            else:
                xmin = float(box.get("xmin", 0.0))
                ymin = float(box.get("ymin", 0.0))
                xmax = float(box.get("xmax", 0.0))
                ymax = float(box.get("ymax", 0.0))
            detections.append(
                BallDetection(
                    label=str(label),
                    confidence=float(score),
                    x=(xmin + xmax) / 2,
                    y=(ymin + ymax) / 2,
                    width=max(0.0, xmax - xmin),
                    height=max(0.0, ymax - ymin),
                    source="huggingface",
                )
            )
        return self._filter_detections(detections)


class LocalYoloBallDetector(BallDetectionProvider):
    """Free, offline detector based on Ultralytics YOLO (COCO class `sports ball`).

    No API key required. Uses a lower confidence threshold plus a small-object
    friendly inference size, because a basketball is usually tiny in shooting videos.
    """

    def __init__(self, config: BallDetectionConfig):
        super().__init__(config)
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Local ball detection requires `pip install ultralytics`."
            ) from exc

        from pathlib import Path

        model_path = config.local_model_path or "yolov8n.pt"
        # A missing custom checkpoint should not break analysis; the stock COCO
        # model still knows "sports ball" and downloads automatically.
        if "/" in model_path.replace("\\", "/") and not Path(model_path).exists():
            model_path = "yolov8n.pt"
        self.model = YOLO(model_path)
        self.imgsz = int(config.local_imgsz or 1280)
        self.confidence = float(config.local_confidence_threshold or 0.12)

    def detect(self, image: Any) -> list[BallDetection]:
        results = self.model.predict(image, imgsz=self.imgsz, conf=self.confidence, verbose=False)
        detections: list[BallDetection] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                label = str(names.get(int(box.cls[0]), ""))
                xyxy = box.xyxy[0].tolist()
                xmin, ymin, xmax, ymax = (float(v) for v in xyxy)
                detections.append(
                    BallDetection(
                        label=label,
                        confidence=float(box.conf[0]),
                        x=(xmin + xmax) / 2,
                        y=(ymin + ymax) / 2,
                        width=max(0.0, xmax - xmin),
                        height=max(0.0, ymax - ymin),
                        source="local-yolo",
                    )
                )
        # Local YOLO uses its own confidence gate; only apply the label filter here.
        return [d for d in detections if d.label.lower() in self.target_labels and d.confidence >= self.confidence]


def build_ball_detector(config: BallDetectionConfig) -> BallDetectionProvider | None:
    provider = (config.provider or "none").strip().lower()
    if provider in {"", "none"}:
        return None
    if provider in {"local", "yolo", "local-yolo"}:
        return LocalYoloBallDetector(config)
    if provider == "roboflow":
        return RoboflowBallDetector(config)
    if provider in {"hf", "huggingface"}:
        return HuggingFaceBallDetector(config)
    raise ValueError(f"Unsupported ball detection provider: {config.provider}")
