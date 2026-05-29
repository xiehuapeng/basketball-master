from __future__ import annotations


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install with `pip install opencv-python`.") from exc
    return cv2


def require_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required. Install with `pip install mediapipe` in a supported Python environment."
        ) from exc
    return mp


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Pillow is required. Install with `pip install pillow`.") from exc
    return Image, ImageDraw, ImageFont

