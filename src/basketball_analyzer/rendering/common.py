from __future__ import annotations

from pathlib import Path

import numpy as np

from basketball_analyzer.dependencies import require_cv2, require_pillow
from basketball_analyzer.models import FrameLandmarks

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def load_chinese_font(size: int = 26):
    _, _, ImageFont = require_pillow()
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def put_chinese_text(frame_bgr, lines: list[str], x: int = 20, y: int = 20, line_h: int = 34, font=None):
    if not lines:
        return frame_bgr

    cv2 = require_cv2()
    Image, ImageDraw, _ = require_pillow()

    if font is None:
        font = load_chinese_font(26)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        yy += line_h

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame, landmarks: FrameLandmarks, connections, color=(0, 255, 0), min_vis: float = 0.3, line_thickness: int = 2, point_radius: int = 4):
    cv2 = require_cv2()
    height, width = frame.shape[:2]

    for start, end in connections:
        x1, y1, _, v1 = landmarks[start]
        x2, y2, _, v2 = landmarks[end]
        if v1 < min_vis or v2 < min_vis:
            continue
        p1 = (int(x1 * width), int(y1 * height))
        p2 = (int(x2 * width), int(y2 * height))
        cv2.line(frame, p1, p2, color, line_thickness)

    for x, y, _, visibility in landmarks:
        if visibility < min_vis:
            continue
        point = (int(x * width), int(y * height))
        cv2.circle(frame, point, point_radius, color, -1)

