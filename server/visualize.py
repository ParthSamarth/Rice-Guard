"""
server/visualize.py

detection/predict_yolo.py:YoloDetector.detect() returns structured boxes
only -- it never renders an annotated image (nothing under detection/ or
pipeline/ does). The Android app needs a "detection visualization" image
(project brief sections 1, 7, 12), so this module draws one: PIL boxes +
labels over the original photo, using ONLY the bbox/class/confidence values
the pipeline already computed. This is presentation logic (Application
Layer), not new detection logic -- it never re-runs or reinterprets the
model's output.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_BOX_COLOR = (46, 139, 87)      # matches evaluation/plot_style.py CLASS_COLORS-adjacent green
_TEXT_BG = (26, 36, 32)
_TEXT_FG = (255, 255, 255)


def _load_font(size: int):
    for candidate in ("arial.ttf", "DejaVuSans-Bold.ttf", "seguisb.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_detection_image(image_path: Path, detections: list[dict], out_path: Path) -> Path:
    """detections: the pipeline's own `yolo.detections` list
    ([{"class", "confidence", "bbox_xyxy": [x0,y0,x1,y1]}, ...]).
    Draws nothing and just copies the source image if `detections` is empty
    (the "no detection -> full-image CNN path" case) -- an empty-but-present
    image is still useful for the Android UI to show "no regions found"
    rather than a broken image link."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw_im = im.copy()

    draw = ImageDraw.Draw(draw_im)
    font_size = max(14, min(draw_im.size) // 40)
    font = _load_font(font_size)

    for det in detections:
        x0, y0, x1, y1 = det["bbox_xyxy"]
        label = f"{det['class']} {det['confidence']:.2f}"
        line_w = max(2, min(draw_im.size) // 300)
        draw.rectangle([x0, y0, x1, y1], outline=_BOX_COLOR, width=line_w)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        pad = 4
        label_y0 = max(0, y0 - text_h - 2 * pad)
        draw.rectangle([x0, label_y0, x0 + text_w + 2 * pad, label_y0 + text_h + 2 * pad], fill=_TEXT_BG)
        draw.text((x0 + pad, label_y0 + pad), label, fill=_TEXT_FG, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_im.save(out_path, format="JPEG", quality=90)
    return out_path
