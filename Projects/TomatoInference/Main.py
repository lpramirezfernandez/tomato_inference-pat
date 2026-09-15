"""
=====================================================================
  TOMATO INFERENCE - Kivy MVP (desktop + Android/APK)
  Detection model:    "tomato-splxa/1"   (Roboflow Serverless API)
  Segmentation model: "unamba_tesis/6"   (Roboflow Serverless API)
=====================================================================

Kivy port of the OpenCV desktop app. ALL the business logic (Roboflow
calls, detection<->segmentation reconciliation, dedup, dashboard
drawing) is reused verbatim from the previous version -- render_dashboard()
still returns a plain numpy/OpenCV BGR image, which is simply converted
to a Kivy Texture for display. Only the navigation/UI shell (buttons,
camera, file picker, drag-to-select) is rebuilt with native Kivy
widgets, since raw cv2 windows + mouse callbacks don't exist on mobile.

Requirements (desktop testing):
    pip install kivy opencv-python numpy python-dotenv inference-sdk plyer

Setup:
    .env file with ROBOFLOW_API_KEY in the same folder.

Run on desktop:
    python main.py

Build APK:
    See buildozer.spec in this same folder, then:
        buildozer -v android debug
    (Buildozer requires Linux or WSL -- it does not run natively on Windows.)
"""

import os
import sys
import time
import threading
import tempfile
import unicodedata

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

from kivy.config import Config
Config.set("graphics", "width", "420")
Config.set("graphics", "height", "740")

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line, Rotate, PushMatrix, PopMatrix
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.animation import Animation
from kivy.utils import platform as kivy_platform

try:
    from kivy.uix.camera import Camera
    _CAMERA_AVAILABLE = True
except Exception:
    _CAMERA_AVAILABLE = False

try:
    from plyer import filechooser
    _FILECHOOSER_AVAILABLE = True
except Exception:
    _FILECHOOSER_AVAILABLE = False

load_dotenv()

# =====================================================================
#  CONFIG (unchanged from the desktop version)
# =====================================================================

DETECTION_MODEL_ID = "tomato-splxa/1"
SEGMENTATION_MODEL_ID = "unamba_tesis/6"
API_URL = "https://serverless.roboflow.com"
CONFIDENCE_THRESHOLD = 0.6

MATCH_IOU_THRESHOLD = 0.3
DEDUPE_IOU_THRESHOLD = 0.4
RECONCILIATION_MODEL_THRESHOLD = 0.5
STRONG_SEGMENTATION_OVERRIDE = 0.92

PALETTE_BGR = [
    (92, 59, 255), (160, 229, 0), (32, 176, 255),
    (255, 140, 91), (255, 109, 200), (209, 209, 0),
]
ACCENT_COLOR = (255, 190, 40)
MANUAL_ACCENT = (255, 90, 220)
PANEL_BG = (28, 22, 18)
BG_TOP = (36, 26, 20)
BG_BOTTOM = (14, 10, 8)
TEXT_PRIMARY = (245, 245, 245)
TEXT_MUTED = (170, 170, 170)
DANGER = (60, 60, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX

# Kivy-side colors (RGBA 0..1), matching the same palette for the shell UI
KV_BG = (14 / 255, 10 / 255, 8 / 255, 1)
KV_PANEL = (28 / 255, 22 / 255, 18 / 255, 1)
KV_ACCENT = (255 / 255, 190 / 255, 40 / 255, 1)
KV_MANUAL = (255 / 255, 90 / 255, 220 / 255, 1)
KV_TEXT = (0.96, 0.96, 0.96, 1)
KV_MUTED = (0.67, 0.67, 0.67, 1)
KV_DANGER = (1, 0.35, 0.35, 1)

TRANSLATION_MAP = {
    "maduro": "ripe", "madura": "ripe", "rojo": "ripe", "roja": "ripe", "madurez": "ripe",
    "pinton": "half_ripe", "pintona": "half_ripe", "medio_maduro": "half_ripe",
    "media_madurez": "half_ripe", "envero": "half_ripe", "pintado": "half_ripe",
    "verde": "unripe", "inmaduro": "unripe", "inmadura": "unripe", "verdoso": "unripe",
    "tomate": "tomato", "fruto": "tomato",
    "ripe": "ripe", "half_ripe": "half_ripe", "unripe": "unripe", "tomato": "tomato",
}
_KEYWORD_PRIORITY = ["medio_maduro", "media_madurez", "pintona", "pinton",
                      "envero", "pintado", "inmaduro", "inmadura", "verdoso",
                      "verde", "maduro", "madura", "madurez", "rojo", "roja",
                      "fruto", "tomate"]


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def to_english(label: str) -> str:
    normalized = _strip_accents(label.strip().lower()).replace(" ", "_")
    if normalized in TRANSLATION_MAP:
        return TRANSLATION_MAP[normalized]
    for keyword in _KEYWORD_PRIORITY:
        if keyword in normalized:
            return TRANSLATION_MAP[keyword]
    return label.replace("_", " ").title()


# =====================================================================
#  DRAWING HELPERS (unchanged -- same OpenCV drawing code as before)
# =====================================================================

def _draw_rounded_rect(img, pt1, pt2, color, radius, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx_, cy_ in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx_, cy_), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _draw_vertical_gradient(img, pt1, pt2, color_top, color_bottom):
    x1, y1 = pt1
    x2, y2 = pt2
    height = max(1, y2 - y1)
    for i in range(height):
        t = i / height
        color = tuple(int(color_top[c] * (1 - t) + color_bottom[c] * t) for c in range(3))
        cv2.line(img, (x1, y1 + i), (x2, y1 + i), color, 1)


def _draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=16, gap_len=10):
    x1, y1 = pt1
    x2, y2 = pt2

    def dashed_line(p1, p2):
        p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
        length = np.linalg.norm(p2 - p1)
        if length == 0:
            return
        direction = (p2 - p1) / length
        dist = 0.0
        draw = True
        while dist < length:
            seg_len = dash_len if draw else gap_len
            seg_end = min(dist + seg_len, length)
            if draw:
                a = tuple((p1 + direction * dist).astype(int))
                b = tuple((p1 + direction * seg_end).astype(int))
                cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
            dist = seg_end
            draw = not draw

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def _place_without_overlap(occupied, x1, y1, x2, y2, step, max_shift=600):
    cur_y1, cur_y2 = y1, y2
    shifted = 0

    def collides(a1, a2):
        return any(not (x2 <= ox1 or x1 >= ox2 or a2 <= oy1 or a1 >= oy2) for (ox1, oy1, ox2, oy2) in occupied)

    while collides(cur_y1, cur_y2) and shifted < max_shift:
        cur_y1 += step
        cur_y2 += step
        shifted += step

    occupied.append((x1, cur_y1, x2, cur_y2))
    return cur_y1, cur_y2


# =====================================================================
#  ROBOFLOW CLIENT (unchanged)
# =====================================================================

def get_client() -> InferenceHTTPClient:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not found in environment / .env file.")
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def run_model(image, model_id: str) -> tuple:
    client = get_client()
    config = InferenceConfiguration(confidence_threshold=CONFIDENCE_THRESHOLD, iou_threshold=0.5)
    client.configure(config)
    start = time.time()
    result = client.infer(image, model_id=model_id)
    elapsed = time.time() - start
    return result, elapsed


def run_inference(image) -> tuple:
    det_result, det_elapsed = run_model(image, DETECTION_MODEL_ID)
    seg_result, seg_elapsed = run_model(image, SEGMENTATION_MODEL_ID)
    result = {
        "det_predictions": det_result.get("predictions", []),
        "seg_predictions": seg_result.get("predictions", []),
    }
    return result, det_elapsed + seg_elapsed


# =====================================================================
#  SEGMENTATION CROSS-CHECK / RECONCILIATION (unchanged)
# =====================================================================

def bbox_of(pred: dict) -> tuple:
    points = pred.get("points")
    if points:
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        return min(xs), min(ys), max(xs), max(ys)
    cx, cy = pred.get("x", 0), pred.get("y", 0)
    pw, ph = pred.get("width", 0), pred.get("height", 0)
    return cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_segmentation(det_box, seg_preds, used_idx):
    best_j, best_iou = None, 0.0
    for j, spred in enumerate(seg_preds):
        if j in used_idx:
            continue
        score = iou(det_box, bbox_of(spred))
        if score > best_iou:
            best_iou, best_j = score, j
    if best_j is not None and best_iou >= MATCH_IOU_THRESHOLD:
        spred = seg_preds[best_j]
        label = to_english(spred.get("class") or spred.get("class_name", "object"))
        return label, spred.get("confidence", 0), best_j
    return None, 0.0, None


def _reconcile(model_label, model_confidence, seg_label, seg_conf, agrees):
    if model_label is None and seg_label is None:
        return None, 0.0, False, None
    if model_label is None:
        return seg_label, seg_conf, True, "detection found nothing here; using segmentation"

    if model_confidence >= RECONCILIATION_MODEL_THRESHOLD:
        if seg_label is None:
            return model_label, model_confidence, False, None
        if agrees:
            return (model_label, model_confidence, False,
                    f"segmentation check OK: {seg_label} ({seg_conf * 100:.0f}%)")
        if seg_conf >= STRONG_SEGMENTATION_OVERRIDE:
            return (seg_label, seg_conf, True,
                    f"reconciled: detection said {model_label} ({model_confidence * 100:.0f}%) but "
                    f"segmentation strongly disagrees ({seg_label} {seg_conf * 100:.0f}%) "
                    f"- likely a detection misclassification")
        return (model_label, model_confidence, False,
                f"segmentation check disagrees: suggests {seg_label} ({seg_conf * 100:.0f}%) "
                f"- detection kept (conf {model_confidence * 100:.0f}% >= "
                f"{RECONCILIATION_MODEL_THRESHOLD * 100:.0f}%, disagreement not strong enough)")

    if seg_label is not None:
        return (seg_label, seg_conf, True,
                f"reconciled: detection confidence too low ({model_confidence * 100:.0f}% < "
                f"{RECONCILIATION_MODEL_THRESHOLD * 100:.0f}%), using segmentation ({seg_conf * 100:.0f}%)")

    return (model_label, model_confidence, False,
            f"detection confidence low ({model_confidence * 100:.0f}%) and segmentation found "
            f"nothing there - keeping detection result")


def classify_crop_with_models(crop: np.ndarray):
    det_label, det_conf = None, 0.0
    try:
        det_result, _ = run_model(crop, DETECTION_MODEL_ID)
        det_preds = det_result.get("predictions", [])
        if det_preds:
            top = max(det_preds, key=lambda p: p.get("confidence", 0))
            det_label = to_english(top.get("class") or top.get("class_name", "object"))
            det_conf = top.get("confidence", 0)
    except Exception:
        pass

    seg_label, seg_conf = None, 0.0
    try:
        seg_result, _ = run_model(crop, SEGMENTATION_MODEL_ID)
        seg_preds = seg_result.get("predictions", [])
        if seg_preds:
            top = max(seg_preds, key=lambda p: p.get("confidence", 0))
            seg_label = to_english(top.get("class") or top.get("class_name", "object"))
            seg_conf = top.get("confidence", 0)
    except Exception:
        pass

    agrees = (seg_label == det_label) if (seg_label is not None and det_label is not None) else None
    return _reconcile(det_label, det_conf, seg_label, seg_conf, agrees)


def _dedupe_entries(entries: list) -> list:
    ordered = sorted(entries, key=lambda e: e["confidence"], reverse=True)
    kept = []
    for entry in ordered:
        if any(iou(entry["box"], k["box"]) >= DEDUPE_IOU_THRESHOLD for k in kept):
            continue
        kept.append(entry)
    return kept


# =====================================================================
#  DASHBOARD RENDERING (unchanged -- still returns a plain numpy BGR image)
# =====================================================================

def render_dashboard(image, result: dict, elapsed_seconds: float = 0.0) -> tuple:
    det_predictions = list(result.get("det_predictions", []))
    seg_predictions = list(result.get("seg_predictions", []))
    manual_predictions = list(result.get("manual_predictions", []))

    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not open image: {image}")
    else:
        img = image.copy()

    h, w = img.shape[:2]
    overlay = img.copy()

    line_thickness = max(2, round(min(w, h) * 0.004))
    corner_len = max(12, round(min(w, h) * 0.025))
    font_scale = max(0.5, min(w, h) * 0.0012)
    font_thickness = max(1, round(font_scale * 2))

    class_colors, class_counts = {}, {}
    color_idx = 0
    occupied_rects = []
    used_seg_idx = set()

    entries = []

    for det_pred in det_predictions:
        model_label = to_english(det_pred.get("class") or det_pred.get("class_name", "object"))
        model_confidence = det_pred.get("confidence", 0)
        det_box = bbox_of(det_pred)

        seg_label, seg_conf, matched_idx = _match_segmentation(det_box, seg_predictions, used_seg_idx)
        if matched_idx is not None:
            used_seg_idx.add(matched_idx)
        agrees = (seg_label == model_label) if seg_label is not None else None

        label, confidence, reconciled, source_note = _reconcile(
            model_label, model_confidence, seg_label, seg_conf, agrees,
        )

        entries.append({
            "box": det_box, "label": label, "confidence": confidence,
            "reconciled": reconciled, "source_note": source_note, "agrees": agrees,
            "kind": "detection",
        })

    for j, spred in enumerate(seg_predictions):
        if j in used_seg_idx:
            continue
        label = to_english(spred.get("class") or spred.get("class_name", "object"))
        confidence = spred.get("confidence", 0)
        entries.append({
            "box": bbox_of(spred), "label": label, "confidence": confidence,
            "reconciled": False, "source_note": None, "agrees": None,
            "kind": "segmentation_only",
        })

    for mpred in manual_predictions:
        entries.append({
            "box": bbox_of(mpred), "label": mpred.get("class"), "confidence": mpred.get("confidence", 0),
            "reconciled": False, "source_note": None, "agrees": None,
            "kind": "manual",
        })

    entries = _dedupe_entries(entries)

    for entry in entries:
        kind = entry["kind"]
        label = entry["label"] or "inconclusive"
        confidence = entry["confidence"]
        reconciled = entry["reconciled"]
        source_note = entry["source_note"]
        agrees = entry["agrees"]

        x1, y1, x2, y2 = (int(v) for v in entry["box"])
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)

        is_manual = (kind == "manual")
        is_seg_only = (kind == "segmentation_only")

        if is_manual:
            box_color = MANUAL_ACCENT
        elif is_seg_only:
            box_color = ACCENT_COLOR
        else:
            if label not in class_colors:
                class_colors[label] = PALETTE_BGR[color_idx % len(PALETTE_BGR)]
                color_idx += 1
            box_color = class_colors[label]

        if label not in class_colors:
            class_colors[label] = box_color
        class_counts[label] = class_counts.get(label, 0) + 1

        if is_manual or is_seg_only:
            _draw_dashed_rect(img, (x1c, y1c), (x2c, y2c), box_color, thickness=line_thickness + 1)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cl = corner_len
            for (px, py), (dxh, dyh), (dxv, dyv) in [
                ((x1, y1), (1, 0), (0, 1)), ((x2, y1), (-1, 0), (0, 1)),
                ((x1, y2), (1, 0), (0, -1)), ((x2, y2), (-1, 0), (0, -1)),
            ]:
                cv2.line(img, (px, py), (px + dxh * cl, py + dyh * cl), box_color, line_thickness + 1, cv2.LINE_AA)
                cv2.line(img, (px, py), (px + dxv * cl, py + dyv * cl), box_color, line_thickness + 1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, line_thickness, cv2.LINE_AA)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ch = max(4, corner_len // 3)
            cv2.line(img, (cx - ch, cy), (cx + ch, cy), box_color, 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy - ch), (cx, cy + ch), box_color, 1, cv2.LINE_AA)

        if is_manual:
            prefix = "MANUAL: "
        elif is_seg_only:
            prefix = "SEGMENTATION ONLY: "
        elif reconciled:
            prefix = "RECONCILED: "
        else:
            prefix = ""
        text = f"{prefix}{label}" if (is_manual and confidence == 0) else f"{prefix}{label}  {confidence * 100:.0f}%"

        (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, font_thickness)
        pad_x, pad_y = 12, 8
        tag_x1, tag_x2 = x1c, x1c + tw + pad_x * 2
        raw_tag_y1 = max(0, y1c - th - pad_y * 2 - 6)
        raw_tag_y2 = raw_tag_y1 + th + pad_y * 2
        tag_y1, tag_y2 = _place_without_overlap(
            occupied_rects, tag_x1, raw_tag_y1, tag_x2, raw_tag_y2, step=raw_tag_y2 - raw_tag_y1 + 6,
        )

        tag_layer = img.copy()
        _draw_rounded_rect(tag_layer, (tag_x1, tag_y1), (tag_x2, tag_y2), box_color, radius=(tag_y2 - tag_y1) // 2)
        cv2.addWeighted(tag_layer, 0.92, img, 0.08, 0, dst=img)
        cv2.putText(img, text, (tag_x1 + pad_x, tag_y2 - pad_y - 2), FONT, font_scale, (15, 15, 20),
                    font_thickness, cv2.LINE_AA)

        if source_note is not None:
            if reconciled:
                note_color = ACCENT_COLOR
            elif agrees:
                note_color = (160, 229, 0)
            else:
                note_color = (110, 110, 255)
            check_scale = font_scale * 0.5
            (ctw, cth), _ = cv2.getTextSize(source_note, FONT, check_scale, 1)
            cpad_x, cpad_y = 6, 4
            raw_c_y1 = tag_y2 + 6
            raw_c_y2 = raw_c_y1 + cth + cpad_y * 2
            c_x1 = tag_x1
            c_x2 = tag_x1 + ctw + cpad_x * 2
            c_y1, c_y2 = _place_without_overlap(
                occupied_rects, c_x1, raw_c_y1, c_x2, raw_c_y2, step=raw_c_y2 - raw_c_y1 + 4,
            )
            c_y1, c_y2 = min(c_y1, h - (raw_c_y2 - raw_c_y1)), min(c_y2, h)

            note_bg = img.copy()
            _draw_rounded_rect(note_bg, (c_x1, c_y1), (c_x2, c_y2), (18, 16, 15), radius=6)
            cv2.addWeighted(note_bg, 0.72, img, 0.28, 0, dst=img)
            cv2.putText(img, source_note, (c_x1 + cpad_x, c_y2 - cpad_y - 2), FONT, check_scale,
                        note_color, 1, cv2.LINE_AA)

    annotated = cv2.addWeighted(overlay, 0.16, img, 0.84, 0)

    bar_h = max(56, int(h * 0.07))
    canvas = np.zeros((h + bar_h, w, 3), np.uint8)
    _draw_vertical_gradient(canvas, (0, 0), (w, bar_h), (22, 16, 14), (44, 32, 26))
    cv2.line(canvas, (0, bar_h), (w, bar_h), ACCENT_COLOR, 2, cv2.LINE_AA)

    cv2.putText(canvas, "TOMATO INFERENCE", (24, int(bar_h * 0.68)), FONT, font_scale * 1.15,
                TEXT_PRIMARY, font_thickness + 1, cv2.LINE_AA)
    subtitle = (f"det: {len(det_predictions)} | seg: {len(seg_predictions)} | "
                f"conf >= {CONFIDENCE_THRESHOLD:.2f} | {elapsed_seconds * 1000:.0f} ms")
    (sub_w, _), _ = cv2.getTextSize(subtitle, FONT, font_scale * 0.55, 1)
    cv2.putText(canvas, subtitle, (w - sub_w - 24, int(bar_h * 0.68)), FONT, font_scale * 0.55,
                TEXT_MUTED, 1, cv2.LINE_AA)
    for i in range(0, w, 40):
        cv2.line(canvas, (i, bar_h - 4), (i, bar_h), ACCENT_COLOR, 1, cv2.LINE_AA)

    canvas[bar_h:bar_h + h, 0:w] = annotated
    h_total = h + bar_h

    pad = 16
    row_h = int(28 * max(1.0, font_scale * 1.6))
    legend_items = sorted(class_counts.items(), key=lambda kv: -kv[1])
    panel_h = pad * 2 + row_h * (len(legend_items) + 1)
    panel_w = max(int(w * 0.26), 220)
    px1, py2 = pad, h_total - pad
    px2, py1 = px1 + panel_w, py2 - panel_h

    panel_layer = canvas.copy()
    _draw_rounded_rect(panel_layer, (px1, py1), (px2, py2), PANEL_BG, radius=14)
    cv2.addWeighted(panel_layer, 0.78, canvas, 0.22, 0, dst=canvas)
    _draw_rounded_rect(canvas, (px1, py1), (px2, py2), ACCENT_COLOR, radius=14, thickness=1)

    cv2.putText(canvas, f"DETECTIONS: {len(entries)}", (px1 + 16, py1 + row_h - 4), FONT,
                font_scale * 0.62, TEXT_PRIMARY, font_thickness, cv2.LINE_AA)
    cv2.line(canvas, (px1 + 12, py1 + row_h + 4), (px2 - 12, py1 + row_h + 4), (70, 60, 55), 1, cv2.LINE_AA)

    for idx, (label, count) in enumerate(legend_items, start=1):
        color = class_colors[label]
        y = py1 + row_h * (idx + 1)
        cv2.circle(canvas, (px1 + 24, y - 6), 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{label}: {count}", (px1 + 42, y), FONT, font_scale * 0.55, TEXT_MUTED, 1, cv2.LINE_AA)

    fully_empty = (len(entries) == 0)
    note = "Neither the detection nor the segmentation model found anything in this photo." if fully_empty else ""
    return canvas, note, fully_empty


# =====================================================================
#  numpy (BGR) <-> Kivy Texture bridge
# =====================================================================

def bgr_to_texture(img_bgr: np.ndarray) -> Texture:
    """Converts an OpenCV BGR numpy image into a Kivy Texture ready to
    assign to an Image widget's `texture` property."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 0)  # Kivy texture origin is bottom-left
    h, w = rgb.shape[:2]
    texture = Texture.create(size=(w, h), colorfmt="rgb")
    texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    return texture


def camera_texture_to_bgr(camera_widget) -> np.ndarray:
    """Grabs the current frame from a Kivy Camera widget as an OpenCV BGR
    array. Handles both colorfmt='rgba' (most providers) and
    colorfmt='bgr' (the opencv provider used on Windows/desktop), since
    assuming RGBA unconditionally corrupts or crashes on the opencv
    backend."""
    texture = camera_widget.texture
    if texture is None:
        raise RuntimeError("Camera has not produced a frame yet -- wait a moment and try again.")

    w, h = texture.size
    colorfmt = (texture.colorfmt or "rgba").lower()
    pixels = texture.pixels
    channels_by_fmt = {"luminance": 1, "rgb": 3, "bgr": 3, "rgba": 4, "bgra": 4}
    channels = channels_by_fmt.get(colorfmt, 4)

    arr = np.frombuffer(pixels, dtype=np.uint8)
    # Some Kivy/provider combinations normalize .pixels to RGBA regardless
    # of texture.colorfmt -- trust the actual buffer size over the label.
    if arr.size == w * h * 4:
        channels = 4
    elif arr.size == w * h * 3:
        channels = 3
    arr = arr.reshape(h, w, channels)

    if channels == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR if colorfmt == "bgra" else cv2.COLOR_RGBA2BGR)
    elif channels == 3:
        bgr = arr.copy() if colorfmt == "bgr" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise RuntimeError(f"Unsupported camera texture format: {colorfmt}")

    bgr = cv2.flip(bgr, 0)  # texture origin is bottom-left
    return bgr


# =====================================================================
#  Small reusable Kivy widgets
# =====================================================================

def styled_button(text, bg=KV_PANEL, color=KV_TEXT, font_size="16sp", bold=True, **kwargs):
    btn = Button(
        text=text, background_normal="", background_down="",
        background_color=bg, color=color, font_size=font_size, bold=bold,
        **kwargs,
    )
    return btn


class TitleLabel(Label):
    pass


class Spinner(Widget):
    """Small rotating arc used on the WORKING screen, echoing the
    OpenCV version's spinning-ellipse loading indicator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.angle = 0
        with self.canvas:
            PushMatrix()
            self._rot = Rotate(angle=0, origin=self.center)
            Color(*KV_ACCENT)
            self._line = Line(circle=(self.center_x, self.center_y, 24, 0, 270), width=3)
            PopMatrix()
        self.bind(pos=self._update, size=self._update)
        Clock.schedule_interval(self._spin, 1 / 30)

    def _update(self, *args):
        self._rot.origin = self.center
        self._line.circle = (self.center_x, self.center_y, 24, 0, 270)

    def _spin(self, dt):
        self._rot.angle = (self._rot.angle + 6) % 360


class DragOverlay(Widget):
    """Transparent overlay used on the MANUAL SELECT screen to draw the
    live rubber-band rectangle while the user drags."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start = None
        self.current = None

    def set_rect(self, start, current):
        self.start = start
        self.current = current
        self.canvas.clear()
        if start is None or current is None:
            return
        x1, y1 = start
        x2, y2 = current
        with self.canvas:
            Color(*KV_MANUAL, 0.25)
            Rectangle(pos=(min(x1, x2), min(y1, y2)), size=(abs(x2 - x1), abs(y2 - y1)))
            Color(*KV_MANUAL, 1)
            Line(rectangle=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)), width=1.5)


# =====================================================================
#  Popup helper
# =====================================================================

def show_popup(title, message, color=KV_ACCENT, on_dismiss=None):
    content = BoxLayout(orientation="vertical", padding=16, spacing=12)
    content.add_widget(Label(text=message, color=KV_MUTED, font_size="14sp", halign="center", valign="middle"))
    ok_btn = styled_button("OK", bg=color, color=(0.05, 0.05, 0.08, 1), size_hint=(1, None), height=44)
    content.add_widget(ok_btn)

    popup = Popup(
        title=title, title_color=color, content=content,
        size_hint=(0.85, 0.4), separator_color=color,
        background_color=KV_BG,
    )
    ok_btn.bind(on_release=lambda *a: popup.dismiss())
    if on_dismiss:
        popup.bind(on_dismiss=lambda *a: on_dismiss())
    popup.open()
    return popup


# =====================================================================
#  SCREENS
# =====================================================================

class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical", padding=24, spacing=16)
        with root.canvas.before:
            Color(*KV_BG)
            self._bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=self._sync_bg, size=self._sync_bg)

        root.add_widget(Label(text="TOMATO INFERENCE", font_size="26sp", bold=True,
                               color=KV_TEXT, size_hint=(1, 0.12)))
        root.add_widget(Label(text="Detection + segmentation dashboard", font_size="13sp",
                               color=KV_MUTED, size_hint=(1, 0.06)))
        root.add_widget(Label(text=f"detection: {DETECTION_MODEL_ID}\nsegmentation: {SEGMENTATION_MODEL_ID}",
                               font_size="11sp", color=KV_MUTED, size_hint=(1, 0.1)))

        spacer = Widget(size_hint=(1, 0.08))
        root.add_widget(spacer)

        load_btn = styled_button("LOAD IMAGE FROM DISK", bg=KV_ACCENT, color=(0.05, 0.05, 0.08, 1),
                                  size_hint=(1, 0.12))
        load_btn.bind(on_release=lambda *a: self.manager.get_screen("menu")._load_image())
        root.add_widget(load_btn)

        cam_btn = styled_button("OPEN CAMERA & CAPTURE", size_hint=(1, 0.12))
        cam_btn.bind(on_release=lambda *a: setattr(self.manager, "current", "camera"))
        root.add_widget(cam_btn)

        quit_btn = styled_button("QUIT", size_hint=(1, 0.12))
        quit_btn.bind(on_release=lambda *a: App.get_running_app().stop())
        root.add_widget(quit_btn)

        root.add_widget(Widget(size_hint=(1, 0.2)))
        self.add_widget(root)

    def _sync_bg(self, widget, *args):
        self._bg.pos = widget.pos
        self._bg.size = widget.size

    def _load_image(self):
        if not _FILECHOOSER_AVAILABLE:
            show_popup("Missing dependency", "plyer is required for the file picker (pip install plyer).",
                       KV_DANGER)
            return
        try:
            filechooser.open_file(on_selection=self._on_file_selected,
                                   filters=[["Images", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]])
        except Exception as e:
            show_popup("File picker error", str(e), KV_DANGER)

    def _on_file_selected(self, selection):
        if not selection:
            return
        path = selection[0]
        Clock.schedule_once(lambda dt: App.get_running_app().start_inference(path))


class CameraScreen(Screen):
    """
    Two camera paths:
      - Desktop (Windows/Linux/Mac): drives cv2.VideoCapture directly in a
        background thread with CAP_DSHOW on Windows, then polls frames via
        Clock -- opening a webcam can take several seconds or hang on
        Windows' MSMF backend, and doing that on the main thread freezes
        the whole app. This mirrors the original pure-OpenCV app, which
        worked reliably.
      - Android: uses Kivy's own Camera widget, since cv2.VideoCapture
        generally can't reach the Android camera hardware.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_android = (kivy_platform == "android")

        self.camera_widget = None   # Android path
        self.cap = None             # Desktop path
        self.frame_event = None     # Desktop path
        self.last_frame = None      # Desktop path

        root = FloatLayout()
        with root.canvas.before:
            Color(*KV_BG)
            self._bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=self._sync_bg, size=self._sync_bg)

        self.camera_holder = FloatLayout(size_hint=(1, 0.85), pos_hint={"x": 0, "y": 0.15})
        root.add_widget(self.camera_holder)

        self.preview = Image(allow_stretch=True, keep_ratio=True,
                              size_hint=(1, 1), pos_hint={"x": 0, "y": 0})
        self.status_label = Label(text="Opening camera...", color=KV_MUTED, font_size="13sp",
                                   size_hint=(1, 1), pos_hint={"x": 0, "y": 0})

        bar = BoxLayout(size_hint=(1, 0.15), pos_hint={"x": 0, "y": 0}, spacing=8, padding=8)
        capture_btn = styled_button("CAPTURE PHOTO", bg=KV_ACCENT, color=(0.05, 0.05, 0.08, 1))
        capture_btn.bind(on_release=lambda *a: self._capture())
        back_btn = styled_button("BACK")
        back_btn.bind(on_release=lambda *a: self._go_back())
        bar.add_widget(capture_btn)
        bar.add_widget(back_btn)
        root.add_widget(bar)

        self.add_widget(root)

    def _sync_bg(self, widget, *args):
        self._bg.pos = widget.pos
        self._bg.size = widget.size

    def on_enter(self):
        self.camera_holder.clear_widgets()
        if self.is_android:
            self._enter_android_camera()
        else:
            self._enter_desktop_camera()

    def on_leave(self):
        if self.is_android:
            if self.camera_widget is not None:
                self.camera_widget.play = False
                self.camera_widget = None
        else:
            if self.frame_event is not None:
                self.frame_event.cancel()
                self.frame_event = None
            cap = self.cap
            self.cap = None
            self.last_frame = None
            if cap is not None:
                threading.Thread(target=cap.release, daemon=True).start()
        self.camera_holder.clear_widgets()

    def _go_back(self):
        self.manager.current = "menu"

    # --- Android path (Kivy Camera widget) ---------------------------

    def _enter_android_camera(self):
        if not _CAMERA_AVAILABLE:
            show_popup("Camera unavailable", "Kivy's camera widget could not be loaded on this platform.",
                       KV_DANGER, on_dismiss=lambda: setattr(self.manager, "current", "menu"))
            return
        try:
            self.camera_widget = Camera(index=0, play=True, resolution=(640, 480))
            self.camera_holder.add_widget(self.camera_widget)
        except Exception as e:
            show_popup("Camera error", str(e), KV_DANGER, on_dismiss=lambda: setattr(self.manager, "current", "menu"))

    # --- Desktop path (cv2.VideoCapture opened off the main thread) --

    def _enter_desktop_camera(self):
        self.status_label.text = "Opening camera..."
        self.camera_holder.add_widget(self.preview)
        self.camera_holder.add_widget(self.status_label)
        threading.Thread(target=self._open_camera_thread, daemon=True).start()

    def _open_camera_thread(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else 0
        cap = cv2.VideoCapture(0, backend)
        ok = cap.isOpened()
        Clock.schedule_once(lambda dt: self._on_camera_opened(cap, ok))

    def _on_camera_opened(self, cap, ok):
        if self.manager.current != "camera":
            # user already navigated away while the camera was opening
            cap.release()
            return
        if not ok:
            cap.release()
            show_popup(
                "Camera error",
                "Could not open the camera. On Windows, check Settings > Privacy & security > "
                "Camera > 'Let desktop apps access your camera', and make sure no other app "
                "(Zoom, Teams, etc.) is currently using it.",
                KV_DANGER, on_dismiss=lambda: setattr(self.manager, "current", "menu"),
            )
            return
        self.cap = cap
        self.status_label.text = ""
        self.frame_event = Clock.schedule_interval(self._update_frame, 1 / 20)

    def _update_frame(self, dt):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self.last_frame = frame
        self.preview.texture = bgr_to_texture(frame)

    # --- Capture (shared) ---------------------------------------------

    def _capture(self):
        if self.is_android:
            if self.camera_widget is None or self.camera_widget.texture is None:
                show_popup("No frame yet", "The camera hasn't produced a frame yet -- wait a moment and try again.",
                           KV_DANGER)
                return
            try:
                frame = camera_texture_to_bgr(self.camera_widget)
            except Exception as e:
                show_popup("Capture error", str(e), KV_DANGER)
                return
        else:
            if self.last_frame is None:
                show_popup("No frame yet", "The camera hasn't produced a frame yet -- wait a moment and try again.",
                           KV_DANGER)
                return
            frame = self.last_frame.copy()

        raw_path = os.path.join(tempfile.gettempdir(), "tomato_capture.jpg")
        cv2.imwrite(raw_path, frame)
        App.get_running_app().start_inference(frame)


class WorkingScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = FloatLayout()
        with root.canvas.before:
            Color(*KV_BG)
            self._bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=self._sync_bg, size=self._sync_bg)

        panel = BoxLayout(orientation="vertical", size_hint=(0.8, 0.3),
                           pos_hint={"center_x": 0.5, "center_y": 0.5}, spacing=10, padding=16)
        with panel.canvas.before:
            Color(*KV_PANEL)
            self._panel_bg = Rectangle(pos=panel.pos, size=panel.size)
        panel.bind(pos=self._sync_panel, size=self._sync_panel)

        self.spinner = Spinner(size_hint=(1, 0.5))
        panel.add_widget(self.spinner)
        self.message_label = Label(text="Running both models...", color=KV_TEXT, font_size="15sp", bold=True)
        panel.add_widget(self.message_label)
        self.sub_label = Label(text="", color=KV_MUTED, font_size="11sp")
        panel.add_widget(self.sub_label)

        root.add_widget(panel)
        self.add_widget(root)

    def _sync_bg(self, widget, *args):
        self._bg.pos = widget.pos
        self._bg.size = widget.size

    def _sync_panel(self, widget, *args):
        self._panel_bg.pos = widget.pos
        self._panel_bg.size = widget.size

    def set_message(self, message, sub=""):
        self.message_label.text = message
        self.sub_label.text = sub


class ResultScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical")
        with root.canvas.before:
            Color(*KV_BG)
            self._bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=self._sync_bg, size=self._sync_bg)

        self.result_image = Image(size_hint=(1, 0.87), allow_stretch=True, keep_ratio=True)
        root.add_widget(self.result_image)

        bar = BoxLayout(size_hint=(1, 0.13), spacing=8, padding=8)
        manual_btn = styled_button("MANUAL CHECK", bg=KV_MANUAL, color=(0.05, 0.05, 0.08, 1))
        manual_btn.bind(on_release=lambda *a: setattr(self.manager, "current", "manual_select"))
        back_btn = styled_button("BACK TO MENU")
        back_btn.bind(on_release=lambda *a: setattr(self.manager, "current", "menu"))
        bar.add_widget(manual_btn)
        bar.add_widget(back_btn)
        root.add_widget(bar)

        self.add_widget(root)

    def _sync_bg(self, widget, *args):
        self._bg.pos = widget.pos
        self._bg.size = widget.size

    def set_result(self, img_bgr: np.ndarray):
        self.result_image.texture = bgr_to_texture(img_bgr)


class ManualSelectScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_img = None
        self._touch_start = None

        root = BoxLayout(orientation="vertical")
        with root.canvas.before:
            Color(*KV_BG)
            self._bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=self._sync_bg, size=self._sync_bg)

        root.add_widget(Label(text="MANUAL CHECK - drag on the photo to mark a missed tomato",
                               size_hint=(1, 0.08), color=KV_TEXT, font_size="13sp"))

        self.stack = FloatLayout(size_hint=(1, 0.79))
        self.photo_image = Image(allow_stretch=True, keep_ratio=True)
        self.overlay = DragOverlay()
        self.stack.add_widget(self.photo_image)
        self.stack.add_widget(self.overlay)
        root.add_widget(self.stack)

        bar = BoxLayout(size_hint=(1, 0.13), spacing=8, padding=8)
        cancel_btn = styled_button("CANCEL")
        cancel_btn.bind(on_release=lambda *a: setattr(self.manager, "current", "result"))
        bar.add_widget(cancel_btn)
        root.add_widget(bar)

        self.add_widget(root)

        self.stack.bind(on_touch_down=self._on_touch_down)
        self.stack.bind(on_touch_move=self._on_touch_move)
        self.stack.bind(on_touch_up=self._on_touch_up)

    def _sync_bg(self, widget, *args):
        self._bg.pos = widget.pos
        self._bg.size = widget.size

    def on_enter(self):
        app = App.get_running_app()
        self.base_img = app.current_image_array()
        if self.base_img is not None:
            self.photo_image.texture = bgr_to_texture(self.base_img)

    def _image_rect_in_widget(self):
        """Returns the on-screen (x, y, w, h) of the actually-rendered
        image inside self.photo_image, accounting for keep_ratio
        letterboxing -- mirrors the old _fit_into() logic."""
        if self.base_img is None:
            return None
        img_h, img_w = self.base_img.shape[:2]
        wx, wy = self.photo_image.pos
        ww, wh = self.photo_image.size
        scale = min(ww / img_w, wh / img_h)
        disp_w, disp_h = img_w * scale, img_h * scale
        off_x = wx + (ww - disp_w) / 2
        off_y = wy + (wh - disp_h) / 2
        return off_x, off_y, disp_w, disp_h, scale

    def _on_touch_down(self, widget, touch):
        if not self.stack.collide_point(*touch.pos):
            return False
        self._touch_start = touch.pos
        self.overlay.set_rect(touch.pos, touch.pos)
        return True

    def _on_touch_move(self, widget, touch):
        if self._touch_start is None:
            return False
        self.overlay.set_rect(self._touch_start, touch.pos)
        return True

    def _on_touch_up(self, widget, touch):
        if self._touch_start is None:
            return False
        start = self._touch_start
        end = touch.pos
        self._touch_start = None
        self.overlay.set_rect(None, None)
        self._finalize_selection(start, end)
        return True

    def _finalize_selection(self, start, end):
        rect = self._image_rect_in_widget()
        if rect is None:
            return
        off_x, off_y, disp_w, disp_h, scale = rect

        x1, x2 = sorted([start[0], end[0]])
        y1, y2 = sorted([start[1], end[1]])
        # clamp to the displayed image area
        x1 = max(off_x, min(x1, off_x + disp_w))
        x2 = max(off_x, min(x2, off_x + disp_w))
        y1 = max(off_y, min(y1, off_y + disp_h))
        y2 = max(off_y, min(y2, off_y + disp_h))

        if (x2 - x1) < 12 or (y2 - y1) < 12:
            return  # too small, ignore accidental taps

        # Kivy y grows upward from bottom-left; numpy rows grow downward
        # from top-left, and the widget may be taller than the image.
        img_h, img_w = self.base_img.shape[:2]
        ox1 = int((x1 - off_x) / scale)
        ox2 = int((x2 - off_x) / scale)
        oy1_top = img_h - int((y2 - off_y) / scale)
        oy2_top = img_h - int((y1 - off_y) / scale)
        ox1, ox2 = max(0, ox1), min(img_w, ox2)
        oy1_top, oy2_top = max(0, oy1_top), min(img_h, oy2_top)

        crop = self.base_img[oy1_top:oy2_top, ox1:ox2]
        App.get_running_app().start_manual_check(crop, ox1, oy1_top, ox2, oy2_top)


# =====================================================================
#  APP
# =====================================================================

class TomatoInferenceApp(App):
    def build(self):
        Window.clearcolor = KV_BG
        self.sm = ScreenManager(transition=FadeTransition(duration=0.12))

        self._current_image = None      # path (str) or ndarray of the last analyzed photo
        self._current_result = None     # {"det_predictions":..., "seg_predictions":...}
        self._current_elapsed = 0.0
        self._manual_predictions = []

        self.sm.add_widget(MenuScreen(name="menu"))
        self.sm.add_widget(CameraScreen(name="camera"))
        self.sm.add_widget(WorkingScreen(name="working"))
        self.sm.add_widget(ResultScreen(name="result"))
        self.sm.add_widget(ManualSelectScreen(name="manual_select"))

        return self.sm

    # -----------------------------------------------------------------
    #  Helpers used by screens
    # -----------------------------------------------------------------

    def current_image_array(self):
        if self._current_image is None:
            return None
        if isinstance(self._current_image, str):
            return cv2.imread(self._current_image)
        return self._current_image

    # -----------------------------------------------------------------
    #  Inference pipeline (runs on a background thread; UI updates are
    #  scheduled back onto the main thread via Clock)
    # -----------------------------------------------------------------

    def start_inference(self, image):
        working = self.sm.get_screen("working")
        working.set_message("Running both models...", f"{DETECTION_MODEL_ID}  +  {SEGMENTATION_MODEL_ID}")
        self.sm.current = "working"
        threading.Thread(target=self._run_inference_thread, args=(image,), daemon=True).start()

    def _run_inference_thread(self, image):
        try:
            result, elapsed = run_inference(image)
        except RuntimeError as e:
            Clock.schedule_once(lambda dt: self._on_inference_error("Configuration error", str(e)))
            return
        except Exception as e:
            Clock.schedule_once(lambda dt: self._on_inference_error("Inference error", str(e)))
            return

        result["manual_predictions"] = []

        try:
            canvas, note, fully_empty = render_dashboard(image, result, elapsed_seconds=elapsed)
        except Exception as e:
            Clock.schedule_once(lambda dt: self._on_inference_error("Rendering error", str(e)))
            return

        Clock.schedule_once(lambda dt: self._on_inference_done(image, result, elapsed, canvas, note, fully_empty))

    def _on_inference_error(self, title, message):
        show_popup(title, message, KV_DANGER, on_dismiss=lambda: setattr(self.sm, "current", "menu"))

    def _on_inference_done(self, image, result, elapsed, canvas, note, fully_empty):
        self._current_image = image
        self._current_result = result
        self._current_elapsed = elapsed
        self._manual_predictions = []

        if fully_empty:
            show_popup("Nothing found", note, KV_ACCENT, on_dismiss=lambda: setattr(self.sm, "current", "menu"))
            return

        cv2.imwrite("tomato_result.png", canvas)
        self.sm.get_screen("result").set_result(canvas)
        self.sm.current = "result"

    # -----------------------------------------------------------------
    #  Manual check pipeline
    # -----------------------------------------------------------------

    def start_manual_check(self, crop, ox1, oy1, ox2, oy2):
        working = self.sm.get_screen("working")
        working.set_message("Analyzing selection...", "running both models on the marked region")
        self.sm.current = "working"
        threading.Thread(target=self._run_manual_thread, args=(crop, ox1, oy1, ox2, oy2), daemon=True).start()

    def _run_manual_thread(self, crop, ox1, oy1, ox2, oy2):
        try:
            label, conf, reconciled, note = classify_crop_with_models(crop)
        except Exception as e:
            Clock.schedule_once(lambda dt: self._on_manual_error(str(e)))
            return
        Clock.schedule_once(lambda dt: self._on_manual_done(label, conf, ox1, oy1, ox2, oy2))

    def _on_manual_error(self, message):
        show_popup("Manual check error", message, KV_DANGER, on_dismiss=lambda: setattr(self.sm, "current", "result"))

    def _on_manual_done(self, label, conf, ox1, oy1, ox2, oy2):
        self._manual_predictions.append({
            "class": label if label else "inconclusive",
            "confidence": conf,
            "x": (ox1 + ox2) / 2,
            "y": (oy1 + oy2) / 2,
            "width": ox2 - ox1,
            "height": oy2 - oy1,
            "_manual_check": True,
        })
        self._rebuild_result()
        self.sm.current = "result"

    def _rebuild_result(self):
        base_result = self._current_result or {"det_predictions": [], "seg_predictions": []}
        combined = dict(base_result)
        combined["manual_predictions"] = self._manual_predictions
        try:
            canvas, _, _ = render_dashboard(self._current_image, combined, elapsed_seconds=self._current_elapsed)
        except Exception as e:
            show_popup("Rendering error", str(e), KV_DANGER)
            return
        cv2.imwrite("tomato_result.png", canvas)
        self.sm.get_screen("result").set_result(canvas)


if __name__ == "__main__":
    TomatoInferenceApp().run()