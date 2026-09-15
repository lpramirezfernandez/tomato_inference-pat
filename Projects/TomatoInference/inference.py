"""
=====================================================================
  TOMATO INFERENCE - Detection Suite
  Detection model:    "tomato-splxa/1"   (Roboflow Serverless API)
  Segmentation model: "unamba_tesis/6"   (Roboflow Serverless API)
=====================================================================

A single-window, fully mouse-driven OpenCV application. Everything
happens inside ONE window that switches between views:

  MENU    -> Load Image / Open Camera / Quit buttons
  CAMERA  -> live feed with on-screen CAPTURE and BACK buttons
             (SPACE also captures, ESC also goes back)
  WORKING -> spinner while both API calls run
  RESULT  -> detection dashboard with a MANUAL CHECK + BACK button

Requirements:
    pip install inference-sdk python-dotenv opencv-python numpy

Setup:
    The API key is loaded automatically from a .env file
    (variable ROBOFLOW_API_KEY) located in the same folder.

Usage:
    python inference.py
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

load_dotenv()

DETECTION_MODEL_ID = "tomato-splxa/1"
SEGMENTATION_MODEL_ID = "unamba_tesis/6"
API_URL = "https://serverless.roboflow.com"
CONFIDENCE_THRESHOLD = 0.6

# --- Segmentation cross-check / reconciliation settings ---
# The detection model is the primary source of truth. Segmentation is a
# second opinion: it only overrides detection when detection is either
# unsure (below RECONCILIATION_MODEL_THRESHOLD) or clearly wrong (a very
# confident disagreement, >= STRONG_SEGMENTATION_OVERRIDE).
MATCH_IOU_THRESHOLD = 0.3            # boxes overlapping at least this much = "the same object"
DEDUPE_IOU_THRESHOLD = 0.4           # final safety net: collapse any remaining overlapping boxes into one
RECONCILIATION_MODEL_THRESHOLD = 0.5
STRONG_SEGMENTATION_OVERRIDE = 0.92

WINDOW_NAME = "Tomato Inference"
WIN_W, WIN_H = 900, 620
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900

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

STATE_MENU = "menu"
STATE_CAMERA = "camera"
STATE_WORKING = "working"
STATE_RESULT = "result"
STATE_POPUP = "popup"
STATE_MANUAL_SELECT = "manual_select"

# Common Spanish ripeness/tomato terms -> English, in case the
# segmentation model returns Spanish class names. Anything not listed
# here just gets underscores turned into spaces and title-cased.
TRANSLATION_MAP = {
    "maduro": "ripe", "madura": "ripe", "rojo": "ripe", "roja": "ripe", "madurez": "ripe",
    "pinton": "half_ripe", "pintona": "half_ripe", "medio_maduro": "half_ripe",
    "media_madurez": "half_ripe", "envero": "half_ripe", "pintado": "half_ripe",
    "verde": "unripe", "inmaduro": "unripe", "inmadura": "unripe", "verdoso": "unripe",
    "tomate": "tomato", "fruto": "tomato",
    "ripe": "ripe", "half_ripe": "half_ripe", "unripe": "unripe", "tomato": "tomato",
}

# Order matters: longer/more specific keywords are checked before shorter
# ones (e.g. "pinton" before a generic fallback) so compound labels like
# "tomate_pinton" or "fruto maduro" still resolve correctly.
_KEYWORD_PRIORITY = ["medio_maduro", "media_madurez", "pintona", "pinton",
                      "envero", "pintado", "inmaduro", "inmadura", "verdoso",
                      "verde", "maduro", "madura", "madurez", "rojo", "roja",
                      "fruto", "tomate"]


def _strip_accents(text: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def to_english(label: str) -> str:
    normalized = _strip_accents(label.strip().lower()).replace(" ", "_")

    if normalized in TRANSLATION_MAP:
        return TRANSLATION_MAP[normalized]

    # Compound / unexpected label (e.g. "tomate_pinton", "fruto maduro"):
    # search for a known Spanish keyword anywhere inside it.
    for keyword in _KEYWORD_PRIORITY:
        if keyword in normalized:
            return TRANSLATION_MAP[keyword]

    return label.replace("_", " ").title()


# =====================================================================
#  DRAWING HELPERS
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


def _text_centered(img, text, cx, cy, scale, color, thickness):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2)), FONT, scale, color, thickness, cv2.LINE_AA)


def _fit_into(img, max_w, max_h):
    """Scales img down (never up) to fit inside max_w x max_h, returns (scaled_img, scale)."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def _draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=16, gap_len=10):
    """Draws a rectangle using dashed lines (used to visually mark
    segmentation-only / manual boxes as distinct from real detection-model
    boxes)."""
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


def _place_without_overlap(occupied: list, x1: int, y1: int, x2: int, y2: int,
                            step: int, max_shift: int = 600) -> tuple:
    """
    Nudges a label rectangle straight down until it no longer overlaps any
    previously placed rectangle in `occupied`, then registers the final
    position. Keeps tags and audit-note text from being drawn on top of
    each other when detections sit close together.
    """
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
#  ROBOFLOW CLIENT
# =====================================================================

def get_client() -> InferenceHTTPClient:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not found in environment / .env file.")
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def run_model(image, model_id: str) -> tuple:
    """image: file path (str) or a BGR numpy array."""
    client = get_client()
    config = InferenceConfiguration(confidence_threshold=CONFIDENCE_THRESHOLD, iou_threshold=0.5)
    client.configure(config)
    start = time.time()
    result = client.infer(image, model_id=model_id)
    elapsed = time.time() - start
    return result, elapsed


def run_inference(image) -> tuple:
    """Runs BOTH models on the full image. Returns (result_dict, elapsed_seconds)."""
    det_result, det_elapsed = run_model(image, DETECTION_MODEL_ID)
    seg_result, seg_elapsed = run_model(image, SEGMENTATION_MODEL_ID)
    result = {
        "det_predictions": det_result.get("predictions", []),
        "seg_predictions": seg_result.get("predictions", []),
    }
    return result, det_elapsed + seg_elapsed


# =====================================================================
#  SEGMENTATION CROSS-CHECK / RECONCILIATION
#  (mirrors the old HSV color-analysis logic, but the "second opinion"
#   now comes from a real segmentation model instead of pixel colors)
# =====================================================================

def bbox_of(pred: dict) -> tuple:
    """Returns (x1, y1, x2, y2), from polygon points if present
    (segmentation), otherwise from the center/width/height box (detection)."""
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


def _match_segmentation(det_box: tuple, seg_preds: list, used_idx: set):
    """Finds the best unused segmentation prediction overlapping det_box.
    Returns (seg_label_english, seg_confidence, matched_index) or
    (None, 0.0, None) if nothing matches closely enough."""
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
    """
    Same reconciliation rule as before: detection wins outright once it's
    confident enough (>= RECONCILIATION_MODEL_THRESHOLD), UNLESS
    segmentation disagrees so strongly (>= STRONG_SEGMENTATION_OVERRIDE)
    that detection was very likely wrong. Below the threshold, detection
    only wins if segmentation has nothing to say.

    Returns (label, confidence, reconciled: bool, source_note: str|None).
    """
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
                    f"— likely a detection misclassification")
        return (model_label, model_confidence, False,
                f"segmentation check disagrees: suggests {seg_label} ({seg_conf * 100:.0f}%) "
                f"— detection kept (conf {model_confidence * 100:.0f}% >= "
                f"{RECONCILIATION_MODEL_THRESHOLD * 100:.0f}%, disagreement not strong enough)")

    if seg_label is not None:
        return (seg_label, seg_conf, True,
                f"reconciled: detection confidence too low ({model_confidence * 100:.0f}% < "
                f"{RECONCILIATION_MODEL_THRESHOLD * 100:.0f}%), using segmentation ({seg_conf * 100:.0f}%)")

    return (model_label, model_confidence, False,
            f"detection confidence low ({model_confidence * 100:.0f}%) and segmentation found "
            f"nothing there — keeping detection result")


def classify_crop_with_models(crop: np.ndarray):
    """Runs both models on a small cropped region (used by manual
    selection) and returns the reconciled (label, confidence, reconciled,
    source_note) for that crop."""
    det_label, det_conf = None, 0.0
    try:
        det_result, _ = run_model(crop, DETECTION_MODEL_ID)
        det_preds = det_result.get("predictions", [])
        if det_preds:
            top = max(det_preds, key=lambda p: p.get("confidence", 0))
            det_label, det_conf = to_english(top.get("class") or top.get("class_name", "object")), top.get("confidence", 0)
    except Exception:
        pass

    seg_label, seg_conf = None, 0.0
    try:
        seg_result, _ = run_model(crop, SEGMENTATION_MODEL_ID)
        seg_preds = seg_result.get("predictions", [])
        if seg_preds:
            top = max(seg_preds, key=lambda p: p.get("confidence", 0))
            seg_label, seg_conf = to_english(top.get("class") or top.get("class_name", "object")), top.get("confidence", 0)
    except Exception:
        pass

    agrees = (seg_label == det_label) if (seg_label is not None and det_label is not None) else None
    return _reconcile(det_label, det_conf, seg_label, seg_conf, agrees)


# =====================================================================
#  FILE PICKER (native OS dialog window, not console)
# =====================================================================

def pick_image_file() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
    )
    root.destroy()
    return path


# =====================================================================
#  DETECTION DASHBOARD RENDERING
# =====================================================================

def _dedupe_entries(entries: list) -> list:
    """
    Final safety net: even after detection<->segmentation matching, boxes
    for the same real object can end up slightly misaligned (a
    segmentation polygon's bounding box often isn't shaped like the
    detection model's box) and slip past MATCH_IOU_THRESHOLD as two
    separate entries. This collapses any remaining significantly
    overlapping boxes down to just the highest-confidence one, so every
    real tomato ends up with exactly one box on screen.
    """
    ordered = sorted(entries, key=lambda e: e["confidence"], reverse=True)
    kept = []
    for entry in ordered:
        if any(iou(entry["box"], k["box"]) >= DEDUPE_IOU_THRESHOLD for k in kept):
            continue
        kept.append(entry)
    return kept


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

    # Build a unified list of "entries to draw": detection-anchored entries
    # (each cross-checked against segmentation), then leftover
    # segmentation-only entries (objects only segmentation found), then
    # manual entries.
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

        # --- Audit line: shows what the segmentation check found/decided,
        #     only when there's something to say (silent otherwise). ---
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

    # --- Header bar lives in its OWN strip, above the image (never overlaps detections) ---
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
#  SINGLE-WINDOW APPLICATION (state machine, all mouse-driven)
# =====================================================================

class TomatoApp:
    def __init__(self):
        self.state = STATE_MENU
        self.buttons = []
        self.hover_idx = None
        self.canvas = None

        self.cap = None
        self.last_camera_frame = None

        self.result_canvas = None
        self.working_message = ""
        self.working_sub = ""

        self.popup_title = ""
        self.popup_message = ""
        self.popup_color = ACCENT_COLOR
        self.popup_return_state = STATE_MENU

        self.pending_job = None

        # --- Manual bounding-box selection (for regions both models missed) ---
        self.current_image = None
        self.current_result = None
        self.current_elapsed = 0.0
        self.manual_predictions = []
        self.manual_base_img = None
        self.manual_scale = 1.0
        self.manual_offset = (0, 0)
        self.drag_start = None
        self.drag_current = None

    # -----------------------------------------------------------------
    #  Button helpers
    # -----------------------------------------------------------------

    def _set_buttons(self, specs):
        self.buttons = [
            {"label": s[0], "action": s[1], "rect": (s[2], s[3], s[4], s[5])}
            for s in specs
        ]

    def _draw_buttons(self, canvas, primary_action=None):
        for i, btn in enumerate(self.buttons):
            x1, y1, x2, y2 = btn["rect"]
            hovered = (i == self.hover_idx)
            is_primary = (btn["action"] == primary_action)

            if hovered:
                fill = (255, 255, 255)
                outline = (255, 255, 255)
                text_color = (15, 15, 20)
            elif is_primary:
                fill = ACCENT_COLOR
                outline = ACCENT_COLOR
                text_color = (15, 15, 20)
            else:
                fill = (60, 46, 36)
                outline = ACCENT_COLOR
                text_color = TEXT_PRIMARY

            _draw_rounded_rect(canvas, (x1, y1), (x2, y2), fill, radius=14)
            _draw_rounded_rect(canvas, (x1, y1), (x2, y2), outline, radius=14, thickness=2)
            _text_centered(canvas, btn["label"], (x1 + x2) // 2, (y1 + y2) // 2, 0.62, text_color, 2)

    # -----------------------------------------------------------------
    #  MENU state
    # -----------------------------------------------------------------

    def _render_menu(self):
        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        _draw_vertical_gradient(canvas, (0, 0), (WIN_W, WIN_H), BG_TOP, BG_BOTTOM)

        cv2.putText(canvas, "TOMATO INFERENCE", (60, 100), FONT, 1.5, TEXT_PRIMARY, 3, cv2.LINE_AA)
        cv2.putText(canvas, "AI-powered tomato detection dashboard", (64, 135), FONT, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.line(canvas, (60, 155), (WIN_W - 60, 155), ACCENT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"detection model: {DETECTION_MODEL_ID}", (60, 185), FONT, 0.5, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"segmentation model: {SEGMENTATION_MODEL_ID}", (60, 210), FONT, 0.5, TEXT_MUTED, 1, cv2.LINE_AA)

        btn_w, btn_h = WIN_W - 120, 74
        self._set_buttons([
            ("LOAD IMAGE FROM DISK", "load", 60, 250, 60 + btn_w, 250 + btn_h),
            ("OPEN CAMERA & CAPTURE", "camera", 60, 340, 60 + btn_w, 340 + btn_h),
            ("QUIT", "quit", 60, 430, 60 + btn_w, 430 + btn_h),
        ])
        self._draw_buttons(canvas, primary_action="load")

        footer = "click a button to continue   |   ESC quits the app"
        cv2.putText(canvas, footer, (60, WIN_H - 24), FONT, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)

        self.canvas = canvas

    def _enter_menu(self):
        self.state = STATE_MENU
        self.hover_idx = None
        self._release_camera()
        self._render_menu()

    # -----------------------------------------------------------------
    #  CAMERA state
    # -----------------------------------------------------------------

    def _enter_camera(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else 0
        self.cap = cv2.VideoCapture(0, backend)
        if not self.cap or not self.cap.isOpened():
            self._enter_popup("Camera error", "Could not access the camera.", DANGER, STATE_MENU)
            return
        self.state = STATE_CAMERA
        self.hover_idx = None

    def _release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _render_camera_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self._enter_popup("Camera error", "Failed to read frame from camera.", DANGER, STATE_MENU)
            return
        self.last_camera_frame = frame.copy()

        h, w = frame.shape[:2]
        display, scale = _fit_into(frame, WIN_W, WIN_H - 90)
        dh, dw = display.shape[:2]

        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        _draw_vertical_gradient(canvas, (0, 0), (WIN_W, WIN_H), BG_TOP, BG_BOTTOM)

        off_x, off_y = (WIN_W - dw) // 2, 70
        canvas[off_y:off_y + dh, off_x:off_x + dw] = display

        for gx in (off_x + dw // 3, off_x + 2 * dw // 3):
            cv2.line(canvas, (gx, off_y), (gx, off_y + dh), (90, 80, 75), 1, cv2.LINE_AA)
        for gy in (off_y + dh // 3, off_y + 2 * dh // 3):
            cv2.line(canvas, (off_x, gy), (off_x + dw, gy), (90, 80, 75), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (off_x, off_y), (off_x + dw, off_y + dh), ACCENT_COLOR, 2, cv2.LINE_AA)

        bar_layer = canvas.copy()
        _draw_vertical_gradient(bar_layer, (0, 0), (WIN_W, 56), (22, 16, 14), (44, 32, 26))
        cv2.addWeighted(bar_layer, 0.9, canvas, 0.1, 0, dst=canvas)
        cv2.line(canvas, (0, 56), (WIN_W, 56), ACCENT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, "LIVE CAMERA - TOMATO CAPTURE", (18, 36), FONT, 0.65, TEXT_PRIMARY, 2, cv2.LINE_AA)

        if int(time.time() * 2) % 2 == 0:
            cv2.circle(canvas, (WIN_W - 34, 28), 7, DANGER, -1, cv2.LINE_AA)
            cv2.putText(canvas, "REC", (WIN_W - 78, 34), FONT, 0.5, DANGER, 1, cv2.LINE_AA)

        btn_h = 46
        self._set_buttons([
            ("CAPTURE PHOTO (SPACE)", "capture", 60, WIN_H - btn_h - 16, WIN_W // 2 - 10, WIN_H - 16),
            ("BACK (ESC)", "back_to_menu", WIN_W // 2 + 10, WIN_H - btn_h - 16, WIN_W - 60, WIN_H - 16),
        ])
        self._draw_buttons(canvas, primary_action="capture")

        self.canvas = canvas

    def _do_capture(self):
        if self.last_camera_frame is None:
            return
        frame = self.last_camera_frame.copy()
        self._release_camera()
        raw_path = os.path.join(tempfile.gettempdir(), "tomato_capture.jpg")
        cv2.imwrite(raw_path, frame)
        self._start_inference(frame)

    # -----------------------------------------------------------------
    #  WORKING state
    # -----------------------------------------------------------------

    def _render_working(self):
        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        _draw_vertical_gradient(canvas, (0, 0), (WIN_W, WIN_H), BG_TOP, BG_BOTTOM)

        cx, cy = WIN_W // 2, WIN_H // 2
        _draw_rounded_rect(canvas, (cx - 260, cy - 100), (cx + 260, cy + 100), PANEL_BG, radius=18)
        _draw_rounded_rect(canvas, (cx - 260, cy - 100), (cx + 260, cy + 100), ACCENT_COLOR, radius=18, thickness=2)

        angle = int((time.time() * 220) % 360)
        cv2.ellipse(canvas, (cx, cy - 25), (26, 26), angle, 0, 270, ACCENT_COLOR, 4, cv2.LINE_AA)

        _text_centered(canvas, self.working_message, cx, cy + 35, 0.65, TEXT_PRIMARY, 2)
        if self.working_sub:
            _text_centered(canvas, self.working_sub, cx, cy + 62, 0.48, TEXT_MUTED, 1)

        self.buttons = []
        self.canvas = canvas

    def _start_inference(self, image):
        self.state = STATE_WORKING
        self.working_message = "Running both models..."
        self.working_sub = f"{DETECTION_MODEL_ID}  +  {SEGMENTATION_MODEL_ID}"
        self.pending_job = lambda: self._do_inference(image)

    def _do_inference(self, image):
        try:
            result, elapsed = run_inference(image)
        except RuntimeError as e:
            self._enter_popup("Configuration error", str(e), DANGER, STATE_MENU)
            return
        except Exception as e:
            self._enter_popup("Inference error", str(e), DANGER, STATE_MENU)
            return

        result["manual_predictions"] = []

        try:
            canvas, note, fully_empty = render_dashboard(image, result, elapsed_seconds=elapsed)
        except Exception as e:
            self._enter_popup("Rendering error", str(e), DANGER, STATE_MENU)
            return

        self.current_image = image
        self.current_result = result
        self.current_elapsed = elapsed
        self.manual_predictions = []

        if fully_empty:
            self.result_canvas = canvas
            self._enter_popup("Nothing found", note, ACCENT_COLOR, STATE_MENU)
            return

        cv2.imwrite("tomato_result.png", canvas)
        self.result_canvas = canvas
        self._enter_result()

    # -----------------------------------------------------------------
    #  RESULT state
    # -----------------------------------------------------------------

    def _enter_result(self):
        self.state = STATE_RESULT
        self.hover_idx = None
        self._render_result()

    def _render_result(self):
        display, scale = _fit_into(self.result_canvas, WIN_W, WIN_H - 70)
        dh, dw = display.shape[:2]

        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        _draw_vertical_gradient(canvas, (0, 0), (WIN_W, WIN_H), BG_TOP, BG_BOTTOM)
        off_x, off_y = (WIN_W - dw) // 2, 10
        canvas[off_y:off_y + dh, off_x:off_x + dw] = display

        btn_h = 44
        self._set_buttons([
            ("MANUAL CHECK", "manual_select", WIN_W // 2 - 300, WIN_H - btn_h - 12, WIN_W // 2 - 10, WIN_H - 12),
            ("BACK TO MENU", "back_to_menu", WIN_W // 2 + 10, WIN_H - btn_h - 12, WIN_W // 2 + 300, WIN_H - 12),
        ])
        self._draw_buttons(canvas, primary_action="manual_select")

        self.canvas = canvas

    # -----------------------------------------------------------------
    #  MANUAL SELECT state (user draws a box for regions both models missed)
    # -----------------------------------------------------------------

    def _enter_manual_select(self):
        if self.current_image is None:
            return
        if isinstance(self.current_image, str):
            base = cv2.imread(self.current_image)
            if base is None:
                self._enter_popup("Error", "Could not reload the source image.", DANGER, STATE_RESULT)
                return
        else:
            base = self.current_image.copy()

        self.manual_base_img = base
        self.drag_start = None
        self.drag_current = None
        self.hover_idx = None
        self.state = STATE_MANUAL_SELECT
        self._render_manual_select()

    def _render_manual_select(self):
        base = self.manual_base_img
        display, scale = _fit_into(base, WIN_W, WIN_H - 110)
        dh, dw = display.shape[:2]

        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        _draw_vertical_gradient(canvas, (0, 0), (WIN_W, WIN_H), BG_TOP, BG_BOTTOM)

        off_x, off_y = (WIN_W - dw) // 2, 60
        canvas[off_y:off_y + dh, off_x:off_x + dw] = display
        self.manual_scale = scale
        self.manual_offset = (off_x, off_y)

        bar_layer = canvas.copy()
        _draw_vertical_gradient(bar_layer, (0, 0), (WIN_W, 50), (22, 16, 14), (44, 32, 26))
        cv2.addWeighted(bar_layer, 0.9, canvas, 0.1, 0, dst=canvas)
        cv2.line(canvas, (0, 50), (WIN_W, 50), MANUAL_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(canvas, "MANUAL CHECK - click and drag on the photo", (18, 33),
                    FONT, 0.6, TEXT_PRIMARY, 2, cv2.LINE_AA)

        cv2.rectangle(canvas, (off_x, off_y), (off_x + dw, off_y + dh), MANUAL_ACCENT, 1, cv2.LINE_AA)

        if self.drag_start and self.drag_current:
            p1, p2 = self.drag_start, self.drag_current
            overlay = canvas.copy()
            cv2.rectangle(overlay, p1, p2, MANUAL_ACCENT, -1)
            cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, dst=canvas)
            cv2.rectangle(canvas, p1, p2, MANUAL_ACCENT, 2, cv2.LINE_AA)

        btn_h = 40
        self._set_buttons([
            ("CANCEL (ESC)", "cancel_manual", WIN_W // 2 - 120, WIN_H - btn_h - 10, WIN_W // 2 + 120, WIN_H - 10),
        ])
        self._draw_buttons(canvas, primary_action=None)

        self.canvas = canvas

    def _on_mouse_manual(self, event, x, y, flags, param):
        off_x, off_y = self.manual_offset
        disp_h = int(self.manual_base_img.shape[0] * self.manual_scale)
        disp_w = int(self.manual_base_img.shape[1] * self.manual_scale)

        for i, btn in enumerate(self.buttons):
            bx1, by1, bx2, by2 = btn["rect"]
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                if event == cv2.EVENT_MOUSEMOVE:
                    if self.hover_idx != i:
                        self.hover_idx = i
                        self._render_manual_select()
                elif event == cv2.EVENT_LBUTTONDOWN:
                    self._handle_action(btn["action"])
                return

        if event == cv2.EVENT_MOUSEMOVE and self.hover_idx is not None and self.drag_start is None:
            self.hover_idx = None
            self._render_manual_select()

        cx = max(off_x, min(x, off_x + disp_w))
        cy = max(off_y, min(y, off_y + disp_h))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (cx, cy)
            self.drag_current = (cx, cy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (cx, cy)
            self._render_manual_select()
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self._finalize_manual_selection(self.drag_start, (cx, cy))
            self.drag_start = None
            self.drag_current = None

    def _finalize_manual_selection(self, p1, p2):
        off_x, off_y = self.manual_offset
        scale = self.manual_scale

        x1d, x2d = sorted([p1[0], p2[0]])
        y1d, y2d = sorted([p1[1], p2[1]])
        if (x2d - x1d) < 8 or (y2d - y1d) < 8:
            return

        ox1 = int((x1d - off_x) / scale)
        oy1 = int((y1d - off_y) / scale)
        ox2 = int((x2d - off_x) / scale)
        oy2 = int((y2d - off_y) / scale)

        base = self.manual_base_img
        crop = base[max(0, oy1):oy2, max(0, ox1):ox2]

        self.working_message = "Analyzing selection..."
        self.working_sub = "running both models on the marked region"
        self.state = STATE_WORKING
        self.pending_job = lambda: self._do_manual_classification(crop, ox1, oy1, ox2, oy2)

    def _do_manual_classification(self, crop, ox1, oy1, ox2, oy2):
        try:
            label, conf, reconciled, note = classify_crop_with_models(crop)
        except Exception as e:
            self._enter_popup("Manual check error", str(e), DANGER, STATE_RESULT)
            return

        self.manual_predictions.append({
            "class": label if label else "inconclusive",
            "confidence": conf,
            "x": (ox1 + ox2) / 2,
            "y": (oy1 + oy2) / 2,
            "width": ox2 - ox1,
            "height": oy2 - oy1,
            "_manual_check": True,
        })

        self._rebuild_result_canvas()
        self._enter_result()

    def _rebuild_result_canvas(self):
        base_result = self.current_result or {"det_predictions": [], "seg_predictions": []}
        combined = dict(base_result)
        combined["manual_predictions"] = self.manual_predictions
        try:
            canvas, _, _ = render_dashboard(self.current_image, combined, elapsed_seconds=self.current_elapsed)
        except Exception as e:
            self._enter_popup("Rendering error", str(e), DANGER, STATE_RESULT)
            return
        cv2.imwrite("tomato_result.png", canvas)
        self.result_canvas = canvas

    # -----------------------------------------------------------------
    #  POPUP state
    # -----------------------------------------------------------------

    def _enter_popup(self, title, message, color, return_state):
        self.state = STATE_POPUP
        self.popup_title = title
        self.popup_message = message
        self.popup_color = color
        self.popup_return_state = return_state
        self._render_popup()

    def _render_popup(self):
        canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        _draw_vertical_gradient(canvas, (0, 0), (WIN_W, WIN_H), BG_TOP, BG_BOTTOM)

        cx, cy = WIN_W // 2, WIN_H // 2
        _draw_rounded_rect(canvas, (cx - 300, cy - 110), (cx + 300, cy + 110), PANEL_BG, radius=18)
        _draw_rounded_rect(canvas, (cx - 300, cy - 110), (cx + 300, cy + 110), self.popup_color, radius=18, thickness=2)
        _text_centered(canvas, self.popup_title, cx, cy - 60, 0.75, self.popup_color, 2)

        words = self.popup_message.split()
        lines, current = [], ""
        for word in words:
            trial = (current + " " + word).strip()
            (tw, _), _ = cv2.getTextSize(trial, FONT, 0.5, 1)
            if tw > 540 and current:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        for i, line in enumerate(lines[:3]):
            _text_centered(canvas, line, cx, cy - 15 + i * 24, 0.5, TEXT_MUTED, 1)

        self._set_buttons([
            ("OK", "dismiss_popup", cx - 80, cy + 55, cx + 80, cy + 95),
        ])
        self._draw_buttons(canvas, primary_action="dismiss_popup")

        self.canvas = canvas

    # -----------------------------------------------------------------
    #  Mouse handling
    # -----------------------------------------------------------------

    def _on_mouse(self, event, x, y, flags, param):
        if self.state == STATE_MANUAL_SELECT:
            self._on_mouse_manual(event, x, y, flags, param)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            new_hover = None
            for i, btn in enumerate(self.buttons):
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    new_hover = i
                    break
            self.hover_idx = new_hover

        elif event == cv2.EVENT_LBUTTONDOWN:
            for btn in self.buttons:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._handle_action(btn["action"])
                    break

    def _handle_action(self, action):
        if action == "load":
            path = pick_image_file()
            if path:
                self._start_inference(path)
            else:
                self._render_menu()

        elif action == "camera":
            self._enter_camera()

        elif action == "capture":
            self._do_capture()

        elif action == "back_to_menu":
            self._enter_menu()

        elif action == "manual_select":
            self._enter_manual_select()

        elif action == "cancel_manual":
            self._enter_result()

        elif action == "dismiss_popup":
            if self.popup_return_state == STATE_MENU:
                self._enter_menu()
            else:
                self.state = self.popup_return_state

        elif action == "quit":
            self._quit()

    def _quit(self):
        self._release_camera()
        cv2.destroyAllWindows()
        sys.exit(0)

    # -----------------------------------------------------------------
    #  Main loop
    # -----------------------------------------------------------------

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        self._enter_menu()

        while True:
            if self.state == STATE_CAMERA and self.cap is not None:
                self._render_camera_frame()
            elif self.state == STATE_MENU:
                self._render_menu()
            elif self.state == STATE_WORKING:
                self._render_working()
            elif self.state == STATE_RESULT:
                self._render_result()
            elif self.state == STATE_MANUAL_SELECT:
                self._render_manual_select()
            elif self.state == STATE_POPUP:
                self._render_popup()

            cv2.imshow(WINDOW_NAME, self.canvas)

            if self.pending_job is not None:
                job = self.pending_job
                self.pending_job = None
                cv2.waitKey(1)
                job()

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                if self.state == STATE_CAMERA:
                    self._enter_menu()
                elif self.state == STATE_MENU:
                    self._quit()
                elif self.state == STATE_POPUP:
                    self._handle_action("dismiss_popup")
                elif self.state == STATE_RESULT:
                    self._enter_menu()
                elif self.state == STATE_MANUAL_SELECT:
                    self._enter_result()
            elif key == 32 and self.state == STATE_CAMERA:  # SPACE
                self._do_capture()

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break


if __name__ == "__main__":
    TomatoApp().run()