"""
=====================================================================
  MODEL COMPARISON: detection ("tomato-splxa/1") vs.
                     segmentation ("unamba_tesis/6")
=====================================================================

Runs both Roboflow models on the same photo and shows them side by
side so you can judge which one is actually better for your use case:
  - LEFT:  detection model — rectangular boxes
  - RIGHT: segmentation model — real pixel masks (if the API returns
           polygon points; falls back to boxes if it doesn't)

Also prints basic comparison stats to the console: object count,
average confidence, and inference latency for each model.

Requirements:
    pip install inference-sdk python-dotenv opencv-python numpy

Usage:
    python compare_models.py <image_path>
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

load_dotenv()

API_URL = "https://serverless.roboflow.com"
DETECTION_MODEL_ID = "tomato-splxa/1"
SEGMENTATION_MODEL_ID = "unamba_tesis/6"
CONFIDENCE_THRESHOLD = 0.5

DET_COLOR = (32, 176, 255)     # amber (BGR)
SEG_COLOR = (160, 229, 0)      # mint green (BGR)
TEXT_PRIMARY = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_DUPLEX


def get_client() -> InferenceHTTPClient:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not found in environment / .env file.")
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def run_model(image_path: str, model_id: str) -> tuple:
    client = get_client()
    config = InferenceConfiguration(confidence_threshold=CONFIDENCE_THRESHOLD, iou_threshold=0.5)
    client.configure(config)
    start = time.time()
    result = client.infer(image_path, model_id=model_id)
    elapsed = time.time() - start
    return result, elapsed


def draw_boxes(img: np.ndarray, predictions: list, color: tuple) -> np.ndarray:
    out = img.copy()
    for pred in predictions:
        label = pred.get("class") or pred.get("class_name", "object")
        conf = pred.get("confidence", 0)
        cx, cy = pred.get("x", 0), pred.get("y", 0)
        pw, ph = pred.get("width", 0), pred.get("height", 0)
        x1, y1 = int(cx - pw / 2), int(cy - ph / 2)
        x2, y2 = int(cx + pw / 2), int(cy + ph / 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        text = f"{label} {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, text, (x1 + 4, y1 - 4), FONT, 0.55, (15, 15, 20), 1, cv2.LINE_AA)
    return out


def draw_masks(img: np.ndarray, predictions: list, color: tuple) -> np.ndarray:
    """
    Draws real segmentation polygons when the API returns "points"
    (a list of {x, y} vertices per prediction). Falls back to a plain
    bounding box for any prediction that has no polygon data, so this
    still works even if the model turns out to be detection-only.
    """
    out = img.copy()
    overlay = img.copy()
    has_any_polygon = False

    for pred in predictions:
        label = pred.get("class") or pred.get("class_name", "object")
        conf = pred.get("confidence", 0)
        points = pred.get("points")

        if points:
            has_any_polygon = True
            pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            label_pos = tuple(pts[0])
        else:
            cx, cy = pred.get("x", 0), pred.get("y", 0)
            pw, ph = pred.get("width", 0), pred.get("height", 0)
            x1, y1 = int(cx - pw / 2), int(cy - ph / 2)
            x2, y2 = int(cx + pw / 2), int(cy + ph / 2)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label_pos = (x1, y1)

        text = f"{label} {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        lx, ly = label_pos
        cv2.rectangle(out, (lx, max(0, ly - th - 8)), (lx + tw + 8, ly), color, -1)
        cv2.putText(out, text, (lx + 4, ly - 4), FONT, 0.55, (15, 15, 20), 1, cv2.LINE_AA)

    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    return out, has_any_polygon


def label_banner(img: np.ndarray, text: str, color: tuple) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    cv2.rectangle(out, (0, 0), (w, 44), (20, 16, 14), -1)
    cv2.rectangle(out, (0, 40), (w, 44), color, -1)
    cv2.putText(out, text, (14, 30), FONT, 0.75, TEXT_PRIMARY, 2, cv2.LINE_AA)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not open image: {image_path}")
        sys.exit(1)

    print(f"Running detection model ({DETECTION_MODEL_ID})...")
    det_result, det_elapsed = run_model(image_path, DETECTION_MODEL_ID)
    det_preds = det_result.get("predictions", [])

    print(f"Running segmentation model ({SEGMENTATION_MODEL_ID})...")
    seg_result, seg_elapsed = run_model(image_path, SEGMENTATION_MODEL_ID)
    seg_preds = seg_result.get("predictions", [])

    # --- Console comparison stats ---
    def stats(preds):
        if not preds:
            return 0, 0.0
        confs = [p.get("confidence", 0) for p in preds]
        return len(preds), sum(confs) / len(confs)

    det_count, det_avg_conf = stats(det_preds)
    seg_count, seg_avg_conf = stats(seg_preds)

    print("\n" + "=" * 60)
    print(f"{'':20}{'DETECTION':>18}{'SEGMENTATION':>20}")
    print(f"{'model':20}{DETECTION_MODEL_ID:>18}{SEGMENTATION_MODEL_ID:>20}")
    print(f"{'objects found':20}{det_count:>18}{seg_count:>20}")
    print(f"{'avg confidence':20}{det_avg_conf * 100:>17.1f}%{seg_avg_conf * 100:>19.1f}%")
    print(f"{'inference time':20}{det_elapsed * 1000:>15.0f} ms{seg_elapsed * 1000:>17.0f} ms")
    print("=" * 60 + "\n")

    # --- Side-by-side visual comparison ---
    det_vis = draw_boxes(img, det_preds, DET_COLOR)
    det_vis = label_banner(det_vis, f"DETECTION: {det_count} found, {det_elapsed*1000:.0f} ms", DET_COLOR)

    seg_vis, has_polygons = draw_masks(img, seg_preds, SEG_COLOR)
    seg_vis = label_banner(seg_vis, f"SEGMENTATION: {seg_count} found, {seg_elapsed*1000:.0f} ms", SEG_COLOR)
    if not has_polygons and seg_preds:
        print("Note: the segmentation model returned no 'points' polygon data — "
              "it may be detection-only, or the response format differs. "
              "Falling back to boxes for the right-hand view.\n")

    h, w = img.shape[:2]
    gap = 6
    combined = np.full((h, w * 2 + gap, 3), 20, dtype=np.uint8)
    combined[0:h, 0:w] = det_vis
    combined[0:h, w + gap:w * 2 + gap] = seg_vis

    max_w, max_h = 1800, 950
    scale = min(max_w / combined.shape[1], max_h / combined.shape[0], 1.0)
    if scale < 1.0:
        combined = cv2.resize(combined, (int(combined.shape[1] * scale), int(combined.shape[0] * scale)))

    cv2.imwrite("model_comparison.png", combined)
    print("Saved side-by-side comparison to: model_comparison.png")

    cv2.namedWindow("Detection vs Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection vs Segmentation", combined.shape[1], combined.shape[0])
    cv2.imshow("Detection vs Segmentation", combined)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()