import base64
import json

import cv2
import numpy as np

try:
    from rembg import remove as _rembg_remove
    from PIL import Image as _PILImage
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def _grabcut_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        return np.full((h, w), 255, dtype=np.uint8)

    mask      = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mx, my    = max(4, w // 10), max(4, h // 10)
    rect      = (mx, my, w - 2 * mx, h - 2 * my)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)
        if fg.sum() < h * w * 0.05 * 255:
            raise ValueError("near-empty mask")
        return fg
    except Exception:
        return np.full((h, w), 255, dtype=np.uint8)


def detect_fish(image: np.ndarray):
    H, W = image.shape[:2]

    if REMBG_AVAILABLE:
        pil_img  = _PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        rgba     = np.array(_rembg_remove(pil_img))
        alpha    = rgba[:, :, 3]
        raw_mask = (alpha > 128).astype(np.uint8) * 255
    else:
        raw_mask = _grabcut_mask(image)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    min_area = H * W * 0.01
    valid    = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        valid = contours

    largest    = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    px = int(w * 0.05); py = int(h * 0.05)
    x1 = max(0, x - px); y1 = max(0, y - py)
    x2 = min(W, x + w + px); y2 = min(H, y + h + py)

    return (x1, y1, x2, y2), mask


def handler(event, context):
    """
    AWS Lambda entry point. Expects {"image_base64": "..."} (invoked directly
    via boto3 invoke_endpoint, not behind API Gateway) and returns
    {"status": "success", "bbox": [...], "mask_base64": "<PNG>"} or
    {"status": "rejected", "detail": "..."}.
    """
    try:
        b64 = event["image_base64"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data  = base64.b64decode(b64)
        arr   = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "rejected", "detail": "Could not decode image bytes."}

        bbox, mask = detect_fish(image)
        if bbox is None:
            return {"status": "rejected", "detail": "No fish foreground could be segmented."}

        _, mask_png = cv2.imencode(".png", mask)
        mask_b64    = base64.b64encode(mask_png).decode()

        return {
            "status": "success",
            "bbox": list(bbox),
            "mask_base64": mask_b64,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
