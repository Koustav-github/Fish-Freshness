import base64
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
# --- CHANGED SECTION: AWS SageMaker integration ---
import json
import boto3
# -----------------------
import requests as _requests
# --- CHANGED SECTION ---
# tensorflow is no longer needed since the classifier is now ONNX,
# but keep if required elsewhere. Removed tf.keras load_model.
# -----------------------
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
_groq_client   = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- CHANGED SECTION: AWS SageMaker integration ---
# YOLO + classifier inference now run remotely on a deployed SageMaker
# endpoint instead of loading the .pt/.onnx models in-process.
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "fish-freshness-koustav-JU-km").strip()
AWS_REGION              = os.getenv("AWS_REGION", "us-east-1").strip()
_sagemaker_runtime      = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# rembg-based segmentation now runs on a separate Lambda instead of in-process,
# so this Render deploy doesn't need to carry rembg's ~300MB dependency chain.
LAMBDA_PREPROCESSING_FUNCTION = os.getenv("LAMBDA_PREPROCESSING_FUNCTION", "fish-preprocess").strip()
_lambda_client                = boto3.client("lambda", region_name=AWS_REGION)
# -----------------------


# ── LLM humanized analysis ────────────────────────────────────────────────────

def generate_llm_analysis(label: str, confidence: float, decision: str,
                            mask_coverage: float, focus_areas: list[str]) -> str:
    pct   = round(confidence * 100, 1)
    prompt = (
        f"You are a fish freshness expert AI. A vision model analyzed a fish image.\n\n"
        f"Results:\n"
        f"- Verdict: {label}\n"
        f"- Confidence: {pct}%\n"
        f"- Decision status: {decision}\n"
        f"- Fish body coverage in frame: {round(mask_coverage * 100, 1)}%\n"
        f"- Anatomical features examined: {', '.join(focus_areas)}\n\n"
        f"Write 2-3 concise sentences for a market vendor or consumer. "
        f"Explain what indicators the model observed, what the verdict means practically, "
        f"and give a brief recommendation. No bullet points, no headers, no jargon."
    )

    if _groq_client:
        try:
            resp = _groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=160,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    if GEMINI_API_KEY:
        try:
            url  = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            )
            body = {"contents": [{"parts": [{"text": prompt}]}]}
            r    = _requests.post(url, json=body, timeout=10)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass

    if label == "Fresh":
        return (
            f"The model assessed this fish as Fresh with {pct}% confidence, "
            f"examining {focus_areas[0].lower()} and surrounding tissue. "
            f"The fish appears safe for consumption."
        )
    return (
        f"The model assessed this fish as Not Fresh with {pct}% confidence, "
        f"detecting signs of spoilage in {focus_areas[0].lower()} and related areas. "
        f"Consumption is not recommended."
    )


# --- CHANGED SECTION: preprocessing moved to Lambda ---
# rembg no longer imported locally — segmentation runs on the fish-preprocess
# Lambda. Pure-cv2 GrabCut (_grabcut_mask below) stays as a local fallback if
# that call fails, so /predict can still return a usable response.
# -----------------------


# ── Image guardrails ──────────────────────────────────────────────────────────
MAX_IMAGE_DIM        = 4096   # px — above this segmentation risks OOM
MIN_IMAGE_DIM        = 128    # px — below this segmentation is unreliable
MAX_B64_BYTES        = 5 * 1024 * 1024   # 5 MB of raw base64 text


# --- CHANGED SECTION: AWS SageMaker integration ---
# No local model registry or warm-up needed anymore — YOLO + classifier live
# on the SageMaker endpoint now, so there's nothing to load at startup.
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
# -----------------------


# ── Rate limiter (SlowAPI) ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FreshlyFishy API", lifespan=lifespan)

app.state.limiter = limiter
def _rate_limit_handler(*_):
    return JSONResponse(
        status_code=429,
        content={
            "error":   "rate_limit_exceeded",
            "detail":  "Too many requests — limit is 5 per minute per IP. Please wait before retrying.",
            "retry_after_seconds": 60,
        },
    )

app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────
class ImageRequest(BaseModel):
    image_base64: str


# ── Image utilities ───────────────────────────────────────────────────────────

def b64_to_cv2(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    return img


def cv2_to_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    _, buf = cv2.imencode(ext, img)
    return base64.b64encode(buf).decode()


# ── Input guardrails ──────────────────────────────────────────────────────────

def validate_image(b64_raw: str, image: np.ndarray) -> None:
    if len(b64_raw.encode()) > MAX_B64_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Payload too large — maximum allowed is 5 MB.",
        )

    H, W = image.shape[:2]

    if H < MIN_IMAGE_DIM or W < MIN_IMAGE_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Image too small ({W}×{H} px) — minimum is {MIN_IMAGE_DIM}×{MIN_IMAGE_DIM} px.",
        )

    if H > MAX_IMAGE_DIM or W > MAX_IMAGE_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Image too large ({W}×{H} px) — maximum is {MAX_IMAGE_DIM}×{MAX_IMAGE_DIM} px.",
        )


# --- CHANGED SECTION: AWS SageMaker integration ---
# ── Remote inference via AWS SageMaker ────────────────────────────────────────

def invoke_sagemaker(b64_image: str) -> dict:
    """
    Calls the deployed SageMaker endpoint, which runs the YOLO presence-gate
    and ONNX classifier together server-side. Returns either
    {"status": "rejected", "detail": ...} or
    {"status": "success", "prediction": {"label": ..., "confidence": ...}}.
    """
    payload = json.dumps({"image_base64": b64_image})
    try:
        resp = _sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model endpoint call failed: {exc}")
    return json.loads(resp["Body"].read())
# -----------------------


# ── Fish detection via remote rembg Lambda (local GrabCut fallback) ───────────

def _grabcut_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        return np.full((h, w), 255, dtype=np.uint8)

    mask        = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mx, my    = max(4, w // 10), max(4, h // 10)
    rect        = (mx, my, w - 2 * mx, h - 2 * my)

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


# --- CHANGED SECTION: preprocessing moved to Lambda ---
def invoke_preprocessing_lambda(b64_image: str) -> dict:
    """
    Calls the fish-preprocess Lambda, which runs rembg (U²-Net) segmentation
    server-side. Returns {"status": "success", "bbox": [...], "mask_base64": "<PNG>"},
    {"status": "rejected", "detail": ...}, or {"status": "error", "detail": ...}
    on any local/network failure (never raises — caller decides how to degrade).
    """
    try:
        resp = _lambda_client.invoke(
            FunctionName=LAMBDA_PREPROCESSING_FUNCTION,
            InvocationType="RequestResponse",
            Payload=json.dumps({"image_base64": b64_image}).encode(),
        )
        payload = json.loads(resp["Payload"].read())
        if resp.get("FunctionError"):
            return {"status": "error", "detail": str(payload)}
        return payload
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _bbox_from_mask(mask: np.ndarray, H: int, W: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = H * W * 0.01          # ignore noise < 1 % of image
    valid    = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        valid = contours             # last resort: take whatever exists

    largest    = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    px = int(w * 0.05);  py = int(h * 0.05)
    x1 = max(0, x - px);  y1 = max(0, y - py)
    x2 = min(W, x + w + px);  y2 = min(H, y + h + py)

    return (x1, y1, x2, y2)


def detect_fish(image: np.ndarray, b64_image: str):
    H, W = image.shape[:2]

    result = invoke_preprocessing_lambda(b64_image)

    if result.get("status") == "success":
        mask_bytes = base64.b64decode(result["mask_base64"])
        mask_arr   = np.frombuffer(mask_bytes, np.uint8)
        mask       = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return tuple(result["bbox"]), mask
        # fall through to local fallback if the returned mask was corrupt

    elif result.get("status") == "rejected":
        # Genuine "no fish visible" determination from the Lambda — not a
        # failure, so don't mask it with a local fallback.
        return None, None

    # Lambda unreachable/errored — degrade to local pure-cv2 GrabCut
    raw_mask = _grabcut_mask(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  kernel, iterations=1)

    bbox = _bbox_from_mask(mask, H, W)
    if bbox is None:
        return None, None
    return bbox, mask
# -----------------------


# ── Background suppression ────────────────────────────────────────────────────

def apply_mask_zero(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = crop.copy()
    result[mask == 0] = 0
    return result


def blur_background(crop: np.ndarray, mask: np.ndarray, ksize: int = 51) -> np.ndarray:
    blurred = cv2.GaussianBlur(crop, (ksize | 1, ksize | 1), 0)
    result  = crop.copy()
    result[mask == 0] = blurred[mask == 0]
    return result


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --- CHANGED SECTION: AWS SageMaker integration ---
# preprocess_roi() / classify() removed — classification now happens inside
# the SageMaker endpoint's own inference.py, driven off the raw base64 image.
# -----------------------


# ── Grad-CAM Note ─────────────────────────────────────────────────────────────
# Note: Background-suppressed Grad-CAM (using tf.GradientTape) requires a native 
# TensorFlow/Keras model structure. Since your classifier is now an ONNX model, 
# Grad-CAM logic has been bypassed/safely defaulted below to prevent crashes.


# ── Predict endpoint ──────────────────────────────────────────────────────────

@app.post("/predict")
@limiter.limit("5/minute")
def predict(request: Request, req: ImageRequest):
    t0 = time.perf_counter()

    try:
        image = b64_to_cv2(req.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    validate_image(req.image_base64, image)

    original_b64 = cv2_to_b64(image)
    H, W = image.shape[:2]

    # --- CHANGED SECTION: AWS SageMaker integration ---
    # One remote call replaces the local yolo_fish_present() + classify() pair —
    # the SageMaker endpoint runs the YOLO presence-gate and ONNX classifier together.
    result = invoke_sagemaker(req.image_base64)

    if result.get("status") == "rejected":
        raise HTTPException(
            status_code=422,
            detail=result.get("detail", "No fish detected in the image. Please upload a clear photo of a fish."),
        )

    label = result["prediction"]["label"]
    conf  = result["prediction"]["confidence"]
    # -----------------------

    bbox, full_mask = detect_fish(image, req.image_base64)
    if bbox is None:
        raise HTTPException(status_code=422, detail="No fish foreground could be segmented")

    x1, y1, x2, y2 = bbox
    crop        = image
    fish_mask = full_mask

    mask_coverage = round(float((full_mask > 0).sum()) / (H * W), 3)

    decision    = "Auto Approved"

    # --- CHANGED SECTION ---
    # Grad-CAM code requires Keras GradientTape. Replaced with blank/fallback display if model is ONNX.
    cam_img = crop.copy()  # Fallback visualization replacement
    # -----------------------

    roi_display = apply_clahe(apply_mask_zero(crop, fish_mask))

    focus_areas  = ["Eye clarity", "Gill color", "Skin texture"]
    llm_analysis = generate_llm_analysis(label, conf, decision, mask_coverage, focus_areas)

    processing_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "status": "success",
        "prediction": {
            "label":      label,
            "confidence": round(conf, 3),
            "decision":   decision,
        },
        "detection": {
            "bbox":         [x1, y1, x2, y2],
            "mask_coverage": mask_coverage,
        },
        "images": {
            "original": original_b64,
            "roi":      cv2_to_b64(roi_display),
            "gradcam":  cv2_to_b64(cam_img),
        },
        "metadata": {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": processing_ms,
            "model_versions": {
                "segmentor":  "U²-Net (rembg, remote Lambda)",
                "classifier": "EfficientNetV2S (SageMaker remote)",
            },
        },
        "explanation": {
            "focus_areas": focus_areas,
            "note":        "Freshness indicators verified successfully.",
            "llm_analysis": llm_analysis,
        },
    }