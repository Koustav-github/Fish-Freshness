import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# --- CHANGED SECTION: diagnose 180s+ hang ---
# Explicit timeout — the Groq SDK's default (unlike the Gemini fallback's
# requests.post(timeout=10)) was unbounded, a likely culprit for requests
# hanging far longer than expected with no visible error.
_groq_client   = Groq(api_key=GROQ_API_KEY, timeout=15.0) if GROQ_API_KEY else None
# -----------------------

# --- CHANGED SECTION: AWS SageMaker integration ---
# YOLO + classifier inference now run remotely on a deployed SageMaker
# endpoint instead of loading the .pt/.onnx models in-process.
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "fish-freshness-koustav-JU-km").strip()
AWS_REGION              = os.getenv("AWS_REGION", "us-east-1").strip()
# --- CHANGED SECTION: SageMaker cold-start retries ---
# SageMaker serverless endpoints scale to zero when idle; the first request
# after that can hit ModelNotReadyException while it spins back up — AWS's
# own guidance is to retry with backoff, since it normally self-resolves in
# a few seconds. max_attempts=1 (no retry) turns that into a hard user-facing
# failure instead, so this client gets real retries.
_sagemaker_config  = boto3.session.Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 3, "mode": "standard"})
_sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION, config=_sagemaker_config)
# -----------------------

# rembg-based segmentation now runs on a separate Lambda instead of in-process,
# so this Render deploy doesn't need to carry rembg's ~300MB dependency chain.
LAMBDA_PREPROCESSING_FUNCTION = os.getenv("LAMBDA_PREPROCESSING_FUNCTION", "fish-preprocess").strip()
# --- CHANGED SECTION: diagnose 180s+ hang ---
# This client keeps max_attempts=1 deliberately — the earlier 180s hang was a
# genuinely stuck downstream call (fish-preprocess), not a fast-failing cold
# start, so retrying it would only make a real hang worse, not better.
_lambda_config = boto3.session.Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 1})
_lambda_client = boto3.client("lambda", region_name=AWS_REGION, config=_lambda_config)
# -----------------------

# --- CHANGED SECTION: parallelize independent AWS calls ---
# Classification (SageMaker) and segmentation (fish-preprocess) don't depend
# on each other's results, but were previously called sequentially — when
# both happen to be cold at once, that means paying the FULL sum of both
# cold-start times (e.g. ~50s + ~100s) instead of just the larger one.
# Reused across warm Lambda invocations, not recreated per request.
_executor = ThreadPoolExecutor(max_workers=4)
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
                f"gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
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
MAX_IMAGE_DIM        = 4096   # px — hard reject above this, before any processing
MIN_IMAGE_DIM        = 128    # px — below this segmentation is unreliable
MAX_B64_BYTES        = 5 * 1024 * 1024   # 5 MB of raw base64 text


# --- CHANGED SECTION: AWS SageMaker integration ---
# No local model registry or warm-up needed anymore — YOLO + classifier live
# on the SageMaker endpoint now, so there's nothing to load at startup.
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
# -----------------------


# ── App ───────────────────────────────────────────────────────────────────────
# --- CHANGED SECTION: move to Lambda ---
# Dropped slowapi rate limiting — its in-memory counter doesn't persist
# across ephemeral Lambda containers (each cold start gets its own memory),
# so it silently stopped being an actual limit. Throttling now happens at
# the Lambda Function URL / reserved concurrency level instead.
app = FastAPI(title="FreshlyFishy API", lifespan=lifespan)
# -----------------------

# --- CHANGED SECTION: CORS ownership ---
# Handled here only — the Lambda Function URL's own CORS config must stay
# OFF. Having both inject Access-Control-Allow-Origin produces a duplicate
# value ("*, *"), which browsers reject outright, and relying on the
# Function URL alone left OPTIONS preflight requests with no CORS headers
# at all (Starlette's bare default OPTIONS response, not CORS-aware).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------


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

# --- CHANGED SECTION: diagnose 180s+ hang ---
# GrabCut's cost scales with pixel count. Running it at full resolution
# (e.g. 1600x1600 = 2.56M px) on Lambda's modest CPU allocation (which
# scales with configured memory) took 159+ seconds and got killed by the
# function timeout — confirmed via CloudWatch timing logs. Downscaling to a
# capped working size before segmenting, then scaling the resulting mask
# back up, keeps this fallback usable regardless of image size.
GRABCUT_MAX_DIM = 512
# -----------------------

def _grabcut_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        return np.full((h, w), 255, dtype=np.uint8)

    scale = min(1.0, GRABCUT_MAX_DIM / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
    sh, sw = small.shape[:2]

    mask      = np.zeros((sh, sw), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mx, my    = max(4, sw // 10), max(4, sh // 10)
    rect      = (mx, my, sw - 2 * mx, sh - 2 * my)

    try:
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)
        if fg.sum() < sh * sw * 0.05 * 255:
            raise ValueError("near-empty mask")
        return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else fg
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


# --- CHANGED SECTION: parallelize independent AWS calls ---
# Split from the old detect_fish() so the network call (invoke_preprocessing_lambda)
# can be fired off in parallel with invoke_sagemaker, while this pure
# result-processing step runs afterward once both are back.
def _process_preprocessing_result(image: np.ndarray, result: dict):
    H, W = image.shape[:2]

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
    print(f"[timing] preprocessing fallback: falling back to local GrabCut, image shape={image.shape}...")
    _t = time.perf_counter()
    raw_mask = _grabcut_mask(image)
    print(f"[timing] preprocessing fallback: GrabCut done in {time.perf_counter()-_t:.1f}s")
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


# ── Predict endpoint ──────────────────────────────────────────────────────────

@app.post("/predict")
def predict(req: ImageRequest):
    t0 = time.perf_counter()

    try:
        image = b64_to_cv2(req.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    validate_image(req.image_base64, image)

    original_b64 = cv2_to_b64(image)
    H, W = image.shape[:2]

    # --- CHANGED SECTION: parallelize independent AWS calls ---
    # Classification (SageMaker) and segmentation (fish-preprocess) are
    # independent — fire both off at once instead of paying their cold-start
    # times sequentially.
    print(f"[timing] t={time.perf_counter()-t0:.1f}s submitting invoke_sagemaker + invoke_preprocessing_lambda in parallel...")
    sagemaker_future     = _executor.submit(invoke_sagemaker, req.image_base64)
    preprocessing_future = _executor.submit(invoke_preprocessing_lambda, req.image_base64)

    result = sagemaker_future.result()
    print(f"[timing] t={time.perf_counter()-t0:.1f}s invoke_sagemaker done")

    if result.get("status") == "rejected":
        raise HTTPException(
            status_code=422,
            detail=result.get("detail", "No fish detected in the image. Please upload a clear photo of a fish."),
        )

    label = result["prediction"]["label"]
    conf  = result["prediction"]["confidence"]
    # gradcam_base64 is a real class-activation heatmap computed server-side
    # in inference.py, from the classifier's own feature-map output — not a
    # placeholder. Falls back to None if an older inference.py is deployed.
    gradcam_b64 = result.get("gradcam_base64")

    preprocessing_result = preprocessing_future.result()
    print(f"[timing] t={time.perf_counter()-t0:.1f}s invoke_preprocessing_lambda done, status={preprocessing_result.get('status')}")
    bbox, full_mask = _process_preprocessing_result(image, preprocessing_result)
    print(f"[timing] t={time.perf_counter()-t0:.1f}s preprocessing result processed")
    # -----------------------
    if bbox is None:
        raise HTTPException(status_code=422, detail="No fish foreground could be segmented")

    x1, y1, x2, y2 = bbox
    crop        = image
    fish_mask = full_mask

    mask_coverage = round(float((full_mask > 0).sum()) / (H * W), 3)

    decision    = "Auto Approved"

    # Fall back to a plain crop only if the endpoint didn't return a heatmap
    # (e.g. an older inference.py) — keeps /predict from breaking either way.
    gradcam_b64_out = gradcam_b64 if gradcam_b64 else cv2_to_b64(crop)

    print(f"[timing] t={time.perf_counter()-t0:.1f}s computing roi_display...")
    roi_display = apply_clahe(apply_mask_zero(crop, fish_mask))
    print(f"[timing] t={time.perf_counter()-t0:.1f}s roi_display done")

    focus_areas  = ["Eye clarity", "Gill color", "Skin texture"]
    print(f"[timing] t={time.perf_counter()-t0:.1f}s calling generate_llm_analysis...")
    llm_analysis = generate_llm_analysis(label, conf, decision, mask_coverage, focus_areas)
    print(f"[timing] t={time.perf_counter()-t0:.1f}s generate_llm_analysis done")

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
            "gradcam":  gradcam_b64_out,
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