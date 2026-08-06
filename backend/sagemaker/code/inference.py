import base64
import os
import json
import cv2
import numpy as np
import onnxruntime as ort

# Both models run through onnxruntime only — no torch, no tensorflow — to
# avoid stacking two heavy framework runtimes on top of the container's own
# baseline. Both preprocessing paths were validated against the real labeled
# test images before deploying (100% on the classifier; YOLO confidence
# confirmed to match the original .pt model's behavior via ultralytics).

# --- Classifier (fish_classifier.onnx) ---
# Input: (batch, 224, 224, 3) NHWC, RAW 0-255 float32 — the model's own
# Rescaling layer (from the source .keras model) normalizes internally, so
# no manual normalization here. Output "dense_2": (batch, 2) softmax probs.
# Class index 0 = "Fresh", index 1 = "Not Fresh".
CLASSIFIER_INPUT_SIZE = 224
CLASS_NAMES = ["Fresh", "Not Fresh"]

# --- YOLO presence-gate (yolo_detection_model.onnx) ---
# Input: (1, 3, 640, 640) NCHW, 0-1 float32, letterboxed (aspect-ratio
# preserved, padded to square) — matches ultralytics' own preprocessing.
# Output "output0": (1, 5, 8400) — row 4 (5th row) is per-anchor confidence
# for this single-class ("fish") model; no NMS needed since we only care
# about the max confidence across all anchors, not individual boxes.
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.20
YOLO_PAD_COLOR = (114, 114, 114)


def model_fn(model_dir):
    """
    Loads both ONNX models into memory when the SageMaker container boots up.
    """
    print("⏳ Loading models for SageMaker endpoint...")

    yolo_path = os.path.join(model_dir, "models", "yolo_detection_model.onnx")
    classifier_path = os.path.join(model_dir, "models", "fish_classifier.onnx")

    yolo_session = ort.InferenceSession(yolo_path, providers=["CPUExecutionProvider"])
    classifier_session = ort.InferenceSession(classifier_path, providers=["CPUExecutionProvider"])

    return {
        "yolo": yolo_session,
        "classifier": classifier_session
    }


def input_fn(request_body, request_content_type):
    """
    Deserializes the incoming request (expects JSON with base64 image string).
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        b64_data = data.get("image_base64", "")
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_data)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes from base64 string.")
        return img

    raise ValueError(f"Unsupported content type: {request_content_type}")


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), YOLO_PAD_COLOR, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def predict_fn(image, models):
    """
    Executes the presence check (YOLO) and classification (both ONNX).
    """
    yolo_session = models["yolo"]
    classifier_session = models["classifier"]

    # 1. YOLO presence gate
    letterboxed = _letterbox(image, YOLO_INPUT_SIZE)
    yolo_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    yolo_chw = np.transpose(yolo_rgb, (2, 0, 1))
    yolo_batch = np.expand_dims(yolo_chw, axis=0).astype(np.float32)

    yolo_input_name = yolo_session.get_inputs()[0].name
    output0 = yolo_session.run(None, {yolo_input_name: yolo_batch})[0]  # [1, 5, 8400]
    max_conf = float(output0[0][4].max())

    if max_conf < YOLO_CONF_THRESHOLD:
        return {"status": "rejected", "detail": "No fish detected in image."}

    # 2. Preprocess for the classifier: 224x224 RGB, raw 0-255 float32
    resized = cv2.resize(image, (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    batch = np.expand_dims(rgb, axis=0)

    # 3. Run ONNX classifier inference (already softmax output)
    classifier_input_name = classifier_session.get_inputs()[0].name
    probs = classifier_session.run(None, {classifier_input_name: batch})[0][0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    conf = float(probs[idx])

    return {
        "status": "success",
        "prediction": {
            "label": label,
            "confidence": round(conf, 3)
        }
    }


def output_fn(prediction_output, accept_content_type):
    """
    Serializes the dictionary output back into a JSON HTTP response.
    """
    if accept_content_type == "application/json":
        return json.dumps(prediction_output), "application/json"
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
