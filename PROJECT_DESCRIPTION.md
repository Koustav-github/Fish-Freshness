# FreshlyFishy — Project Description

AI-powered fish freshness detection: a two-stage computer vision pipeline (YOLOv8 detection → U²-Net segmentation → EfficientNetV2S classification → Grad-CAM explainability → LLM humanisation), fully deployed on a serverless AWS backend with a Next.js frontend on Vercel.

**Team:** Koustav Manna · Satyam Singh

---

## 1. Model Training (offline, prior to deployment)

Two models were trained independently on Kaggle (T4 GPU) before any deployment work began.

### 1.1 Fish Detector — YOLOv8

| | |
|---|---|
| Base model | `yolov8l.pt` (COCO-pretrained, 25M params) |
| Auto-labeling | `yolov8l-worldv2.pt` (open-vocabulary YOLO-World, prompt: `"fish"`) auto-annotated all 22,000 classification images with bounding boxes — no manual annotation needed |
| Label filters | bbox area ≥ 3% of image · aspect ratio ≥ 1.2 · max 6 fish/image |
| Training | 1280px image size, 100 epochs with early stopping (patience 20), mosaic/copy-paste/flip/rotation/HSV augmentation |
| Loss weights | box=7.5 · cls=0.5 · dfl=1.5 |

**Results** (641 test images / 866 instances): Precision **0.956**, Recall **0.875**, mAP@50 **0.938**, mAP@50-95 **0.757**, inference **7.4 ms/image**.

### 1.2 Freshness Classifier — EfficientNetV2S

| | |
|---|---|
| Dataset | 22,000 images total (9,000 fresh · 13,000 not fresh), Kaggle: `satyamsingh0912/fish-classifier-dataset` |
| Backbone | EfficientNetV2S, ImageNet-pretrained, `include_preprocessing=True` |
| Custom head | GlobalAveragePooling → BatchNorm → Dense(512, GELU) → Dense(256, GELU) → Dense(2, softmax) |
| Class weights | corrects for the 13k/9k imbalance |
| Augmentation | RandomFlip/Rotation(±10°)/Zoom/Translation/Brightness(±25%)/Contrast |
| Phase 1 | 25 epochs, frozen backbone, AdamW lr=3e-4, label smoothing 0.05 |
| Phase 2 | 25 epochs, top-60 layers unfrozen, cosine decay 5e-5 → 0 |
| Monitored metric | `val_AUC` (robust to class imbalance) |

**Results:** Test accuracy **94.378%**, ROC-AUC **0.982**, Weighted F1 **0.914**.

Both notebooks (`fish-classifier-final.ipynb`, `build_yolo_notebook.py`) and the trained artifacts (`fish_classifier.keras`, `yolo_detection_model.pt`) live in `backend/`.

---

## 2. From Local Models to Cloud-Native Deployment

The original design ran both models **in-process** inside a single FastAPI server (`main.py`), loaded via `torch`/`ultralytics` (YOLO) and `tensorflow`/`keras` (classifier), plus `rembg` for U²-Net segmentation — all on one machine.

This project's deployment work rebuilt that into a **fully serverless AWS architecture**, split into three independently-deployed pieces:

```
┌──────────────┐     POST /predict      ┌─────────────────────┐
│   Frontend   │ ─────────────────────▶ │  fish-api (Lambda)  │
│  (Vercel)    │ ◀───────────────────── │  FastAPI + Mangum   │
└──────────────┘      JSON response      └──────────┬───────────┘
                                                     │
                                    ┌────────────────┼────────────────┐
                                    ▼                                 ▼
                         ┌─────────────────────┐         ┌──────────────────────┐
                         │  SageMaker Endpoint  │         │ fish-preprocess       │
                         │  (serverless)        │         │ (Lambda)              │
                         │  YOLO + Classifier    │         │ rembg segmentation    │
                         │  ONNX, dual-output    │         │                       │
                         │  (probs + Grad-CAM    │         │  bbox + mask →        │
                         │   feature map)        │         │  main.py composites   │
                         └─────────────────────┘         └──────────────────────┘
```

### 2.1 Why this split

- **`fish-api`** (AWS Lambda, container image, fronted by a **Lambda Function URL**) hosts the FastAPI app that used to run on Render. It's the orchestration layer: decodes the request, calls the two services below, composites the response, calls the LLM, returns JSON. It carries no ML framework at all — just `boto3`, `opencv-python-headless`, `numpy`, `fastapi`.
- **SageMaker serverless endpoint** runs YOLO presence-detection + EfficientNetV2S classification, both converted to **ONNX** and executed through a single `onnxruntime` session — no `torch`, no `tensorflow` in the serving container.
- **`fish-preprocess`** (a second Lambda) runs `rembg`/U²-Net segmentation independently, since its dependency chain (`pymatting`, `scipy`, `numba`, `scikit-image`) is ~300MB and unrelated to classification.

This mirrors the original architecture's separation of concerns (detection → segmentation → classification → explainability) but maps each stage onto the AWS service best suited to it, instead of one process holding every framework in memory at once.

### 2.2 Model conversion — PyTorch/TensorFlow → ONNX

Both trained models were exported to ONNX so the whole serving stack could run on a single lightweight `onnxruntime`, rather than paying the import/memory cost of `torch` **and** `tensorflow` in the same container:

- **YOLOv8 → ONNX** via Ultralytics' built-in exporter. Output `(1, 5, 8400)` is decoded manually (max confidence across anchors — no NMS needed since only presence, not localisation, matters downstream).
- **EfficientNetV2S (Keras) → ONNX** via `tf2onnx`. Two non-trivial fixes were required:
  - The model's exact GELU activation decomposes into an `Erfc` op that `tf2onnx` has no conversion rule for. Fixed by swapping to the tanh-approximation GELU variant before export (no weights touched — GELU has no trainable parameters, and the swap is numerically negligible, verified at 100% agreement across all validation images).
  - The model was re-exported with a **second output** exposing `top_activation` (the last conv feature map, `(7,7,1280)`, before global-average-pooling) — this had no output before. That feature map is what powers the real Grad-CAM heatmap computed server-side in `inference.py` (channel-wise mean → ReLU → normalise → resize → JET colormap → blend), replacing the earlier placeholder image.

Every conversion was validated against the real held-out test images (not just "does it load") before being deployed — including a caught case where the naive preprocessing pipeline was silently zeroing out the model's discriminative signal entirely, only revealed by checking accuracy against ground truth rather than trusting a clean HTTP 200.

### 2.3 Infrastructure inventory

| Resource | Purpose |
|---|---|
| S3 bucket | Stores `model.tar.gz` (both ONNX model files) for SageMaker |
| SageMaker serverless endpoint | Hosts `inference.py` — YOLO gate + classifier + Grad-CAM, 3072MB |
| ECR repo `fish-api` | Container image for the orchestration Lambda |
| ECR repo `fish-preprocess` | Container image for the segmentation Lambda |
| Lambda `fish-api` | FastAPI (via Mangum) behind a public Function URL, CORS-enabled |
| Lambda `fish-preprocess` | `rembg` segmentation, invoked by `fish-api` |
| IAM execution role | Scoped to `sagemaker:InvokeEndpoint` + `lambda:InvokeFunction` on the two specific resource ARNs — not broad account access |
| CloudWatch Logs | Per-function logs, used throughout for diagnosing cold-start/timeout issues |

### 2.4 Deployment workflow

Both Lambdas follow the same cycle:

```
docker build --platform linux/amd64 --provenance=false --sbom=false ...
docker tag ... 504509954111.dkr.ecr.us-east-1.amazonaws.com/<repo>:latest
docker push ...
# then, in the AWS console: Lambda → Deploy new image
```

SageMaker updates go through a small custom `deploy.py` (using `boto3` directly rather than the SDK's `Model.deploy(update_endpoint=True)`, which has a bug for serverless configs) that registers a new Model + EndpointConfig and swaps the existing endpoint onto it with zero downtime.

### 2.5 Problems found and fixed along the way

This is the real substance of the deployment work — a working demo isn't just "it returned 200 once":

| Symptom | Root cause | Fix |
|---|---|---|
| SageMaker `Worker died` on every request | `PyTorchModel(..., entry_point=...)` without `source_dir` silently never packaged `requirements.txt` — none of the ML dependencies were ever installed | Set `source_dir` explicitly |
| Still crashing after the above | `numpy` unpinned in `requirements.txt` resolved to 2.x, breaking the container's own pre-installed `scipy` (ABI break) | Pinned `numpy<2` |
| ONNX export of the classifier failed to load | `tf2onnx` has no rule for the `Erfc` op the model's GELU decomposes into | Swapped to tanh-approximation GELU before export |
| Browser requests failed with "Failed to fetch" | CORS configured in **both** FastAPI's `CORSMiddleware` and the Lambda Function URL, producing a duplicate `Access-Control-Allow-Origin: *, *` header that browsers reject outright | Consolidated CORS ownership into FastAPI only, disabled it at the Function URL |
| Backend server (originally on Render) hit an OOM/free-tier memory limit | Large uploaded images processed at full resolution created several ~50MB in-memory copies per request | Root-caused, then superseded by the move to Lambda (no longer memory-constrained) |
| Requests hanging 90–180+ seconds, eventually timing out | `cv2.grabCut` (the local fallback segmentation path) was running at full image resolution (e.g. 1600×1600) — confirmed via CloudWatch timing instrumentation and a local benchmark (134.6s full-res vs 13.1s downscaled) | Downscale to a capped working resolution before segmenting, scale the mask back up |
| `fish-preprocess` crashing on every invocation | Numba (a `rembg`/`pymatting` dependency) tries to write its JIT compile cache next to the read-only `site-packages` files — fails on Lambda's read-only filesystem | Set `NUMBA_CACHE_DIR=/tmp`, the one writable directory in a Lambda container |
| Segmentation Lambda still slow/failing after the above | Its memory had been reset to well below what heavy imports (`rembg`, `numba`, `onnxruntime`, `scipy`) need — Lambda's CPU allocation scales with memory | Increased to 3008MB |
| Grad-CAM panel showed a plain, unannotated crop | The original placeholder (`cam_img = crop.copy()`) was a stand-in left over from when Grad-CAM required TensorFlow's `GradientTape`, which the ONNX migration removed | Re-exported the classifier with a second output (feature map) and computed a real heatmap server-side |

Each of these was diagnosed from first principles — reproducing the failure directly against the deployed endpoint, reading CloudWatch logs line-by-line, and in several cases writing small local benchmarks (e.g. the GrabCut timing comparison) to confirm a root cause before shipping a fix, rather than guessing.

---

## 3. Frontend

Next.js 16 (App Router) app deployed on **Vercel**, auto-deploying from GitHub on every push to `main`. The API base URL is a `NEXT_PUBLIC_API_URL` environment variable (falling back to the Lambda Function URL in code), so the frontend can be repointed at a different backend without a code change.

**Live:**
- Frontend: `https://fish-freshness-vert.vercel.app`
- Backend: AWS Lambda Function URL (`fish-api`)

---

## 4. Why This Architecture

- **No idle cost, no cold-server risk of a free-tier host spinning down mid-demo** — everything scales to zero and back on demand.
- **Each service is independently deployable and debuggable** — a segmentation bug doesn't require redeploying the classifier, and vice versa.
- **The classifier and detector run through one shared, minimal `onnxruntime` runtime** instead of two full frameworks (`torch` + `tensorflow`) coexisting in the same process — smaller images, faster cold starts, less to go wrong.
- **Least-privilege IAM** — the orchestration Lambda can only invoke the two specific resources it needs, nothing else.
