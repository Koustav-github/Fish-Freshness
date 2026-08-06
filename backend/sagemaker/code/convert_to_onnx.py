"""
One-time conversion of both models to ONNX, so the deployed inference.py can
run everything through a single onnxruntime session instead of loading both
torch (for YOLO) and tensorflow (for the Keras classifier).

Requires (local machine only, not needed at deploy time):
    uv pip install ultralytics tensorflow tf2onnx --python .venv/Scripts/python.exe

Run from backend/sagemaker/code/:
    ../../.venv/Scripts/python.exe convert_to_onnx.py

Outputs, written next to the source files in ../models/:
    yolo_detection_model.onnx
    fish_classifier.onnx
"""

import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def convert_yolo():
    from ultralytics import YOLO

    pt_path = os.path.join(MODELS_DIR, "yolo_detection_model.pt")
    print(f"Exporting YOLO model: {pt_path}")

    model = YOLO(pt_path)
    exported_path = model.export(format="onnx", imgsz=640)
    print(f"YOLO ONNX export written to: {exported_path}")


def convert_classifier():
    import tensorflow as tf
    import tf2onnx

    keras_path = os.path.join(MODELS_DIR, "fish_classifier.keras")
    onnx_path = os.path.join(MODELS_DIR, "fish_classifier.onnx")
    print(f"Exporting Keras classifier: {keras_path}")

    model = tf.keras.models.load_model(keras_path)

    # tf2onnx has no conversion rule for the Erfc op that this model's exact
    # (erf-based) GELU decomposes into — fails regardless of opset. Swap
    # `dense`/`dense_1` to the tanh-approximation GELU variant (built from
    # ops tf2onnx does support) before export. This doesn't touch any
    # weights — GELU has no trainable parameters of its own, it's a pointwise
    # nonlinearity applied after each Dense layer's linear transform — so
    # the swap only changes ~1e-3-level numerical approximation, not the
    # learned behavior.
    def clone_fn(layer):
        config = layer.get_config()
        if config.get("activation") == "gelu":
            config = dict(config)
            config["activation"] = lambda x: tf.keras.activations.gelu(x, approximate=True)
            return layer.__class__.from_config(config)
        return layer.__class__.from_config(layer.get_config())

    export_model = tf.keras.models.clone_model(model, clone_function=clone_fn)
    export_model.set_weights(model.get_weights())

    # Matches the model's real input contract: (batch, 224, 224, 3) NHWC,
    # raw 0-255 float32 — the model's own Rescaling layer normalizes
    # internally, and that layer is preserved as part of the exported graph.
    input_signature = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

    tf2onnx.convert.from_keras(
        export_model,
        input_signature=input_signature,
        opset=17,
        output_path=onnx_path,
    )
    print(f"Classifier ONNX export written to: {onnx_path}")


if __name__ == "__main__":
    convert_yolo()
    convert_classifier()
    print("Done. Re-validate both .onnx outputs against the real labeled images before deploying them.")
