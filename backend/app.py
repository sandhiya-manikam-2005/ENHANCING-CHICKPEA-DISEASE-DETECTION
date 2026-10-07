"""
Flask REST API — Chickpea Disease Detection
────────────────────────────────────────────
Endpoints:
  POST /predict          – Upload leaf image, get disease prediction
  GET  /health           – Server health check
  GET  /classes          – List all disease classes
  GET  /metrics          – Trained model metrics
  POST /preprocess       – Return preprocessed image (debug endpoint)
"""

import os
import sys
import json
import traceback
import time

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    get_model, DISEASE_CLASSES, CLASS_DESCRIPTIONS, CLASS_TREATMENTS
)
from preprocessing import (
    preprocess_for_inference, pil_to_base64, IMAGE_SIZE
)


# ─── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="../frontend")
CORS(app)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pth")
NUM_CLASSES = 4

model = None   # loaded lazily on first request / explicit call


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    global model
    if model is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] No trained model found at {MODEL_PATH}. "
              f"Using random-weight model (run train.py first).")
        m = get_model(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
        m.eval()
        model = m.to(DEVICE)
        return False   # not trained

    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    m     = get_model(num_classes=ckpt.get("num_classes", NUM_CLASSES),
                      image_size=ckpt.get("image_size", IMAGE_SIZE))
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    model = m.to(DEVICE)
    print(f"[OK] Loaded model (val_acc={ckpt.get('val_acc', 'N/A'):.2%})")
    return True


# ─── Inference helper ─────────────────────────────────────────────────────────

def predict_image(image_bytes: bytes) -> dict:
    """Run full pipeline on raw image bytes, return prediction dict."""
    load_model()

    t_start = time.time()

    # Preprocess
    tensor, pil_proc, pil_orig = preprocess_for_inference(image_bytes)
    tensor = tensor.to(DEVICE)

    # Inference
    with torch.no_grad():
        logits      = model(tensor)                      # (1, 4)
        probs       = F.softmax(logits, dim=1)[0]        # (4,)
        pred_idx    = probs.argmax().item()
        pred_label  = DISEASE_CLASSES[pred_idx]
        confidence  = probs[pred_idx].item()

    elapsed = (time.time() - t_start) * 1000   # ms

    # All class probabilities
    class_probs = {
        DISEASE_CLASSES[i]: round(probs[i].item() * 100, 2)
        for i in range(len(DISEASE_CLASSES))
    }

    # Preprocessed image for display
    proc_b64 = pil_to_base64(pil_proc)

    return {
        "prediction":    pred_label,
        "confidence":    round(confidence * 100, 2),
        "description":   CLASS_DESCRIPTIONS[pred_label],
        "treatments":    CLASS_TREATMENTS[pred_label],
        "all_probs":     class_probs,
        "preprocessed":  proc_b64,
        "inference_ms":  round(elapsed, 1),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "device":  str(DEVICE),
        "model":   "loaded" if model else "not_loaded",
        "version": "1.0.0"
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({
        "classes": [
            {
                "id":          i,
                "name":        name,
                "description": CLASS_DESCRIPTIONS[name],
                "treatments":  CLASS_TREATMENTS[name],
            }
            for i, name in DISEASE_CLASSES.items()
        ]
    })


@app.route("/metrics", methods=["GET"])
def get_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "No metrics file found. Train the model first."}), 404
    with open(metrics_path) as f:
        return jsonify(json.load(f))


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (key='image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {"png", "jpg", "jpeg", "bmp", "webp"}
    ext     = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        image_bytes = file.read()
        result      = predict_image(image_bytes)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/preprocess", methods=["POST"])
def preprocess_debug():
    """Debug endpoint: returns the preprocessed image as base64."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    try:
        image_bytes        = request.files["image"].read()
        _, pil_proc, pil_orig = preprocess_for_inference(image_bytes)
        return jsonify({
            "original":     pil_to_base64(pil_orig),
            "preprocessed": pil_to_base64(pil_proc),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve frontend (single-page)
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if path and os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    return send_from_directory(frontend_dir, "index.html")


# ─── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Chickpea Disease Detection API")
    print("=" * 55)
    load_model()
    port = int(os.environ.get("PORT", 5000))
    print(f"  Server: http://localhost:{port}")
    print(f"  Device: {DEVICE}")
    app.run(host="0.0.0.0", port=port, debug=False)
