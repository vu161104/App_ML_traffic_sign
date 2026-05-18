"""
Flask API — HOG + SVM Traffic Sign Classifier
Truy cập: http://127.0.0.1:5000
"""

import os, pickle, warnings
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from skimage.feature import hog

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hog_svm_model.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

clf       = bundle["clf"]
scaler    = bundle["scaler"]
LABEL_MAP = bundle["label_map"]
IMG_SIZE  = bundle["img_size"]
HOG_CFG   = bundle["hog_config"]

def extract_hog(image_bgr):
    img = cv2.resize(image_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = hog(
        img,
        orientations    = HOG_CFG["orientations"],
        pixels_per_cell = tuple(HOG_CFG["pixels_per_cell"]),
        cells_per_block = tuple(HOG_CFG["cells_per_block"]),
        block_norm      = HOG_CFG["block_norm"],
        transform_sqrt  = HOG_CFG["transform_sqrt"],
        feature_vector  = True,
    )
    return feat.astype(np.float32)

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ── Serve giao diện HTML ──────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

# ── Health check ──────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ── Predict ───────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không tìm thấy field 'image'."}), 400
    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Không thể đọc ảnh."}), 400
    feat        = extract_hog(img_bgr).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred_class  = int(clf.predict(feat_scaled)[0])
    proba       = clf.predict_proba(feat_scaled)[0]
    top3_idx    = np.argsort(proba)[::-1][:3]
    top3 = [
        {
            "class_id":   int(i),
            "label":      LABEL_MAP.get(int(i), str(i)),
            "confidence": round(float(proba[i]) * 100, 2),
        }
        for i in top3_idx
    ]
    return jsonify({
        "predicted_class": pred_class,
        "predicted_label": LABEL_MAP.get(pred_class, str(pred_class)),
        "confidence":      round(float(proba[pred_class]) * 100, 2),
        "top3":            top3,
        "image_size":      {"width": img_bgr.shape[1], "height": img_bgr.shape[0]},
    })

# ── Labels list ───────────────────────────────
@app.route("/labels")
def labels():
    return jsonify({
        "labels": [{"class_id": k, "label": v} for k, v in sorted(LABEL_MAP.items())]
    })

if __name__ == "__main__":
    print("=" * 50)
    print("  Traffic Sign HOG+SVM API")
    print("  Giao dien: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)