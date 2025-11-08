from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pth"
LABELS_PATH = BASE_DIR / "labels.json"
IMG_SIZE = 224

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    raise RuntimeError("Model or labels not found. Train the model first by running `python train_model.py`.")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, len(CLASS_NAMES))
model.load_state_dict(checkpoint["model_state"])
model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Mapping from waste types to bin names
WASTE_TO_BIN = {
    "cardboard": "paper",
    "glass": "trash",
    "metal": "trash",
    "organic": "organic",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
}


def get_bin_name(waste_type: str) -> str:
    """Map waste type to bin name."""
    return WASTE_TO_BIN.get(waste_type.lower(), "trash")


def prepare_image(file_stream: io.BytesIO | bytes) -> torch.Tensor:
    if isinstance(file_stream, bytes):
        file_stream = io.BytesIO(file_stream)
    with Image.open(file_stream) as image:
        image = image.convert("RGB")
        tensor = preprocess(image).unsqueeze(0)
    return tensor


def classify_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    image_data = prepare_image(image_bytes)
    with torch.no_grad():
        outputs = model(image_data)
        probabilities = torch.softmax(outputs, dim=1)[0]
    top_idx = int(torch.argmax(probabilities).item())
    prediction = CLASS_NAMES[top_idx]
    bin_name = get_bin_name(prediction)
    confidence = float(probabilities[top_idx].item())
    return {"prediction": prediction, "bin": bin_name, "confidence": confidence}


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    bin_name = None
    confidence = None

    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        if file.filename:
            try:
                result = classify_image_bytes(file.read())
                prediction = result["prediction"]
                bin_name = result["bin"]
                confidence = result["confidence"]
            except Exception as exc:  # pylint: disable=broad-except
                prediction = f"Error: {exc}"
                bin_name = None
                confidence = None

    return render_template("index.html", prediction=prediction, bin=bin_name, confidence=confidence)


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    try:
        result = classify_image_bytes(file.read())
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)

