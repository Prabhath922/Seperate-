from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from flask import Flask, render_template, request
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


def prepare_image(file_stream) -> torch.Tensor:
    image = Image.open(file_stream).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    return tensor


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        if file.filename:
            try:
                image_data = prepare_image(io.BytesIO(file.read()))
                with torch.no_grad():
                    outputs = model(image_data)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                top_idx = int(torch.argmax(probabilities).item())
                prediction = CLASS_NAMES[top_idx]
                confidence = float(probabilities[top_idx].item())
            except Exception as exc:  # pylint: disable=broad-except
                prediction = f"Error: {exc}"
                confidence = None

    return render_template("index.html", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)

