# Waste Classifier Demo

This project trains a lightweight image classifier for common waste categories and exposes it through a very simple Flask web UI.

## Prerequisites

- Python 3.10 or newer
- A virtual environment (`python -m venv .venv`) is recommended
- Dataset directory at `/Users/prabhathsundarapalli/Downloads/dataset-resized` with subfolders `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`

Install dependencies (PyTorch wheels are provided for macOS ARM and Intel, as well as Linux/Windows; the requirement ranges let pip choose the newest compatible wheel):

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py --epochs 8 --finetune-epochs 3
```

The script saves `model.pth` and `labels.json` in the project root. Use `--help` to see optional arguments (dataset path, batch size, learning rates, epochs, etc.).

## Run the web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser, upload an image of waste, and the app will return the predicted class with a confidence score.
