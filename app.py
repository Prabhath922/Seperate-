from __future__ import annotations
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ===========================
# ⚙️ PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="♻️ AI Waste Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================
# 🎨 CSS STYLING
# ===========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

.stApp {
    background: linear-gradient(145deg, #dfffe0, #f7fff5);
    font-family: 'Poppins', sans-serif;
    overflow: hidden;
}

/* Floating eco icons */
.eco-icon {
    position: fixed;
    opacity: 0.12;
    font-size: 60px;
    animation: floatAround 25s ease-in-out infinite;
    z-index: 0;
}

@keyframes floatAround {
    0%, 100% { transform: translate(0,0) rotate(0deg); }
    25% { transform: translate(60px,-60px) rotate(90deg); }
    50% { transform: translate(0,-100px) rotate(180deg); }
    75% { transform: translate(-60px,-60px) rotate(270deg); }
}

/* Header */
.header {
    text-align: center;
    background: white;
    margin: 20px auto;
    padding: 20px 30px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    width: 85%;
}
.header h1 {
    font-weight: 800;
    font-size: 46px;
    background: linear-gradient(135deg, #2d5016 0%, #5a9216 50%, #86ba3f 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header p {
    font-size: 18px;
    color: #3d6b1f;
    margin: 5px 0 0;
}

/* Columns */
.container {
    display: flex;
    justify-content: center;
    align-items: start;
    gap: 50px;
    margin-top: 40px;
    z-index: 10;
    position: relative;
}

/* Left Upload Box */
.upload-box {
    flex: 1;
    background: white;
    padding: 50px 30px;
    border-radius: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    text-align: center;
    min-height: 500px;
}
.upload-box h2 {
    color: #2d5016;
    font-weight: 700;
    margin-bottom: 10px;
}
.upload-icon {
    font-size: 100px;
    margin: 20px 0;
    animation: bounce 3s infinite ease-in-out;
}
@keyframes bounce {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

/* Right Bin Section */
.bins-box {
    flex: 1;
    background: white;
    border-radius: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    padding: 40px 20px;
    position: relative;
    min-height: 500px;
}
.bins-title {
    text-align: center;
    color: #2d5016;
    font-weight: 800;
    font-size: 32px;
}

/* Bin Grid */
.bin-grid {
    display: grid;
    grid-template-columns: repeat(2,1fr);
    gap: 20px;
    padding: 30px;
    opacity: 1;
    transition: opacity 0.5s ease;
}
.bin-card {
    background: linear-gradient(135deg,#ffffff,#f0f9f9);
    border-radius: 20px;
    padding: 40px 10px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transition: all 0.4s ease;
}
.bin-card:hover {
    transform: scale(1.05);
}
.bin-icon {
    font-size: 60px;
}
.bin-label {
    font-size: 22px;
    font-weight: 700;
    color: #2d5016;
    margin-top: 10px;
}

/* Highlighted Bin */
.highlighted-bin {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    border-radius: 30px;
    box-shadow: 0 15px 60px rgba(0,0,0,0.25);
    padding: 50px 40px;
    text-align: center;
    animation: popIn 0.6s ease-out;
}
@keyframes popIn {
    0% { transform: translate(-50%, -50%) scale(0); opacity: 0; }
    100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
}
.highlighted-bin .bin-icon {
    font-size: 130px;
    animation: wiggle 1s ease-in-out infinite;
}
@keyframes wiggle {
    0%,100% { transform: rotate(-5deg); }
    50% { transform: rotate(5deg); }
}
.confidence-badge {
    background: rgba(90,146,22,0.15);
    padding: 10px 25px;
    border-radius: 25px;
    margin-top: 20px;
    display: inline-block;
    font-weight: 700;
    color: #2d5016;
}
</style>
""", unsafe_allow_html=True)

# Floating eco icons
st.markdown("""
<div class="eco-icon" style="top: 10%; left: 10%;">🌿</div>
<div class="eco-icon" style="top: 30%; right: 8%; animation-delay: 4s;">🌍</div>
<div class="eco-icon" style="bottom: 20%; left: 12%; animation-delay: 6s;">♻️</div>
<div class="eco-icon" style="bottom: 35%; right: 10%; animation-delay: 2s;">🍃</div>
""", unsafe_allow_html=True)

# ===========================
# 🧠 MODEL SETUP
# ===========================
MODEL_PATH = "model.pth"

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            state = checkpoint["model_state"]
        else:
            state = checkpoint
        model.load_state_dict(state, strict=False)
    except Exception:
        st.warning("⚠️ Model not found — running in demo mode (random predictions).")
    model.eval()
    return model

model = load_model()

LABELS = ["Organic", "Paper", "Plastic", "Trash"]
EMOJIS = ["🍃", "📄", "🧴", "🗑️"]
BIN_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#6b7280"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def classify_image(img):
    try:
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(img_t)
            probs = F.softmax(logits, dim=1).squeeze(0)
            top_idx = torch.argmax(probs).item()
            conf = float(probs[top_idx].item())
        return LABELS[top_idx], conf
    except Exception:
        import random
        idx = random.randint(0, 3)
        return LABELS[idx], 0.5

# ===========================
# 🔄 STATE VARIABLES
# ===========================
if "label" not in st.session_state:
    st.session_state.label = None
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

# ===========================
# 🖥️ MAIN LAYOUT
# ===========================
st.markdown("""
<div class="header">
    <h1>♻️ AI Waste Classifier</h1>
    <p>Smart sorting for a cleaner planet — made for everyone 🌍</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

# -------- LEFT PANEL (UPLOAD) --------
with col1:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown('<div class="upload-icon">📸</div>', unsafe_allow_html=True)
    st.markdown("<h2>Upload or Capture Waste</h2>", unsafe_allow_html=True)
    method = st.radio("Select method", ["Upload Image", "Take Photo"], horizontal=True, label_visibility="collapsed")

    image = None
    if method == "Upload Image":
        uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
        if cam:
            image = Image.open(cam).convert("RGB")

    if image:
        label, conf = classify_image(image)
        st.session_state.label = label
        st.session_state.confidence = conf
        st.rerun()

    if st.session_state.label:
        if st.button("🔄 Try Another"):
            st.session_state.label = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -------- RIGHT PANEL (BINS) --------
with col2:
    st.markdown('<div class="bins-box">', unsafe_allow_html=True)
    st.markdown('<div class="bins-title">🗑️ Recycling Bins</div>', unsafe_allow_html=True)

    if st.session_state.label:
        idx = LABELS.index(st.session_state.label)
        emoji = EMOJIS[idx]
        color = BIN_COLORS[idx]
        st.markdown(f"""
        <div class="highlighted-bin" style="border:5px solid {color};">
            <div class="bin-icon">{emoji}</div>
            <div class="bin-label">{st.session_state.label} Bin</div>
            <div class="confidence-badge">Confidence: {int(st.session_state.confidence*100)}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="bin-grid">', unsafe_allow_html=True)
        for label, emoji in zip(LABELS, EMOJIS):
            st.markdown(f"""
            <div class="bin-card">
                <div class="bin-icon">{emoji}</div>
                <div class="bin-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
