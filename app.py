"""
Streamlit app for the Hybrid Classification-Regression Framework
for Ordinal Diabetic Retinopathy (DR) Grading — APTOS 2019.

Based on: Group-21 Hybrid Classification-Regression Framework
(EfficientNet-B3 dual-head: classification + regression, QWK-optimized thresholds,
Grad-CAM++ explainability).
"""

import io
import json
import os

import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.special import softmax

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 288
CHECKPOINT_PATH = os.environ.get("DR_CHECKPOINT_PATH", "models/best_hybrid_model.pth")
THRESHOLDS_PATH = os.environ.get("DR_THRESHOLDS_PATH", "models/thresholds.json")
DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

CLASS_NAMES = [
    "Grade 0 — No DR",
    "Grade 1 — Mild NPDR",
    "Grade 2 — Moderate NPDR",
    "Grade 3 — Severe NPDR",
    "Grade 4 — Proliferative DR",
]

CLASS_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Early microaneurysms present. Routine monitoring recommended.",
    2: "Haemorrhages and exudates visible. Referral for evaluation advised.",
    3: "Extensive haemorrhages across retinal quadrants. Urgent referral advised.",
    4: "Neovascularization present — highest severity. Immediate ophthalmology referral required.",
}


# --------------------------------------------------------------------------
# Model definition (matches the training notebook)
# --------------------------------------------------------------------------
class DRHybrid(nn.Module):
    def __init__(self, model_name="efficientnet_b3", num_classes=5, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        nf = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.fc_class = nn.Linear(nf, num_classes)
        self.fc_reg = nn.Linear(nf, 1)

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return self.fc_class(features), self.fc_reg(features).squeeze(-1)


# --------------------------------------------------------------------------
# Preprocessing (Ben Graham method, as used in the notebook)
# --------------------------------------------------------------------------
def crop_image_from_gray(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def apply_ben_graham(img_rgb, size=IMG_SIZE, sigma_x=10):
    img = crop_image_from_gray(img_rgb)
    img = cv2.resize(img, (size, size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma_x), -4, 128)
    return img


def to_tensor(img_uint8):
    img = img_uint8.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(checkpoint_bytes: bytes | None, checkpoint_path: str):
    model = DRHybrid().to(DEVICE)
    loaded_real_weights = False
    load_msg = ""

    state_dict = None
    if checkpoint_bytes is not None:
        try:
            state_dict = torch.load(io.BytesIO(checkpoint_bytes), map_location=DEVICE)
            load_msg = "Loaded checkpoint uploaded by user."
        except Exception as e:
            load_msg = f"Could not read uploaded checkpoint: {e}"
    elif os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            load_msg = f"Loaded checkpoint from {checkpoint_path}."
        except Exception as e:
            load_msg = f"Could not read checkpoint at {checkpoint_path}: {e}"

    if state_dict is not None:
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        try:
            model.load_state_dict(state_dict, strict=False)
            loaded_real_weights = True
        except Exception as e:
            load_msg = f"Checkpoint found but incompatible with model definition: {e}"

    model.eval()
    return model, loaded_real_weights, load_msg


@st.cache_resource(show_spinner=False)
def load_thresholds(path: str):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_THRESHOLDS


def thresholds_to_class(value, thresholds):
    t = sorted(thresholds)
    return int(np.digitize([value], t)[0])


def run_gradcam(model, input_tensor, rgb_img_float):
    target_layers = [model.backbone.conv_head]
    cam = GradCAMPlusPlus(model=model_wrapper_for_cam(model), target_layers=target_layers)
    # targets=None -> Grad-CAM explains the highest-scoring (predicted) class
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    return visualization


class ClassifierOnlyWrapper(nn.Module):
    """Grad-CAM needs a single scalar/class output; wrap to expose the classification head."""

    def __init__(self, dr_model):
        super().__init__()
        self.dr_model = dr_model

    def forward(self, x):
        logits, _ = self.dr_model(x)
        return logits


def model_wrapper_for_cam(model):
    return ClassifierOnlyWrapper(model)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
_page_icon = Image.open(LOGO_PATH) if os.path.exists(LOGO_PATH) else "🩺"

st.set_page_config(
    page_title="RetinaAI — DR Grading",
    page_icon=_page_icon,
    layout="wide",
)

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.title("🩺 About this project")
    st.markdown(
        """
**Hybrid Classification-Regression Framework**
for Ordinal Diabetic Retinopathy Grading (APTOS 2019)

Dual-head **EfficientNet-B3**:
- Classification head → 5-way grade
- Regression head → continuous severity, thresholds tuned via Nelder-Mead to
  maximize **Quadratic Weighted Kappa (QWK)**

**Reported results (paper / repo):**
| Strategy | Accuracy | QWK |
|---|---|---|
| Classification Ensemble | 86.08% | 0.9113 |
| Regression Ensemble | 82.67% | **0.9202** |
| 4-Fold CV (mean) | 82.49% | 0.9131 |
        """
    )
    st.divider()
    st.subheader("Model weights")
    uploaded_ckpt = st.file_uploader(
        "Upload trained checkpoint (.pth)", type=["pth", "pt"]
    )
    st.caption(
        "If no checkpoint is uploaded and none is bundled with the app, "
        "the app runs in **demo mode** with an ImageNet-pretrained backbone only "
        "(untrained heads) — predictions won't be clinically meaningful."
    )
    st.divider()
    st.caption(
        "⚠️ Research / educational project only. Not a certified medical device. "
        "Not for clinical diagnosis."
    )

header_col1, header_col2 = st.columns([1, 5])
with header_col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
with header_col2:
    st.title("Diabetic Retinopathy Severity Grading")
    st.caption(
        "Upload a retinal fundus photograph to grade DR severity (0–4) using the "
        "hybrid classification-regression model, with Grad-CAM++ explainability."
    )

ckpt_bytes = uploaded_ckpt.read() if uploaded_ckpt is not None else None
model, loaded_real_weights, load_msg = load_model(ckpt_bytes, CHECKPOINT_PATH)
thresholds = load_thresholds(THRESHOLDS_PATH)

if loaded_real_weights:
    st.success(load_msg)
else:
    st.warning(
        "⚠️ **Demo mode** — no trained checkpoint loaded. "
        + (load_msg + " " if load_msg else "")
        + "Upload your `best_hybrid_model.pth` (or `best_regression_model.pth`) "
        "in the sidebar to get real predictions."
    )

uploaded_file = st.file_uploader(
    "Upload a fundus image", type=["png", "jpg", "jpeg"], key="fundus_uploader"
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_img)

    processed = apply_ben_graham(img_rgb)
    input_tensor = to_tensor(processed).to(DEVICE)

    with st.spinner("Running inference..."):
        with torch.no_grad():
            logits, reg_output = model(input_tensor)
            probs = softmax(logits.cpu().numpy()[0])
            reg_value = float(reg_output.cpu().numpy()[0])

        cls_pred = int(np.argmax(probs))
        reg_pred = thresholds_to_class(reg_value, thresholds)

        # Weighted fusion as in the paper: 0.6 classification / 0.4 regression
        reg_probs = np.zeros(5)
        reg_probs[min(reg_pred, 4)] = 1.0
        blended_probs = 0.6 * probs + 0.4 * reg_probs
        final_pred = int(np.argmax(blended_probs))

        rgb_float = processed.astype(np.float32) / 255.0
        try:
            cam_image = run_gradcam(model, input_tensor, rgb_float)
        except Exception as e:
            cam_image = None
            cam_error = str(e)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)
    with col2:
        st.subheader("Preprocessed (Ben Graham)")
        st.image(processed, use_container_width=True)
    with col3:
        st.subheader("Grad-CAM++")
        if cam_image is not None:
            st.image(cam_image, use_container_width=True)
        else:
            st.info("Grad-CAM unavailable for this run.")

    st.divider()
    st.subheader("Prediction")

    r1, r2 = st.columns([2, 1])
    with r1:
        st.markdown(f"### {CLASS_NAMES[final_pred]}")
        st.write(CLASS_DESCRIPTIONS[final_pred])
        st.caption(
            f"Classification head grade: **{cls_pred}** · "
            f"Regression score: **{reg_value:.2f}** → grade **{reg_pred}** · "
            f"Blended (0.6/0.4) final grade: **{final_pred}**"
        )
    with r2:
        st.metric("Confidence (classification head)", f"{probs[cls_pred] * 100:.1f}%")

    st.write("Class probability distribution (classification head):")
    st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})

    st.warning(
        "This tool is for research/educational demonstration only and must not "
        "be used to make real clinical decisions."
    )
else:
    st.info("👆 Upload a fundus image (PNG/JPG) to get a prediction.")