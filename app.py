import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DR Screening System",
    page_icon="🩺",
    layout="wide",
)

# ================= CONSTANTS =================
MODEL_NAME = "tf_efficientnet_b3"
MODEL_PATH = "best_subset_model.pth"
NUM_CLASSES = 5
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

# ================= STYLING =================
st.markdown("""
<style>
html, body, .stApp { background:#ffffff; color:#0f172a; }
.block-container { max-width: 1200px; padding-top: 1.5rem; }

h1 { font-size:28px; font-weight:700; }
h2 { font-size:20px; font-weight:600; }
.label { font-size:12px; color:#64748b; }
.metric { font-size:22px; font-weight:700; }

.card {
  background:#ffffff;
  border:1px solid #e5e7eb;
  border-radius:14px;
  padding:20px;
}

.stButton button {
  background:linear-gradient(135deg,#2563eb,#0ea5e9);
  color:white;
  font-weight:600;
  border-radius:10px;
  padding:0.7rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ================= TRANSFORMS =================
infer_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model

MODEL = load_model()

# ================= BIOMARKERS =================
def binarize_vessels(rgb):
    green = rgb[:, :, 1]
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(green)
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )
    return (binary > 0).astype(np.uint8)

def vessel_density(b): 
    return float(b.sum() / b.size)

def vessel_tortuosity(b):
    if b.sum() == 0: return 0.0
    sk = skeletonize(b.astype(bool))
    return float(sk.sum() / (np.count_nonzero(sk) + 1e-6))

def fractal_dimension(b):
    Z = b.astype(np.uint8)
    n = min(Z.shape)
    Z = Z[:n, :n]
    sizes = 2 ** np.arange(1, int(np.log2(n)))
    counts = [np.count_nonzero(
        np.add.reduceat(
            np.add.reduceat(Z, np.arange(0,n,s), axis=0),
            np.arange(0,n,s), axis=1
        )
    ) for s in sizes]
    counts = np.array(counts)
    valid = counts > 0
    if valid.sum() < 2: return 0.0
    return float(-np.polyfit(np.log(sizes[valid]), np.log(counts[valid]), 1)[0])

# ================= GRAD-CAM =================
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self.activations = o
        o.requires_grad_(True)
        o.register_hook(lambda g: setattr(self, "gradients", g))

    def generate(self, x, idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[:, idx].backward()
        w = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam[0].detach().cpu().numpy()

def overlay_cam(img, cam, alpha):
    cam = cv2.resize(cam, img.shape[1::-1])
    heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1-alpha, heat, alpha, 0)

# ================= HEADER =================
st.markdown("<h1>Diabetic Retinopathy Screening</h1>", unsafe_allow_html=True)
st.markdown("<p>AI-assisted severity assessment with vascular biomarkers and explainable visual evidence.</p>", unsafe_allow_html=True)

# ================= MAIN =================
uploaded = st.file_uploader("Upload retinal fundus image", ["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.session_state.image = image

    col_left, col_right = st.columns([0.45, 0.55], gap="large")

    # ===== LEFT =====
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, width=360)

        st.markdown("### Actions")

        run_pred = st.button("Run DR Screening", use_container_width=True)
        run_bio = st.button("Compute Vascular Biomarkers", use_container_width=True)
        run_cam = st.button("Generate Grad-CAM", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ===== RIGHT =====
    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # --- DR SCREENING ---
        if run_pred:
            with torch.no_grad():
                x = infer_tfms(image).unsqueeze(0).to(DEVICE)
                logits = MODEL(x)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                idx = int(np.argmax(probs))

            st.markdown("### DR Screening Result")
            st.markdown(f"<div class='metric'>{CLASS_NAMES[idx]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='label'>Confidence</div><div class='metric'>{probs[idx]*100:.2f}%</div>", unsafe_allow_html=True)
            st.session_state.pred_idx = idx

        # --- BIOMARKERS ---
        if run_bio:
            img_np = np.array(image)
            b = binarize_vessels(img_np)

            density = vessel_density(b)
            tort = vessel_tortuosity(b)
            fract = fractal_dimension(b)

            st.markdown("### Vascular Biomarkers")

            c1, c2, c3 = st.columns(3)
            c1.metric("Vessel Density", f"{density:.4f}")
            c2.metric("Tortuosity", f"{tort:.4f}")
            c3.metric("Fractal Dimension", f"{fract:.4f}")

            # Line graph
            fig, ax = plt.subplots()
            ax.plot(
                ["Density", "Tortuosity", "Fractal Dim"],
                [density, tort, fract],
                marker="o"
            )
            ax.set_ylabel("Value")
            ax.set_title("Vascular Biomarker Profile")
            st.pyplot(fig)

        # --- GRAD CAM ---
        if run_cam and "pred_idx" in st.session_state:
            alpha = st.slider("Overlay intensity", 0.1, 0.8, 0.42, 0.01)
            x = infer_tfms(image).unsqueeze(0).to(DEVICE)
            cam = GradCAM(MODEL, MODEL.conv_head).generate(x, st.session_state.pred_idx)
            overlay = overlay_cam(np.array(image), cam, alpha)
            st.image(overlay, caption="Grad-CAM Heatmap", width=420)

        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown(
    "<p style='font-size:11px;color:#64748b;text-align:center;margin-top:30px;'>"
    "Research and educational prototype. Not a medical device.</p>",
    unsafe_allow_html=True
)
