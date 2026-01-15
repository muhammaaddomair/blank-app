import os
import torch
import timm
import torch.nn.functional as F
from torchvision import transforms

# ================= CONFIG =================
MODEL_NAME = "tf_efficientnet_b4"   # MUST match training
NUM_CLASSES = 5

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infer_tfms = transforms.Compose([
    transforms.Resize((384, 384)),  # MUST match training
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= MODEL =================
def load_model(model_path="best_subset_model.pth"):
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    state = torch.load(model_path, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=True)

    model.to(DEVICE)
    model.eval()
    return model

# ================= PREDICT =================
@torch.no_grad()
def predict(model, pil_image):
    x = infer_tfms(pil_image).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]

    idx = int(torch.argmax(probs))
    conf = float(torch.max(probs))

    return CLASS_NAMES[idx], conf, idx, probs.cpu().numpy()
