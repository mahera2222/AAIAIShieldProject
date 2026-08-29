import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from models.efficientnet_lite0 import EfficientNetLite0Tamper

# Loading model
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = EfficientNetLite0Tamper(num_classes=2)
model.load_state_dict(torch.load("saved_models/efficient_lite0.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Grad-CAM Helper
def generate_gradcam(model, img_tensor):

    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # Target layer (efficientnet_lite0 last Conv)
    target_layer = model.base.conv_head

    # Hook for activations
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    forward_handle.remove()
    backward_handle.remove()

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    score = output[0, pred_class]
    score.backward()

    # Extract CAM
    act = activations[0].detach().cpu().numpy()[0]  # shape: [C, H, W]
    grad = gradients[0].detach().cpu().numpy()[0]   # shape: [C, H, W]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam_max = cam.max()
    if cam_max > 0:
         cam = cam / cam_max

    return cam, output


# Streamlit UI
st.set_page_config(layout="wide", page_title="AIShield Tamper Detection")

st.title(" Image Tampering Detection (EfficientNet-Lite0 + Grad-CAM)")

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded:

    # display uploaded image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=500)

    # preprocess
    img_tensor = transform(img)

    # Grad-CAM + prediction
    cam, output = generate_gradcam(model, img_tensor)

    # Softmax probabilities
    probs = F.softmax(output, dim=1)[0]
    clean_prob = float(probs[0].item())
    tampered_prob = float(probs[1].item())

    # Determine label
    label = "Clean" if clean_prob > tampered_prob else "Tampered"
    confidence = max(clean_prob, tampered_prob) * 100

    # Prediction Output
    st.markdown(f"## Prediction: **{label} ({confidence:.2f}% confidence)**")

    # Probability bars
    st.subheader("📊 Prediction Probability")

    st.write(f"**Clean:** {clean_prob * 100:.2f}%")
    st.progress(clean_prob)

    st.write(f"**Tampered:** {tampered_prob * 100:.2f}%")
    st.progress(tampered_prob)

    # Grad-CAM Overlay
    img_cv = np.array(img)
    img_cv = cv2.resize(img_cv, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_cv, 0.55, heatmap, 0.45, 0)

    st.image(overlay, caption="Grad-CAM Overlay", width=500)

    # Interpretation
    st.subheader(" Interpretation")
    st.write("""
    - **Red/Yellow areas** = high-attention regions used by the model  
    - **Blue areas** = low-attention regions  
    """)

