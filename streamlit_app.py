import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from models.efficientnet_lite0 import EfficientNetLite0Tamper


# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# Load model
model = EfficientNetLite0Tamper(num_classes=2, pretrained=False)
model.load_state_dict(
    torch.load(
        "saved_models/efficient_lite0.pth",
        map_location=DEVICE
    )
)
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


def generate_gradcam(model, img_tensor):
    """
    Generate a Grad-CAM heatmap for the predicted class.
    """

    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # Last convolutional layer of EfficientNet-Lite0
    target_layer = model.base.conv_head

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks before forward/backward pass
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass for predicted class
    model.zero_grad()
    score = output[0, pred_class]
    score.backward()

    # Remove hooks after activations/gradients have been captured
    forward_handle.remove()
    backward_handle.remove()

    # Extract feature maps and gradients
    activations_array = activations[0].detach().cpu().numpy()[0]
    gradients_array = gradients[0].detach().cpu().numpy()[0]

    # Global-average-pool gradients to obtain channel weights
    weights = np.mean(gradients_array, axis=(1, 2))

    cam = np.zeros(
        activations_array.shape[1:],
        dtype=np.float32
    )

    for i, weight in enumerate(weights):
        cam += weight * activations_array[i]

    # ReLU
    cam = np.maximum(cam, 0)

    # Resize to model input size
    cam = cv2.resize(cam, (224, 224))

    # Normalize heatmap
    cam -= cam.min()

    cam_max = cam.max()
    if cam_max > 0:
        cam /= cam_max

    return cam, output


# Streamlit UI
st.set_page_config(
    layout="wide",
    page_title="AIShield Tamper Detection"
)

st.title("Image Tampering Detection (EfficientNet-Lite0 + Grad-CAM)")

uploaded = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    # Load and display uploaded image
    img = Image.open(uploaded).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        width=500
    )

    # Preprocess image
    img_tensor = transform(img)

    # Prediction + Grad-CAM
    cam, output = generate_gradcam(
        model,
        img_tensor
    )

    # Softmax probabilities
    probs = F.softmax(output, dim=1)[0]

    clean_prob = float(probs[0].item())
    tampered_prob = float(probs[1].item())

    # Determine predicted label
    if clean_prob > tampered_prob:
        label = "Clean"
        confidence = clean_prob * 100
    else:
        label = "Tampered"
        confidence = tampered_prob * 100

    # Prediction output
    st.markdown(
        f"## Prediction: **{label} "
        f"({confidence:.2f}% confidence)**"
    )

    # Probability display
    st.subheader("Prediction Probability")

    st.write(
        f"**Clean:** "
        f"{clean_prob * 100:.2f}%"
    )
    st.progress(clean_prob)

    st.write(
        f"**Tampered:** "
        f"{tampered_prob * 100:.2f}%"
    )
    st.progress(tampered_prob)

    # Prepare Grad-CAM overlay
    img_cv = np.array(img)
    img_cv = cv2.resize(
        img_cv,
        (224, 224)
    )

    heatmap = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        img_cv,
        0.55,
        heatmap,
        0.45,
        0
    )

    st.image(
        overlay,
        caption="Grad-CAM Overlay",
        width=500
    )

    # Interpretation
    st.subheader("Interpretation")

    st.write(
        """
        - **Red/Yellow areas** = regions that had higher influence on the model prediction
        - **Blue areas** = regions that had lower influence on the model prediction
        """
    )
