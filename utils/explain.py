import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance

# ---------------------------
# 1. GRAD-CAM IMPLEMENTATION
# ---------------------------

def get_gradcam(model, img_tensor, target_layer):
    """
    Compute Grad-CAM heatmap for a MobileNetV3 model.
    """

    activations = []
    gradients = []

    # Hook: forward pass
    def forward_hook(module, input, output):
        activations.append(output)

    # Hook: backward pass
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    model.eval()
    img_tensor = img_tensor.unsqueeze(0)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backprop for target class
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    # Extract
    grad = gradients[0]          # [1, C, H, W]
    act = activations[0]         # [1, C, H, W]

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  
    cam = torch.sum(weights * act, dim=1).squeeze().detach().cpu().numpy()

    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam

    return cam, pred_class


# ---------------------------
# 2. ERROR LEVEL ANALYSIS (ELA)
# ---------------------------
def compute_ela(image_path, quality=90):
    image = Image.open(image_path).convert("RGB")

    # Save temporary JPEG
    temp_path = "temp_ela.jpg"
    image.save(temp_path, "JPEG", quality=quality)

    compressed = Image.open(temp_path)
    ela_image = ImageChops.difference(image, compressed)

    # Enhance the difference
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return np.array(ela_image)


# ---------------------------
# 3. NOISE RESIDUAL MAP
# ---------------------------
def compute_noise_residual(image):
    """
    High-pass filter to extract image manipulation traces.
    """
    blur = cv2.GaussianBlur(image, (5,5), 0)
    residual = cv2.absdiff(image, blur)
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    return residual


# ---------------------------
# 4. MAIN FORENSIC EXPLAINER
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def explain_image(model, image_path, device="cpu"):
    """
    Runs: Grad-CAM + ELA + Noise Residual
    Returns results for Streamlit UI.
    """

    # 1️⃣ Load Image
    pil_img = Image.open(image_path).convert("RGB")
    img = np.array(pil_img)
    img_tensor = transform(pil_img).to(device)

    # 2️⃣ GRAD-CAM (target last conv layer)
    target_layer = model.model.features[12]
    cam, pred_class = get_gradcam(model, img_tensor, target_layer)

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    gradcam_overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 3️⃣ ELA Map
    ela_map = compute_ela(image_path)

    # 4️⃣ Noise Residual
    noise_map = compute_noise_residual(img)

    # 5️⃣ Why flagged as tampered?
    explanation = ""
    if pred_class == 1:
        explanation = (
            "⚠ *Model marked this as **TAMPERED*** because:\n"
            "- Grad-CAM highlights unnatural texture boundaries.\n"
            "- ELA shows high recompression artifacts in specific regions.\n"
            "- Noise residual reveals inconsistent noise patterns.\n"
            "These combined signals are common signs of digital manipulation."
        )
    else:
        explanation = (
            "✅ *Model marked this as **CLEAN***.\n"
            "No strong inconsistencies found in:\n"
            "- Texture heatmap (Grad-CAM)\n"
            "- JPEG recompression artifacts (ELA)\n"
            "- Noise pattern consistency\n"
        )

    return {
        "prediction": "Tampered" if pred_class == 1 else "Clean",
        "gradcam_overlay": gradcam_overlay,
        "ela_map": ela_map,
        "noise_map": noise_map,
        "text_explanation": explanation
    }
