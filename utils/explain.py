import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance


# Match the EfficientNet-Lite0 training/inference preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_gradcam(model, img_tensor, target_layer, device="cpu"):
    """
    Compute a Grad-CAM heatmap for the predicted class.
    """

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.eval()

    img_tensor = img_tensor.unsqueeze(0).to(device)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    forward_handle.remove()
    backward_handle.remove()

    grad = gradients[0]
    act = activations[0]

    weights = torch.mean(
        grad,
        dim=(2, 3),
        keepdim=True
    )

    cam = torch.sum(
        weights * act,
        dim=1
    ).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, pred_class, output


def compute_ela(image_path, quality=90):
    """
    Compute an Error Level Analysis (ELA) image.
    """

    image = Image.open(image_path).convert("RGB")

    temp_path = "temp_ela.jpg"
    image.save(temp_path, "JPEG", quality=quality)

    compressed = Image.open(temp_path).convert("RGB")
    ela_image = ImageChops.difference(image, compressed)

    extrema = ela_image.getextrema()
    max_diff = max(channel[1] for channel in extrema)

    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return np.array(ela_image)


def compute_noise_residual(image):
    """
    Compute a simple high-frequency noise residual map.
    """

    blur = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    residual = cv2.absdiff(
        image,
        blur
    )

    residual = cv2.normalize(
        residual,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return residual


def explain_image(model, image_path, device="cpu"):
    """
    Run model prediction together with Grad-CAM, ELA,
    and noise-residual visualizations.

    ELA and noise residual are supplementary forensic
    visualizations and do not directly affect the model prediction.
    """

    pil_img = Image.open(image_path).convert("RGB")
    img = np.array(pil_img)

    img_tensor = transform(pil_img)

    # EfficientNet-Lite0 final convolutional layer
    target_layer = model.base.conv_head

    cam, pred_class, output = get_gradcam(
        model,
        img_tensor,
        target_layer,
        device=device
    )

    probabilities = F.softmax(
        output,
        dim=1
    )[0]

    clean_prob = float(
        probabilities[0].item()
    )

    tampered_prob = float(
        probabilities[1].item()
    )

    cam_resized = cv2.resize(
        cam,
        (img.shape[1], img.shape[0])
    )

    heatmap = cv2.applyColorMap(
        np.uint8(cam_resized * 255),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    gradcam_overlay = cv2.addWeighted(
        img,
        0.6,
        heatmap,
        0.4,
        0
    )

    ela_map = compute_ela(image_path)
    noise_map = compute_noise_residual(img)

    prediction = (
        "Tampered"
        if pred_class == 1
        else "Clean"
    )

    explanation = (
        f"Model prediction: {prediction}. "
        "Grad-CAM highlights regions that influenced the model's decision. "
        "ELA and noise residual maps are provided as supplementary "
        "image-forensics visualizations."
    )

    return {
        "prediction": prediction,
        "clean_probability": clean_prob,
        "tampered_probability": tampered_prob,
        "gradcam_overlay": gradcam_overlay,
        "ela_map": ela_map,
        "noise_map": noise_map,
        "text_explanation": explanation
    }
