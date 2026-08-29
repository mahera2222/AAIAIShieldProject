import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def make_collage(root_dir="data/train", save_path="results/dataset_collage.png"):
    clean_dir = os.path.join(root_dir, "clean")
    tampered_dir = os.path.join(root_dir, "tampered")

    # Pick 6 clean and 6 tampered images
    clean_samples = random.sample(os.listdir(clean_dir), 6)
    tampered_samples = random.sample(os.listdir(tampered_dir), 6)

    images = []

    for img_name in clean_samples:
        img = Image.open(os.path.join(clean_dir, img_name)).convert("RGB")
        images.append(img.resize((224, 224)))

    for img_name in tampered_samples:
        img = Image.open(os.path.join(tampered_dir, img_name)).convert("RGB")
        images.append(img.resize((224, 224)))

    # Build collage 3×4
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))

    idx = 0
    for r in range(3):
        for c in range(4):
            axes[r][c].imshow(images[idx])
            if idx < 6:
                axes[r][c].set_title("Clean")
            else:
                axes[r][c].set_title("Tampered")
            axes[r][c].axis("off")
            idx += 1

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Collage saved to {save_path}")

if __name__ == "__main__":
    make_collage()
