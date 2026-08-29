import os
import shutil
import random

# Path to extracted CASIA dataset
CASIA_ROOT = "CASIA"

SOURCE_AU = f"{CASIA_ROOT}/Au"
SOURCE_TP = f"{CASIA_ROOT}/Tp"

DEST = "data"

TRAIN_CLEAN = f"{DEST}/train/clean"
TRAIN_TAMP = f"{DEST}/train/tampered"
VAL_CLEAN = f"{DEST}/val/clean"
VAL_TAMP = f"{DEST}/val/tampered"

# Reproducible train/validation split
random.seed(42)

os.makedirs(TRAIN_CLEAN, exist_ok=True)
os.makedirs(TRAIN_TAMP, exist_ok=True)
os.makedirs(VAL_CLEAN, exist_ok=True)
os.makedirs(VAL_TAMP, exist_ok=True)


def copy_split(src_folder, dest_train, dest_val, split_ratio=0.8):
    files = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        )
    ]

    random.shuffle(files)

    split = int(len(files) * split_ratio)

    train_files = files[:split]
    val_files = files[split:]

    for filename in train_files:
        shutil.copy(
            os.path.join(src_folder, filename),
            dest_train
        )

    for filename in val_files:
        shutil.copy(
            os.path.join(src_folder, filename),
            dest_val
        )

    print(
        f"{src_folder}: "
        f"{len(train_files)} train, "
        f"{len(val_files)} validation"
    )


copy_split(
    SOURCE_AU,
    TRAIN_CLEAN,
    VAL_CLEAN
)

copy_split(
    SOURCE_TP,
    TRAIN_TAMP,
    VAL_TAMP
)

print("CASIA dataset organized into train/validation successfully!")
