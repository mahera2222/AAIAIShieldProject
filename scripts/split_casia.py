import os
import shutil
import random

# Path to your extracted CASIA dataset
CASIA_ROOT = "CASIA"

SOURCE_AU = f"{CASIA_ROOT}/Au"
SOURCE_TP = f"{CASIA_ROOT}/Tp"

DEST = "data"

TRAIN_CLEAN = f"{DEST}/train/clean"
TRAIN_TAMP = f"{DEST}/train/tampered"
TEST_CLEAN = f"{DEST}/test/clean"
TEST_TAMP = f"{DEST}/test/tampered"

os.makedirs(TRAIN_CLEAN, exist_ok=True)
os.makedirs(TRAIN_TAMP, exist_ok=True)
os.makedirs(TEST_CLEAN, exist_ok=True)
os.makedirs(TEST_TAMP, exist_ok=True)

def copy_split(src_folder, dest_train, dest_test, split_ratio=0.8):
    files = [f for f in os.listdir(src_folder)
             if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tiff",".webp"))]

    random.shuffle(files)
    split = int(len(files) * split_ratio)

    train_files = files[:split]
    test_files = files[split:]

    for f in train_files:
        shutil.copy(os.path.join(src_folder, f), dest_train)

    for f in test_files:
        shutil.copy(os.path.join(src_folder, f), dest_test)

    print(f"{src_folder}: {len(train_files)} train, {len(test_files)} test")

copy_split(SOURCE_AU, TRAIN_CLEAN, TEST_CLEAN)
copy_split(SOURCE_TP, TRAIN_TAMP, TEST_TAMP)

print("✔ CASIA dataset organized into train/test successfully!")
