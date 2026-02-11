import os
import cv2
import numpy as np

INPUT_DIR = "archive"
OUTPUT_DIR = "archive_resized"
IMG_SIZE = (64, 64)

VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n Starting dataset resizing...\n")

for class_name in os.listdir(INPUT_DIR):

    class_path = os.path.join(INPUT_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f" Processing folder: {class_name}")

    output_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)
    # LOOP THROUGH EVERY IMAGE FILE 
    for root, _, files in os.walk(class_path):

        for file_name in files:

            ext = os.path.splitext(file_name)[1].lower()

            if ext not in VALID_EXTS:
                continue

            img_path = os.path.join(root, file_name)

            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping unreadable image: {file_name}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            resized_img = cv2.resize(img, IMG_SIZE)

            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)

            save_path = os.path.join(output_class_path, file_name)
            cv2.imwrite(save_path, resized_img)

    print(f" Finished folder: {class_name}\n")

print(" ALL IMAGES RESIZED SUCCESSFULLY!")
print(" Output saved inside:", OUTPUT_DIR)

