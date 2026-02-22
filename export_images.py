from datasets import load_from_disk
import os

# ==============================
# PATHS
# ==============================
DATASET_PATH = r"D:\CV PROJ\dataset\train"
OUTPUT_FOLDER = r"D:\CV PROJ\exported_images"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# LOAD DATASET
# ==============================
dataset = load_from_disk(DATASET_PATH)

print("Total samples:", len(dataset))

# ==============================
# EXPORT IMAGES
# ==============================
for i in range(len(dataset)):
    img = dataset[i]["image"]   # This is already a PIL Image

    save_path = os.path.join(OUTPUT_FOLDER, f"img_{i}.jpg")
    img.save(save_path)

    if i % 500 == 0:
        print(f"Saved {i} images")

print("✅ All images exported successfully!")