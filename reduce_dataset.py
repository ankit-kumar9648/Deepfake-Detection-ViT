import os
import random

random.seed(42)

LIMITS = {
    "dataset/train/fake": 2000,
    "dataset/train/real": 2000,
    "dataset/validation/fake": 500,
    "dataset/validation/real": 500,
    "dataset/test/fake": 500,
    "dataset/test/real": 500,
}

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for folder, keep in LIMITS.items():
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]

    print(f"{folder}: {len(files)} images")

    if len(files) <= keep:
        print("Already within limit.\n")
        continue

    random.shuffle(files)

    remove_files = files[keep:]

    for f in remove_files:
        os.remove(os.path.join(folder, f))

    print(f"Kept {keep}, Deleted {len(remove_files)}\n")

print("Dataset reduced successfully!")