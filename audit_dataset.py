
from pathlib import Path
import cv2
import hashlib

ROOT = Path("data")
SPLITS = ["Faces", "Validation", "Queries"]

hash_locations = {}
total = 0
unreadable = []

for split in SPLITS:
    print(f"\n===== {split} =====")

    for path in sorted((ROOT / split).rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        total += 1
        img = cv2.imread(str(path))

        if img is None:
            unreadable.append(str(path))
            print("UNREADABLE:", path)
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_locations.setdefault(digest, []).append(str(path))

        print(f"OK: {path} | Size: {img.shape[1]}x{img.shape[0]}")

print("\n===== DUPLICATE IMAGE CHECK =====")

duplicates = [
    locations for locations in hash_locations.values()
    if len(locations) > 1
]

if duplicates:
    for group in duplicates:
        print("\nDUPLICATE:")
        for location in group:
            print(" ", location)
else:
    print("No exact duplicate files found.")

print("\nUnreadable images:", len(unreadable))
print("Total image files:", total)
