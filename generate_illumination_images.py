
from pathlib import Path
import cv2
import numpy as np

SOURCE = Path("data/Queries")
DESTINATION = Path("data/IlluminationQueries")

CONDITIONS = {
    "low_illumination": 0.60,
    "high_illumination": 1.40
}

for identity_dir in sorted(SOURCE.iterdir()):

    if not identity_dir.is_dir():
        continue

    for path in sorted(identity_dir.iterdir()):

        if path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]:
            continue

        image = cv2.imread(str(path))

        if image is None:
            print("Unreadable:", path)
            continue

        for condition, factor in CONDITIONS.items():

            # Adjust brightness while preserving image dimensions
            adjusted = np.clip(
                image.astype(np.float32) * factor,
                0,
                255
            ).astype(np.uint8)

            output_dir = (
                DESTINATION / condition / identity_dir.name
            )

            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{path.stem}.jpg"

            cv2.imwrite(str(output_path), adjusted)

print("\nIllumination image generation completed.")
print("Original Queries folder was not modified.")
