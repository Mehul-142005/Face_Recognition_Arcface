
from pathlib import Path
import cv2

SOURCE = Path("data/Queries")
DESTINATION = Path("data/DegradedQueries")

def degrade_image(image, condition):

    if condition == "blur_sigma_1":
        return cv2.GaussianBlur(image, (5, 5), 1)

    if condition == "blur_sigma_3":
        return cv2.GaussianBlur(image, (9, 9), 3)

    if condition == "downsample_56":
        small = cv2.resize(image, (56, 56))
        return cv2.resize(
            small, (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    if condition == "downsample_28":
        small = cv2.resize(image, (28, 28))
        return cv2.resize(
            small, (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    return image


conditions = [
    "blur_sigma_1",
    "blur_sigma_3",
    "downsample_56",
    "downsample_28"
]

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
            print("Skipped unreadable:", path)
            continue

        for condition in conditions:

            degraded = degrade_image(image, condition)

            output_dir = DESTINATION / condition / identity_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{path.stem}.jpg"

            success = cv2.imwrite(str(output_path), degraded)

            if not success:
                print("Failed:", output_path)

print("\nDegraded image generation completed.")
print("Original Queries folder was not modified.")
