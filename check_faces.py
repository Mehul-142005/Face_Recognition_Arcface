
from pathlib import Path
from core_face_recognition.face_recognition import initialize_arcface
import cv2

app = initialize_arcface()

for split in ["Faces", "Validation", "Queries"]:
    print(f"\n===== {split} =====")

    for path in sorted((Path("data") / split).rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        image = cv2.imread(str(path))

        if image is None:
            print("UNREADABLE:", path)
            continue

        faces = app.get(image)

        if len(faces) == 1:
            print("PASS:", path)
        elif len(faces) == 0:
            print("NO FACE:", path)
        else:
            print(f"MULTIPLE FACES ({len(faces)}):", path)

print("\nFace detection check completed.")
