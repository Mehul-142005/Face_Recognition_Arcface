
from pathlib import Path
import csv
import cv2
import numpy as np

from core_face_recognition.face_recognition import initialize_arcface
from core_face_recognition.recognition_utils import normalize_embedding

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize the actual ArcFace model
print("Initializing ArcFace...")
app = initialize_arcface()

# -------------------------------------------------
# 1. Build identity templates from enrollment images
# -------------------------------------------------

print("\nBuilding enrollment gallery...")

gallery = {}

for identity_dir in sorted((DATA_DIR / "Faces").iterdir()):
    if not identity_dir.is_dir():
        continue

    embeddings = []

    for path in sorted(identity_dir.iterdir()):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        image = cv2.imread(str(path))

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) != 1:
            raise RuntimeError(
                f"Expected exactly one face: {path}"
            )

        embeddings.append(
            normalize_embedding(faces[0].embedding)
        )

    if embeddings:
        average = np.mean(np.stack(embeddings), axis=0)
        gallery[identity_dir.name] = normalize_embedding(average)

        print(
            f"{identity_dir.name}: "
            f"{len(embeddings)} enrollment images"
        )

if len(gallery) < 2:
    raise RuntimeError("At least two enrolled identities are required.")

print("\nGallery created:", list(gallery.keys()))

# -------------------------------------------------
# 2. Extract scores from Validation and Queries
# -------------------------------------------------

def evaluate_split(split_name):

    split_dir = DATA_DIR / split_name
    records = []

    print(f"\nEvaluating {split_name}...")

    for identity_dir in sorted(split_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        true_identity = identity_dir.name

        for path in sorted(identity_dir.iterdir()):
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            image = cv2.imread(str(path))

            if image is None:
                raise RuntimeError(f"Cannot read image: {path}")

            faces = app.get(image)

            if len(faces) != 1:
                raise RuntimeError(
                    f"Expected exactly one face: {path}"
                )

            query = normalize_embedding(faces[0].embedding)

            scores = {
                identity: float(np.dot(query, template))
                for identity, template in gallery.items()
            }

            ranked = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True
            )

            best_identity, best_score = ranked[0]
            second_score = ranked[1][1]
            margin = best_score - second_score

            records.append({
                "split": split_name,
                "true_identity": true_identity,
                "predicted_identity": best_identity,
                "top_score": best_score,
                "second_score": second_score,
                "margin": margin,
                "correct_rank1": (
                    true_identity != "Unknown"
                    and best_identity == true_identity
                )
            })

            print(
                f"{path.name} | True: {true_identity} | "
                f"Top: {best_identity} | "
                f"Score: {best_score:.4f} | "
                f"Margin: {margin:.4f}"
            )

    return records


validation_records = evaluate_split("Validation")
query_records = evaluate_split("Queries")

# -------------------------------------------------
# 3. Save raw similarity scores
# -------------------------------------------------

fields = [
    "split",
    "true_identity",
    "predicted_identity",
    "top_score",
    "second_score",
    "margin",
    "correct_rank1"
]

for split_name, records in [
    ("validation", validation_records),
    ("queries", query_records)
]:

    output_path = RESULTS_DIR / f"{split_name}_scores.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved: {output_path}")

print("\nSimilarity score extraction completed.")
print("Validation samples:", len(validation_records))
print("Query samples:", len(query_records))
