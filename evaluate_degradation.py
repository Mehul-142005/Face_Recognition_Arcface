
from pathlib import Path
import csv
import cv2
import numpy as np

from core_face_recognition.face_recognition import initialize_arcface
from core_face_recognition.recognition_utils import normalize_embedding


# =====================================================
# CONFIGURATION
# =====================================================

DATA = Path("data")
RESULTS = Path("results")

RESULTS.mkdir(exist_ok=True)

COSINE_THRESHOLD = 0.3826
MARGIN_THRESHOLD = 0.2245

CONDITIONS = [
    "blur_sigma_1",
    "blur_sigma_3",
    "downsample_56",
    "downsample_28"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# =====================================================
# INITIALIZE ARCFACE
# =====================================================

print("\nInitializing ArcFace...")

app = initialize_arcface()


# =====================================================
# BUILD ORIGINAL ENROLLMENT GALLERY
# =====================================================

print("\nBuilding enrollment gallery...")

gallery = {}

for identity_dir in sorted((DATA / "Faces").iterdir()):

    if not identity_dir.is_dir():
        continue

    embeddings = []

    for path in sorted(identity_dir.iterdir()):

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(f"Cannot read enrollment image: {path}")

        faces = app.get(image)

        if len(faces) != 1:
            raise RuntimeError(
                f"Enrollment face detection failed: {path}"
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

print("\nGallery identities:", list(gallery.keys()))


# =====================================================
# EVALUATE ONE DEGRADATION CONDITION
# =====================================================

def evaluate_condition(condition):

    folder = DATA / "DegradedQueries" / condition

    if not folder.exists():
        raise FileNotFoundError(
            f"Degradation folder not found: {folder}"
        )

    records = []

    print("\n====================================")
    print("Evaluating:", condition)
    print("====================================")

    for identity_dir in sorted(folder.iterdir()):

        if not identity_dir.is_dir():
            continue

        true_identity = identity_dir.name

        for path in sorted(identity_dir.iterdir()):

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(path))

            if image is None:
                raise RuntimeError(f"Cannot read image: {path}")

            faces = app.get(image)

            # Record face-detection failures instead of
            # stopping the entire experiment.

            if len(faces) != 1:

                print("DETECTION FAILURE:", path)

                records.append({
                    "condition": condition,
                    "filename": path.name,
                    "true_identity": true_identity,
                    "predicted_identity": "Detection_Failure",
                    "top_score": "",
                    "second_score": "",
                    "margin": "",
                    "rank1_correct": False,
                    "detection_success": False
                })

                continue

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

            predicted_identity, top_score = ranked[0]

            second_score = ranked[1][1]

            margin = top_score - second_score

            rank1_correct = (
                true_identity != "Unknown"
                and predicted_identity == true_identity
            )

            records.append({
                "condition": condition,
                "filename": path.name,
                "true_identity": true_identity,
                "predicted_identity": predicted_identity,
                "top_score": top_score,
                "second_score": second_score,
                "margin": margin,
                "rank1_correct": rank1_correct,
                "detection_success": True
            })

            print(
                f"{path.name} | "
                f"True: {true_identity} | "
                f"Predicted: {predicted_identity} | "
                f"Score: {top_score:.4f} | "
                f"Margin: {margin:.4f}"
            )

    return records


# =====================================================
# CALCULATE METRICS
# =====================================================

def calculate_metrics(records, condition):

    known_all = [
        r for r in records
        if r["true_identity"] != "Unknown"
    ]

    unknown_all = [
        r for r in records
        if r["true_identity"] == "Unknown"
    ]

    known = [
        r for r in known_all
        if r["detection_success"]
    ]

    unknown = [
        r for r in unknown_all
        if r["detection_success"]
    ]

    detection_failures = [
        r for r in records
        if not r["detection_success"]
    ]

    total = len(records)

    known_count = len(known_all)
    unknown_count = len(unknown_all)

    failed_known = sum(
        not r["detection_success"] for r in known_all
    )

    failed_unknown = sum(
        not r["detection_success"] for r in unknown_all
    )

    # -------------------------------------------------
    # THRESHOLD-ONLY POLICY
    # -------------------------------------------------

    genuine_accepted = sum(
        float(r["top_score"]) >= COSINE_THRESHOLD
        for r in known
    )

    unknown_accepted = sum(
        float(r["top_score"]) >= COSINE_THRESHOLD
        for r in unknown
    )

    # Detection failures count as rejected samples.
    tar = genuine_accepted / known_count if known_count else 0

    frr = 1 - tar

    # Detection failures on unknown images count as rejected,
    # not as false acceptances.
    far = unknown_accepted / unknown_count if unknown_count else 0

    unknown_rejection = 1 - far

    rank1_correct = sum(
        r["rank1_correct"] for r in known
    )

    rank1_accuracy = (
        rank1_correct / known_count if known_count else 0
    )

    # -------------------------------------------------
    # THRESHOLD + MARGIN POLICY
    # -------------------------------------------------

    margin_genuine_accepted = sum(
        float(r["top_score"]) >= COSINE_THRESHOLD
        and float(r["margin"]) >= MARGIN_THRESHOLD
        for r in known
    )

    margin_unknown_accepted = sum(
        float(r["top_score"]) >= COSINE_THRESHOLD
        and float(r["margin"]) >= MARGIN_THRESHOLD
        for r in unknown
    )

    margin_tar = (
        margin_genuine_accepted / known_count
        if known_count else 0
    )

    margin_frr = 1 - margin_tar

    margin_far = (
        margin_unknown_accepted / unknown_count
        if unknown_count else 0
    )

    margin_unknown_rejection = 1 - margin_far

    # -------------------------------------------------
    # FINAL METRICS
    # -------------------------------------------------

    result = {
        "condition": condition,
        "total_samples": total,
        "known_samples": known_count,
        "unknown_samples": unknown_count,

        "detection_failures": len(detection_failures),
        "detection_failure_rate": (
            len(detection_failures) / total if total else 0
        ),

        "known_detection_failures": failed_known,
        "unknown_detection_failures": failed_unknown,

        "TAR_threshold": tar,
        "FRR_threshold": frr,
        "FAR_threshold": far,
        "unknown_rejection_threshold": unknown_rejection,

        "rank1_accuracy": rank1_accuracy,

        "TAR_margin": margin_tar,
        "FRR_margin": margin_frr,
        "FAR_margin": margin_far,
        "unknown_rejection_margin": margin_unknown_rejection
    }

    print("\n----- CONDITION SUMMARY -----")

    for key, value in result.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return result


# =====================================================
# RUN ALL EXPERIMENTS
# =====================================================

all_records = []
all_results = []

for condition in CONDITIONS:

    records = evaluate_condition(condition)

    result = calculate_metrics(records, condition)

    all_records.extend(records)

    all_results.append(result)


# =====================================================
# SAVE INDIVIDUAL IMAGE RESULTS
# =====================================================

if all_records:

    with open(
        RESULTS / "degradation_scores.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_records[0].keys()
        )

        writer.writeheader()

        writer.writerows(all_records)


# =====================================================
# SAVE CONDITION-WISE METRICS
# =====================================================

if all_results:

    with open(
        RESULTS / "degradation_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()

        writer.writerows(all_results)


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n====================================")
print("DEGRADATION EVALUATION COMPLETED")
print("====================================")

print("\nResults saved:")

print("results/degradation_scores.csv")
print("results/degradation_metrics.csv")

print("\nAll degradation conditions processed.")
