
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from core_face_recognition.face_recognition import initialize_arcface
from core_face_recognition.recognition_utils import (
    cosine_scores,
    decide_identity,
    normalize_embedding,
    build_identity_templates
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GALLERY_DIR = Path("data/Faces")
ILLUMINATION_DIR = Path("data/IlluminationQueries")
RESULTS_DIR = Path("results")

COSINE_THRESHOLD = 0.3826
MARGIN_THRESHOLD = 0.2245

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# BUILD ENROLLMENT GALLERY
# --------------------------------------------------

def load_gallery(app):

    embeddings_by_identity = {}

    for identity_dir in sorted(GALLERY_DIR.iterdir()):

        if not identity_dir.is_dir():
            continue

        embeddings = []

        for path in sorted(identity_dir.iterdir()):

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(path))

            if image is None:
                print("Unreadable gallery image:", path)
                continue

            faces = app.get(image)

            if len(faces) == 1:
                embeddings.append(
                    normalize_embedding(faces[0].embedding)
                )

        if embeddings:
            embeddings_by_identity[identity_dir.name] = embeddings

    templates, counts = build_identity_templates(
        embeddings_by_identity
    )

    print("\nEnrollment image counts:")

    for identity, count in counts.items():
        print(f"{identity}: {count}")

    return templates


# --------------------------------------------------
# EVALUATE ONE ILLUMINATION CONDITION
# --------------------------------------------------

def evaluate_condition(app, templates, condition):

    condition_dir = ILLUMINATION_DIR / condition

    rows = []

    for identity_dir in sorted(condition_dir.iterdir()):

        if not identity_dir.is_dir():
            continue

        actual_identity = identity_dir.name

        is_unknown = (
            actual_identity.lower() == "unknown"
        )

        for path in sorted(identity_dir.iterdir()):

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(path))

            row = {
                "condition": condition,
                "actual_identity": actual_identity,
                "image": path.name,
                "is_unknown": is_unknown,
                "detection_success": False,
                "top_identity": None,
                "top_score": None,
                "margin": None,
                "threshold_prediction": "Unknown",
                "margin_prediction": "Unknown"
            }

            if image is None:
                print("Unreadable query image:", path)
                rows.append(row)
                continue

            faces = app.get(image)

            # Reject images with zero or multiple detected faces
            if len(faces) != 1:
                rows.append(row)
                continue

            row["detection_success"] = True

            embedding = normalize_embedding(
                faces[0].embedding
            )

            scores = cosine_scores(
                embedding,
                templates
            )

            # Cosine-threshold decision
            threshold_result = decide_identity(
                scores,
                threshold=COSINE_THRESHOLD,
                use_margin=False
            )

            # Cosine threshold + top-two margin decision
            margin_result = decide_identity(
                scores,
                threshold=COSINE_THRESHOLD,
                use_margin=True,
                margin_threshold=MARGIN_THRESHOLD
            )

            ranked = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True
            )

            row["top_identity"] = ranked[0][0]
            row["top_score"] = ranked[0][1]

            if len(ranked) > 1:
                row["margin"] = (
                    ranked[0][1] - ranked[1][1]
                )

            # decide_identity returns a tuple:
            # (predicted_identity, best_score, margin)

            row["threshold_prediction"] = threshold_result[0]
            row["margin_prediction"] = margin_result[0]

            rows.append(row)

    return rows


# --------------------------------------------------
# CALCULATE PERFORMANCE METRICS
# --------------------------------------------------

def calculate_metrics(rows, condition):

    df = pd.DataFrame(rows)

    known = df[~df["is_unknown"]]
    unknown = df[df["is_unknown"]]

    known_count = len(known)
    unknown_count = len(unknown)

    detection_failures = int(
        (~df["detection_success"]).sum()
    )

    def policy_metrics(prediction_column):

        # A prediction is accepted only if it is not Unknown
        known_accepted = (
            known[prediction_column] != "Unknown"
        )

        unknown_accepted = (
            unknown[prediction_column] != "Unknown"
        )

        # Correctly identified known people
        correct_known = (
            known[prediction_column]
            == known["actual_identity"]
        )

        # Correct top-ranked identity among known people
        rank1_correct = (
            known["top_identity"]
            == known["actual_identity"]
        )

        # True Acceptance Rate
        tar = (
            int((known_accepted & correct_known).sum())
            / known_count
            if known_count else 0
        )

        # False Rejection Rate
        frr = 1 - tar

        # False Acceptance Rate
        far = (
            int(unknown_accepted.sum())
            / unknown_count
            if unknown_count else 0
        )

        # Unknown Person Rejection Rate
        unknown_rejection = 1 - far

        # Rank-1 accuracy; detection failures count as incorrect
        rank1 = (
            int(
                (
                    rank1_correct
                    & known["detection_success"]
                ).sum()
            ) / known_count
            if known_count else 0
        )

        return {
            "TAR": tar,
            "FRR": frr,
            "FAR": far,
            "unknown_rejection": unknown_rejection,
            "rank1_accuracy": rank1
        }

    threshold_metrics = policy_metrics(
        "threshold_prediction"
    )

    margin_metrics = policy_metrics(
        "margin_prediction"
    )

    total_images = len(df)

    summary = {
        "condition": condition,
        "total_images": total_images,
        "known_images": known_count,
        "unknown_images": unknown_count,
        "detection_failures": detection_failures,
        "detection_failure_rate": (
            detection_failures / total_images
            if total_images else 0
        )
    }

    for key, value in threshold_metrics.items():
        summary[f"threshold_{key}"] = value

    for key, value in margin_metrics.items():
        summary[f"margin_{key}"] = value

    return summary


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Initializing ArcFace...")

    app = initialize_arcface()

    print("\nBuilding enrollment gallery...")

    templates = load_gallery(app)

    print("\nGallery identities:", list(templates.keys()))

    all_rows = []
    summaries = []

    conditions = [
        "low_illumination",
        "high_illumination"
    ]

    for condition in conditions:

        print("\n" + "=" * 55)
        print("Evaluating:", condition)
        print("=" * 55)

        rows = evaluate_condition(
            app,
            templates,
            condition
        )

        all_rows.extend(rows)

        summary = calculate_metrics(
            rows,
            condition
        )

        summaries.append(summary)

        print("\nResults:")

        for key, value in summary.items():

            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    # Save individual image-level results
    scores_path = (
        RESULTS_DIR / "illumination_scores.csv"
    )

    pd.DataFrame(all_rows).to_csv(
        scores_path,
        index=False
    )

    # Save condition-level summary
    metrics_path = (
        RESULTS_DIR / "illumination_metrics.csv"
    )

    pd.DataFrame(summaries).to_csv(
        metrics_path,
        index=False
    )

    print("\n" + "=" * 55)
    print("ILLUMINATION EVALUATION COMPLETED")
    print("=" * 55)

    print("\nFiles saved:")
    print(scores_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
