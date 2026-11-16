
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core_face_recognition.face_recognition import initialize_arcface

from core_face_recognition.recognition_utils import (
    normalize_embedding,
    build_identity_templates,
    cosine_scores,
    decide_identity,
    laplacian_blur_score
)


# ============================================================
# CONFIGURATION
# ============================================================

GALLERY_DIR = Path("data/Faces")
VALIDATION_DIR = Path("data/Validation")
RESULTS_DIR = Path("results")

TEST_CONDITIONS = {
    "clean": Path("data/Queries"),
    "blur_sigma_1": Path(
        "data/DegradedQueries/blur_sigma_1"
    ),
    "blur_sigma_3": Path(
        "data/DegradedQueries/blur_sigma_3"
    )
}

COSINE_THRESHOLD = 0.3826
MARGIN_THRESHOLD = 0.2245

QUALITY_PERCENTILE = 10

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# BUILD ENROLLMENT GALLERY
# ============================================================

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
                    normalize_embedding(
                        faces[0].embedding
                    )
                )

        if embeddings:
            embeddings_by_identity[
                identity_dir.name
            ] = embeddings

    templates, counts = build_identity_templates(
        embeddings_by_identity
    )

    print("\nGallery enrollment counts:")

    for identity, count in counts.items():
        print(f"{identity}: {count}")

    return templates


# ============================================================
# EXTRACT FACE EMBEDDING AND BLUR SCORE
# ============================================================

def extract_face_data(app, image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    faces = app.get(image)

    # Require exactly one detected face
    if len(faces) != 1:
        return None

    face = faces[0]

    x1, y1, x2, y2 = face.bbox.astype(int)

    height, width = image.shape[:2]

    # Clip bounding box to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    blur_score = laplacian_blur_score(crop)

    embedding = normalize_embedding(
        face.embedding
    )

    return {
        "embedding": embedding,
        "blur_score": blur_score
    }


# ============================================================
# CALIBRATE QUALITY THRESHOLD ON VALIDATION DATA
# ============================================================

def calibrate_quality_threshold(app):

    print("\nCalibrating quality threshold...")
    print("Using known-person Validation images only.")

    known_blur_scores = []
    validation_rows = []

    for identity_dir in sorted(VALIDATION_DIR.iterdir()):

        if not identity_dir.is_dir():
            continue

        # Exclude unknown identities from calibration
        if identity_dir.name.lower() == "unknown":
            continue

        for path in sorted(identity_dir.iterdir()):

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            result = extract_face_data(app, path)

            if result is None:

                validation_rows.append({
                    "identity": identity_dir.name,
                    "image": path.name,
                    "detection_success": False,
                    "blur_score": None
                })

                continue

            score = result["blur_score"]

            known_blur_scores.append(score)

            validation_rows.append({
                "identity": identity_dir.name,
                "image": path.name,
                "detection_success": True,
                "blur_score": score
            })

    if not known_blur_scores:
        raise RuntimeError(
            "No known-person validation face scores found."
        )

    quality_threshold = float(
        np.percentile(
            known_blur_scores,
            QUALITY_PERCENTILE
        )
    )

    validation_df = pd.DataFrame(validation_rows)

    validation_df.to_csv(
        RESULTS_DIR / "quality_gate_validation.csv",
        index=False
    )

    validation_failures = sum(
        not row["detection_success"]
        for row in validation_rows
    )

    print("\nValidation calibration details:")
    print("Known validation images:", len(known_blur_scores))
    print("Detection failures:", validation_failures)
    print("Quality percentile:", QUALITY_PERCENTILE)

    print(
        f"Selected blur threshold: {quality_threshold:.4f}"
    )

    return quality_threshold


# ============================================================
# EVALUATE ONE TEST CONDITION
# ============================================================

def evaluate_condition(
    app,
    templates,
    condition,
    image_root,
    quality_threshold
):

    print("\nEvaluating:", condition)

    rows = []

    for identity_dir in sorted(image_root.iterdir()):

        if not identity_dir.is_dir():
            continue

        actual_identity = identity_dir.name

        is_unknown = (
            actual_identity.lower() == "unknown"
        )

        for path in sorted(identity_dir.iterdir()):

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            result = extract_face_data(app, path)

            row = {
                "condition": condition,
                "actual_identity": actual_identity,
                "image": path.name,
                "is_unknown": is_unknown,
                "detection_success": False,
                "blur_score": None,
                "top_identity": None,
                "top_score": None,
                "margin": None,
                "baseline_prediction": "Unknown",
                "quality_gate_prediction": "Unknown",
                "quality_gate_rejected": False
            }

            # Detection failure is treated as rejection
            if result is None:
                rows.append(row)
                continue

            row["detection_success"] = True

            embedding = result["embedding"]
            blur_score = result["blur_score"]

            row["blur_score"] = blur_score

            scores = cosine_scores(
                embedding,
                templates
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

            # ------------------------------------------------
            # BASELINE: WITHOUT QUALITY GATE
            # ------------------------------------------------

            baseline_result = decide_identity(
                scores,
                threshold=COSINE_THRESHOLD,
                use_margin=False
            )

            row["baseline_prediction"] = (
                baseline_result[0]
            )

            # ------------------------------------------------
            # WITH QUALITY GATE
            # ------------------------------------------------

            if blur_score < quality_threshold:

                row["quality_gate_prediction"] = "Unknown"

                row["quality_gate_rejected"] = True

            else:

                gated_result = decide_identity(
                    scores,
                    threshold=COSINE_THRESHOLD,
                    use_margin=False
                )

                row["quality_gate_prediction"] = (
                    gated_result[0]
                )

            rows.append(row)

    return rows


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    rows,
    condition,
    quality_threshold
):

    df = pd.DataFrame(rows)

    known = df[~df["is_unknown"]]
    unknown = df[df["is_unknown"]]

    known_count = len(known)
    unknown_count = len(unknown)

    total_images = len(df)

    detection_failures = int(
        (~df["detection_success"]).sum()
    )

    def get_metrics(prediction_column):

        known_predictions = known[prediction_column]
        unknown_predictions = unknown[prediction_column]

        known_accepted = (
            known_predictions != "Unknown"
        )

        unknown_accepted = (
            unknown_predictions != "Unknown"
        )

        correct_known = (
            known_predictions == known["actual_identity"]
        )

        rank1_correct = (
            known["top_identity"] == known["actual_identity"]
        )

        correct_acceptances = int(
            (known_accepted & correct_known).sum()
        )

        false_acceptances = int(
            unknown_accepted.sum()
        )

        # True Acceptance Rate
        tar = (
            correct_acceptances / known_count
            if known_count else 0
        )

        # False Rejection Rate
        frr = 1 - tar

        # False Acceptance Rate
        far = (
            false_acceptances / unknown_count
            if unknown_count else 0
        )

        # Unknown Person Rejection Rate
        unknown_rejection = 1 - far

        # Rank-1 accuracy
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

    baseline = get_metrics(
        "baseline_prediction"
    )

    gated = get_metrics(
        "quality_gate_prediction"
    )

    quality_rejections = int(
        df["quality_gate_rejected"].sum()
    )

    known_quality_rejections = int(
        known["quality_gate_rejected"].sum()
    )

    unknown_quality_rejections = int(
        unknown["quality_gate_rejected"].sum()
    )

    summary = {
        "condition": condition,
        "total_images": total_images,
        "known_images": known_count,
        "unknown_images": unknown_count,
        "detection_failures": detection_failures,
        "quality_threshold": quality_threshold,
        "quality_gate_rejections": quality_rejections,
        "known_quality_rejections": known_quality_rejections,
        "unknown_quality_rejections": unknown_quality_rejections
    }

    for key, value in baseline.items():
        summary[f"baseline_{key}"] = value

    for key, value in gated.items():
        summary[f"gated_{key}"] = value

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ARCFACE QUALITY GATE EVALUATION")
    print("=" * 60)

    print("\nInitializing ArcFace...")

    app = initialize_arcface()

    print("\nBuilding enrollment gallery...")

    templates = load_gallery(app)

    print(
        "\nGallery identities:",
        list(templates.keys())
    )

    # Calibrate threshold using Validation only
    quality_threshold = calibrate_quality_threshold(app)

    all_rows = []
    all_summaries = []

    for condition, image_root in TEST_CONDITIONS.items():

        if not image_root.exists():
            print("\nSkipping missing folder:", image_root)
            continue

        rows = evaluate_condition(
            app,
            templates,
            condition,
            image_root,
            quality_threshold
        )

        all_rows.extend(rows)

        # Pass the calibrated threshold explicitly
        summary = calculate_metrics(
            rows,
            condition,
            quality_threshold
        )

        all_summaries.append(summary)

        print("\nCondition summary:", condition)

        for key, value in summary.items():

            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    # Save individual image-level results
    scores_path = (
        RESULTS_DIR / "quality_gate_scores.csv"
    )

    pd.DataFrame(all_rows).to_csv(
        scores_path,
        index=False
    )

    # Save condition-level summary
    metrics_path = (
        RESULTS_DIR / "quality_gate_metrics.csv"
    )

    pd.DataFrame(all_summaries).to_csv(
        metrics_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("QUALITY GATE EVALUATION COMPLETED")
    print("=" * 60)

    print("\nSaved files:")
    print("results/quality_gate_validation.csv")
    print("results/quality_gate_scores.csv")
    print("results/quality_gate_metrics.csv")


if __name__ == "__main__":
    main()
