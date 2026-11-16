
from pathlib import Path
from itertools import product

import cv2
import numpy as np
import pandas as pd

from core_face_recognition.face_recognition import initialize_arcface

from core_face_recognition.recognition_utils import (
    normalize_embedding,
    build_identity_templates,
    cosine_scores
)


# ============================================================
# CONFIGURATION
# ============================================================

GALLERY_DIR = Path("data/Faces")
VALIDATION_DIR = Path("data/Validation")
QUERY_DIR = Path("data/Queries")
RESULTS_DIR = Path("results")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}

RESULTS_DIR.mkdir(exist_ok=True)

# All three folders must remain unchanged during evaluation.


# ============================================================
# READ IMAGES AND EXTRACT ARCFACE EMBEDDINGS
# ============================================================

def extract_dataset_embeddings(app, root_dir, dataset_name):

    print(f"\nExtracting embeddings from {dataset_name}...")

    records = []

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root_dir}"
        )

    for identity_dir in sorted(root_dir.iterdir()):

        if not identity_dir.is_dir():
            continue

        identity = identity_dir.name

        for path in sorted(identity_dir.iterdir()):

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(path))

            if image is None:
                print("Unreadable image:", path)
                records.append({
                    "identity": identity,
                    "image": path.name,
                    "embedding": None,
                    "detection_success": False
                })
                continue

            faces = app.get(image)

            # Exactly one face is required
            if len(faces) != 1:

                print(
                    f"Detection failure: {identity}/{path.name}"
                )

                records.append({
                    "identity": identity,
                    "image": path.name,
                    "embedding": None,
                    "detection_success": False
                })

                continue

            embedding = normalize_embedding(
                faces[0].embedding
            )

            records.append({
                "identity": identity,
                "image": path.name,
                "embedding": embedding,
                "detection_success": True
            })

    successful = sum(
        record["detection_success"]
        for record in records
    )

    print(
        f"{dataset_name}: {successful}/{len(records)} "
        "images processed successfully"
    )

    return records


# ============================================================
# BUILD MULTI-IMAGE GALLERY
# ============================================================

def build_multi_image_gallery(gallery_records):

    embeddings_by_identity = {}

    for record in gallery_records:

        if not record["detection_success"]:
            continue

        identity = record["identity"]

        embeddings_by_identity.setdefault(
            identity, []
        ).append(record["embedding"])

    templates, counts = build_identity_templates(
        embeddings_by_identity
    )

    return templates, counts


# ============================================================
# BUILD SINGLE-IMAGE GALLERY
# ============================================================

def build_single_image_gallery(selected_records):

    embeddings_by_identity = {}

    for record in selected_records:

        identity = record["identity"]

        embeddings_by_identity[identity] = [
            record["embedding"]
        ]

    templates, _ = build_identity_templates(
        embeddings_by_identity
    )

    return templates


# ============================================================
# GET TOP MATCH
# ============================================================

def get_top_match(embedding, templates):

    scores = cosine_scores(
        embedding,
        templates
    )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_identity, best_score = ranked[0]

    return best_identity, float(best_score)


# ============================================================
# CALIBRATE COSINE THRESHOLD USING VALIDATION DATA
# ============================================================

def calibrate_threshold(templates, validation_records):

    known_results = []
    unknown_results = []

    for record in validation_records:

        is_unknown = (
            record["identity"].lower() == "unknown"
        )

        if not record["detection_success"]:

            if is_unknown:
                unknown_results.append(None)
            else:
                known_results.append(None)

            continue

        predicted_identity, best_score = get_top_match(
            record["embedding"],
            templates
        )

        if is_unknown:

            unknown_results.append(best_score)

        else:

            correct_identity = (
                predicted_identity == record["identity"]
            )

            known_results.append(
                (best_score, correct_identity)
            )

    if not known_results or not unknown_results:
        raise RuntimeError(
            "Validation must contain known and unknown images."
        )

    valid_scores = []

    for result in known_results:

        if result is not None:
            valid_scores.append(result[0])

    for result in unknown_results:

        if result is not None:
            valid_scores.append(result)

    if not valid_scores:
        raise RuntimeError(
            "No valid validation scores available."
        )

    # Candidate thresholds include boundaries around each score.
    candidates = {-1.000001, 1.000001}

    for score in valid_scores:

        candidates.add(float(score))

        candidates.add(
            float(np.nextafter(score, np.inf))
        )

    best_threshold = None
    best_balanced_accuracy = -1.0

    for threshold in sorted(candidates):

        known_correct = 0
        known_total = len(known_results)

        for result in known_results:

            if result is None:
                continue

            score, correct_identity = result

            if score >= threshold and correct_identity:
                known_correct += 1

        unknown_rejected = 0
        unknown_total = len(unknown_results)

        for score in unknown_results:

            if score is None:
                # Detection failure counts as rejection.
                unknown_rejected += 1

            elif score < threshold:
                unknown_rejected += 1

        tar = (
            known_correct / known_total
            if known_total else 0
        )

        unknown_rejection = (
            unknown_rejected / unknown_total
            if unknown_total else 0
        )

        balanced_accuracy = (
            tar + unknown_rejection
        ) / 2

        # On ties, select the higher threshold.
        if (
            balanced_accuracy > best_balanced_accuracy
            or (
                np.isclose(
                    balanced_accuracy,
                    best_balanced_accuracy
                )
                and (
                    best_threshold is None
                    or threshold > best_threshold
                )
            )
        ):

            best_balanced_accuracy = balanced_accuracy
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_balanced_accuracy)
    )


# ============================================================
# EVALUATE ONE GALLERY ON QUERY DATA
# ============================================================

def evaluate_gallery(
    templates,
    query_records,
    strategy_name,
    threshold
):

    known_total = 0
    unknown_total = 0

    correct_known = 0
    false_acceptances = 0
    rank1_correct = 0
    detection_failures = 0

    detailed_rows = []

    for record in query_records:

        actual_identity = record["identity"]

        is_unknown = (
            actual_identity.lower() == "unknown"
        )

        if is_unknown:
            unknown_total += 1
        else:
            known_total += 1

        row = {
            "strategy": strategy_name,
            "actual_identity": actual_identity,
            "image": record["image"],
            "threshold": threshold,
            "detection_success": record["detection_success"],
            "predicted_identity": "Unknown",
            "top_score": None,
            "correct": False
        }

        if not record["detection_success"]:

            detection_failures += 1
            detailed_rows.append(row)
            continue

        predicted_identity, best_score = get_top_match(
            record["embedding"],
            templates
        )

        row["predicted_identity"] = (
            predicted_identity
            if best_score >= threshold
            else "Unknown"
        )

        row["top_score"] = best_score

        if not is_unknown:

            if predicted_identity == actual_identity:
                rank1_correct += 1

            if (
                best_score >= threshold
                and predicted_identity == actual_identity
            ):

                correct_known += 1
                row["correct"] = True

        else:

            if best_score >= threshold:
                false_acceptances += 1

        detailed_rows.append(row)

    tar = (
        correct_known / known_total
        if known_total else 0
    )

    frr = 1 - tar

    far = (
        false_acceptances / unknown_total
        if unknown_total else 0
    )

    unknown_rejection = 1 - far

    rank1_accuracy = (
        rank1_correct / known_total
        if known_total else 0
    )

    total = known_total + unknown_total

    metrics = {
        "strategy": strategy_name,
        "threshold": threshold,
        "known_images": known_total,
        "unknown_images": unknown_total,
        "total_images": total,
        "detection_failures": detection_failures,
        "correct_known_acceptances": correct_known,
        "false_acceptances": false_acceptances,
        "TAR": tar,
        "FRR": frr,
        "FAR": far,
        "unknown_rejection": unknown_rejection,
        "rank1_accuracy": rank1_accuracy
    }

    return metrics, detailed_rows


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("SINGLE-IMAGE VS MULTI-IMAGE ENROLLMENT")
    print("=" * 65)

    print("\nInitializing ArcFace...")

    app = initialize_arcface()

    # --------------------------------------------------------
    # STEP 1: EXTRACT EMBEDDINGS ONLY ONCE
    # --------------------------------------------------------

    gallery_records = extract_dataset_embeddings(
        app,
        GALLERY_DIR,
        "Enrollment Gallery"
    )

    validation_records = extract_dataset_embeddings(
        app,
        VALIDATION_DIR,
        "Validation Dataset"
    )

    query_records = extract_dataset_embeddings(
        app,
        QUERY_DIR,
        "Query Dataset"
    )

    # --------------------------------------------------------
    # STEP 2: CHECK ENROLLMENT IDENTITIES
    # --------------------------------------------------------

    gallery_by_identity = {}

    for record in gallery_records:

        if not record["detection_success"]:
            continue

        gallery_by_identity.setdefault(
            record["identity"], []
        ).append(record)

    identities = sorted(gallery_by_identity.keys())

    print("\nEnrollment counts:")

    for identity in identities:

        print(
            f"{identity}: "
            f"{len(gallery_by_identity[identity])}"
        )

    if len(identities) != 3:
        raise RuntimeError(
            "Expected three enrolled identities. "
            "Check data/Faces."
        )

    if any(
        len(gallery_by_identity[identity]) == 0
        for identity in identities
    ):
        raise RuntimeError(
            "An identity has no valid enrollment images."
        )

    # --------------------------------------------------------
    # STEP 3: MULTI-IMAGE STRATEGY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MULTI-IMAGE ENROLLMENT")
    print("=" * 65)

    multi_templates, enrollment_counts = (
        build_multi_image_gallery(gallery_records)
    )

    multi_threshold, multi_validation_accuracy = (
        calibrate_threshold(
            multi_templates,
            validation_records
        )
    )

    print(
        f"Multi-image validation threshold: "
        f"{multi_threshold:.6f}"
    )

    print(
        f"Multi-image validation balanced accuracy: "
        f"{multi_validation_accuracy:.4f}"
    )

    multi_metrics, multi_details = evaluate_gallery(
        multi_templates,
        query_records,
        "Multi-image",
        multi_threshold
    )

    multi_metrics["validation_balanced_accuracy"] = (
        multi_validation_accuracy
    )

    # --------------------------------------------------------
    # STEP 4: ALL SINGLE-IMAGE COMBINATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SINGLE-IMAGE ENROLLMENT")
    print("=" * 65)

    identity_options = [
        gallery_by_identity[identity]
        for identity in identities
    ]

    combinations = list(
        product(*identity_options)
    )

    print(
        "\nNumber of possible single-image galleries:",
        len(combinations)
    )

    single_metrics = []
    single_details = []

    for index, selected_records in enumerate(
        combinations, start=1
    ):

        strategy_name = f"Single-image-{index:02d}"

        templates = build_single_image_gallery(
            selected_records
        )

        threshold, validation_accuracy = (
            calibrate_threshold(
                templates,
                validation_records
            )
        )

        metrics, details = evaluate_gallery(
            templates,
            query_records,
            strategy_name,
            threshold
        )

        metrics["validation_balanced_accuracy"] = (
            validation_accuracy
        )

        selected_images = "; ".join(
            f"{record['identity']}={record['image']}"
            for record in selected_records
        )

        metrics["selected_enrollment_images"] = (
            selected_images
        )

        single_metrics.append(metrics)
        single_details.extend(details)

        print(
            f"{index:02d}/{len(combinations)} "
            f"| Threshold: {threshold:.4f} "
            f"| TAR: {metrics['TAR']:.4f} "
            f"| FAR: {metrics['FAR']:.4f} "
            f"| Rank-1: {metrics['rank1_accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # STEP 5: SAVE DETAILED RESULTS
    # --------------------------------------------------------

    all_metrics = single_metrics + [multi_metrics]

    metrics_df = pd.DataFrame(all_metrics)

    details_df = pd.DataFrame(
        single_details + multi_details
    )

    metrics_df.to_csv(
        RESULTS_DIR / "enrollment_strategy_metrics.csv",
        index=False
    )

    details_df.to_csv(
        RESULTS_DIR / "enrollment_strategy_scores.csv",
        index=False
    )

    # --------------------------------------------------------
    # STEP 6: SUMMARY STATISTICS
    # --------------------------------------------------------

    metric_columns = [
        "TAR",
        "FRR",
        "FAR",
        "unknown_rejection",
        "rank1_accuracy",
        "validation_balanced_accuracy"
    ]

    summary_rows = []

    for metric in metric_columns:

        values = metrics_df[
            metrics_df["strategy"].str.startswith(
                "Single-image-"
            )
        ][metric].astype(float)

        summary_rows.append({
            "metric": metric,
            "single_image_mean": values.mean(),
            "single_image_std": values.std(ddof=0),
            "single_image_min": values.min(),
            "single_image_max": values.max(),
            "multi_image": float(
                multi_metrics[metric]
            )
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        RESULTS_DIR / "enrollment_strategy_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # STEP 7: DISPLAY FINAL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL ENROLLMENT COMPARISON")
    print("=" * 65)

    print("\nMulti-image results:")

    for key, value in multi_metrics.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nSingle-image average results:")

    for metric in metric_columns:

        values = metrics_df[
            metrics_df["strategy"].str.startswith(
                "Single-image-"
            )
        ][metric].astype(float)

        print(
            f"{metric}: "
            f"Mean={values.mean():.4f}, "
            f"Std={values.std(ddof=0):.4f}, "
            f"Min={values.min():.4f}, "
            f"Max={values.max():.4f}"
        )

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED")
    print("=" * 65)

    print("\nFiles saved:")

    print(
        "results/enrollment_strategy_metrics.csv"
    )

    print(
        "results/enrollment_strategy_scores.csv"
    )

    print(
        "results/enrollment_strategy_summary.csv"
    )


if __name__ == "__main__":
    main()
