
"""Evaluation utilities for open-set ArcFace recognition."""

import csv
from pathlib import Path


def calculate_metrics(records, threshold):
    """
    Each record contains:
    genuine: True for a registered identity, False for an unknown identity.
    score: Highest cosine similarity.
    predicted_known: Whether the system accepted an identity.
    """

    genuine_count = 0
    impostor_count = 0

    true_accepts = 0
    false_rejects = 0
    false_accepts = 0
    true_rejects = 0

    for record in records:
        genuine = bool(record["genuine"])
        score = float(record["score"])

        accepted = score >= threshold

        if genuine:
            genuine_count += 1
            if accepted:
                true_accepts += 1
            else:
                false_rejects += 1
        else:
            impostor_count += 1
            if accepted:
                false_accepts += 1
            else:
                true_rejects += 1

    tar = (
        true_accepts / genuine_count
        if genuine_count else 0.0
    )

    frr = (
        false_rejects / genuine_count
        if genuine_count else 0.0
    )

    far = (
        false_accepts / impostor_count
        if impostor_count else 0.0
    )

    precision = (
        true_accepts / (true_accepts + false_accepts)
        if true_accepts + false_accepts else 0.0
    )

    recall = tar

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": threshold,
        "genuine_samples": genuine_count,
        "impostor_samples": impostor_count,
        "true_accepts": true_accepts,
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "true_rejects": true_rejects,
        "TAR": tar,
        "FAR": far,
        "FRR": frr,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def evaluate_thresholds(records, thresholds):
    """Evaluate the same score records at multiple thresholds."""

    return [
        calculate_metrics(records, threshold)
        for threshold in thresholds
    ]


def save_results_csv(results, output_path):
    """Save threshold evaluation results to CSV."""

    if not results:
        raise ValueError("No evaluation results to save.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":

    # Synthetic examples for testing the evaluation logic.
    # These are NOT actual ArcFace experimental results.

    sample_records = [
        {"genuine": True, "score": 0.91},
        {"genuine": True, "score": 0.82},
        {"genuine": True, "score": 0.68},
        {"genuine": False, "score": 0.72},
        {"genuine": False, "score": 0.39},
        {"genuine": False, "score": 0.25},
    ]

    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]

    results = evaluate_thresholds(
        sample_records, thresholds
    )

    output = Path("results") / "threshold_evaluation.csv"

    save_results_csv(results, output)

    print("\nThreshold Evaluation (Synthetic Test Data)")
    print("-" * 65)

    for result in results:
        print(
            f"Threshold={result['threshold']:.2f} | "
            f"TAR={result['TAR']:.2f} | "
            f"FAR={result['FAR']:.2f} | "
            f"FRR={result['FRR']:.2f}"
        )

    print(f"\nCSV saved to: {output}")
