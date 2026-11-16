
import csv
from pathlib import Path

RESULTS = Path("results")

# Load previously generated ArcFace scores
def load_records(filename):
    with open(RESULTS / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

validation = load_records("validation_scores.csv")
queries = load_records("queries_scores.csv")

# Keep the cosine threshold selected in the previous experiment
BASE_THRESHOLD = 0.3826


def calculate_margin_metrics(records, threshold, margin_threshold):
    known = [r for r in records if r["true_identity"] != "Unknown"]
    unknown = [r for r in records if r["true_identity"] == "Unknown"]

    genuine_accepted = sum(
        float(r["top_score"]) >= threshold
        and float(r["margin"]) >= margin_threshold
        for r in known
    )

    unknown_accepted = sum(
        float(r["top_score"]) >= threshold
        and float(r["margin"]) >= margin_threshold
        for r in unknown
    )

    tar = genuine_accepted / len(known) if known else 0
    frr = 1 - tar
    far = unknown_accepted / len(unknown) if unknown else 0
    unknown_rejection = 1 - far

    # Rank-1 correctness among accepted known samples
    accepted_known = [
        r for r in known
        if float(r["top_score"]) >= threshold
        and float(r["margin"]) >= margin_threshold
    ]

    correct = sum(
        r["predicted_identity"] == r["true_identity"]
        for r in accepted_known
    )

    rank1 = correct / len(known) if known else 0

    return {
        "margin_threshold": margin_threshold,
        "known_count": len(known),
        "unknown_count": len(unknown),
        "TAR": tar,
        "FRR": frr,
        "FAR": far,
        "unknown_rejection": unknown_rejection,
        "rank1_accuracy": rank1,
        "balanced_accuracy": (tar + unknown_rejection) / 2
    }


# Candidate margins derived from validation data only
validation_margins = sorted(
    set(float(r["margin"]) for r in validation)
)

candidates = [0.0]

for a, b in zip(validation_margins, validation_margins[1:]):
    candidates.append((a + b) / 2)

candidates.append(validation_margins[-1] + 0.000001)

# Calibrate margin using validation data
validation_results = [
    calculate_margin_metrics(
        validation, BASE_THRESHOLD, margin
    )
    for margin in candidates
]

best = max(
    validation_results,
    key=lambda r: (r["balanced_accuracy"], r["margin_threshold"])
)

selected_margin = best["margin_threshold"]

print("\n===== VALIDATION MARGIN CALIBRATION =====")
print("Cosine threshold:", BASE_THRESHOLD)
print("Selected margin threshold:", round(selected_margin, 6))

print("\n===== VALIDATION METRICS =====")
for key, value in best.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float)
          else f"{key}: {value}")

# Evaluate on test data using fixed thresholds
test_result = calculate_margin_metrics(
    queries, BASE_THRESHOLD, selected_margin
)

print("\n===== MARGIN-BASED TEST METRICS =====")
for key, value in test_result.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float)
          else f"{key}: {value}")

# Save the experiment results
with open(RESULTS / "margin_test_metrics.csv", "w", newline="",
          encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=test_result.keys()
    )
    writer.writeheader()
    writer.writerow(test_result)

with open(RESULTS / "margin_validation_sweep.csv", "w", newline="",
          encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=validation_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(validation_results)

print("\nSaved:")
print("results/margin_test_metrics.csv")
print("results/margin_validation_sweep.csv")
