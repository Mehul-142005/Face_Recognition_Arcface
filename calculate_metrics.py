
import csv
from pathlib import Path

RESULTS = Path("results")

def load_records(filename):
    with open(RESULTS / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


validation = load_records("validation_scores.csv")
queries = load_records("queries_scores.csv")


def calculate(records, threshold):
    known = [r for r in records if r["true_identity"] != "Unknown"]
    unknown = [r for r in records if r["true_identity"] == "Unknown"]

    genuine_accepted = sum(
        float(r["top_score"]) >= threshold for r in known
    )
    unknown_accepted = sum(
        float(r["top_score"]) >= threshold for r in unknown
    )

    tar = genuine_accepted / len(known) if known else 0
    frr = 1 - tar
    far = unknown_accepted / len(unknown) if unknown else 0
    unknown_rejection = 1 - far

    rank1_correct = sum(
        r["predicted_identity"] == r["true_identity"]
        for r in known
    )
    rank1 = rank1_correct / len(known) if known else 0

    balanced_accuracy = (tar + unknown_rejection) / 2

    return {
        "threshold": threshold,
        "known_count": len(known),
        "unknown_count": len(unknown),
        "TAR": tar,
        "FRR": frr,
        "FAR": far,
        "unknown_rejection": unknown_rejection,
        "rank1_accuracy": rank1,
        "balanced_accuracy": balanced_accuracy
    }


# Candidate thresholds from validation scores only
scores = sorted(set(float(r["top_score"]) for r in validation))

candidates = [scores[0] - 0.000001]

for a, b in zip(scores, scores[1:]):
    candidates.append((a + b) / 2)

candidates.append(scores[-1] + 0.000001)

validation_results = [
    calculate(validation, t) for t in candidates
]

# Select threshold with highest validation balanced accuracy.
# On ties, choose the higher threshold.
best = max(
    validation_results,
    key=lambda r: (r["balanced_accuracy"], r["threshold"])
)

selected_threshold = best["threshold"]

print("\n===== VALIDATION CALIBRATION =====")
print("Selected threshold:", round(selected_threshold, 6))

print("\n===== VALIDATION METRICS =====")
for key, value in best.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float)
          else f"{key}: {value}")

# Apply the fixed validation threshold to test data
test_result = calculate(queries, selected_threshold)

print("\n===== FINAL TEST METRICS =====")
for key, value in test_result.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float)
          else f"{key}: {value}")

# Save threshold sweep
with open(RESULTS / "threshold_sweep.csv", "w", newline="",
          encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=validation_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(validation_results)

# Save final test metrics
with open(RESULTS / "final_test_metrics.csv", "w", newline="",
          encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=test_result.keys()
    )
    writer.writeheader()
    writer.writerow(test_result)

print("\nSaved results:")
print("results/threshold_sweep.csv")
print("results/final_test_metrics.csv")
