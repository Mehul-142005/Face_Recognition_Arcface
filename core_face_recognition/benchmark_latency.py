
import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from core_face_recognition.face_recognition import initialize_arcface


def percentile(values, p):
    return float(np.percentile(values, p))


def run_benchmark(warmup_frames=20, test_frames=100, camera_index=0):
    print("Initializing ArcFace on CPU...")
    app = initialize_arcface()

    print("Opening webcam...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Close other camera applications and retry."
        )

    timings = []
    detected_faces = []

    try:
        print(f"\nWarm-up: processing {warmup_frames} frames...")

        for i in range(warmup_frames):
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Could not read a frame from the webcam.")

            app.get(frame)

            if (i + 1) % 5 == 0:
                print(f"Warm-up: {i + 1}/{warmup_frames}")

        print(f"\nBenchmark: processing {test_frames} frames...")

        for i in range(test_frames):
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Could not read a frame from the webcam.")

            start = time.perf_counter()
            faces = app.get(frame)
            end = time.perf_counter()

            inference_ms = (end - start) * 1000

            timings.append(inference_ms)
            detected_faces.append(len(faces))

            if (i + 1) % 10 == 0:
                print(
                    f"Processed {i + 1}/{test_frames} frames | "
                    f"Latest inference: {inference_ms:.2f} ms"
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not timings:
        raise RuntimeError("No benchmark measurements were collected.")

    average = statistics.mean(timings)
    median = statistics.median(timings)
    p95 = percentile(timings, 95)
    minimum = min(timings)
    maximum = max(timings)

    print("\n========== CPU LATENCY RESULTS ==========")
    print(f"Frames measured       : {len(timings)}")
    print(f"Average latency       : {average:.2f} ms")
    print(f"Median latency        : {median:.2f} ms")
    print(f"95th percentile       : {p95:.2f} ms")
    print(f"Minimum latency       : {minimum:.2f} ms")
    print(f"Maximum latency       : {maximum:.2f} ms")
    print(f"Average detected faces: {statistics.mean(detected_faces):.2f}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / "cpu_latency_results.csv"

    with output_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_number", "inference_ms", "detected_faces"])

        for i, (latency, count) in enumerate(
            zip(timings, detected_faces), start=1
        ):
            writer.writerow([i, f"{latency:.4f}", count])

    summary_file = results_dir / "cpu_latency_summary.csv"

    with summary_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value_ms"])
        writer.writerow(["average", f"{average:.4f}"])
        writer.writerow(["median", f"{median:.4f}"])
        writer.writerow(["p95", f"{p95:.4f}"])
        writer.writerow(["minimum", f"{minimum:.4f}"])
        writer.writerow(["maximum", f"{maximum:.4f}"])

    print("\nSaved detailed results:", output_file)
    print("Saved summary results :", summary_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ArcFace CPU inference latency."
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    if args.warmup < 0 or args.frames < 1:
        parser.error("--warmup must be non-negative and --frames must be positive")

    run_benchmark(args.warmup, args.frames, args.camera)
