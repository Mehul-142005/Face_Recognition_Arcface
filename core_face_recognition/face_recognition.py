
"""ArcFace enrollment and live recognition.

Run from project root:
    python -m core_face_recognition.face_recognition --build-gallery
    python -m core_face_recognition.face_recognition --webcam

Dataset layout:
    data/Faces/<identity_name>/*.jpg

The pretrained InsightFace buffalo_l model remains fixed.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

try:
    from .recognition_utils import (
        build_identity_templates,
        cosine_scores,
        decide_identity,
        laplacian_blur_score,
        normalize_embedding,
    )
except ImportError:
    from recognition_utils import (
        build_identity_templates,
        cosine_scores,
        decide_identity,
        laplacian_blur_score,
        normalize_embedding,
    )


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR.parent / "data" / "Faces"
DEFAULT_GALLERY = BASE_DIR / "face_templates.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def initialize_arcface() -> FaceAnalysis:
    print("Initializing InsightFace buffalo_l on CPU...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(ctx_id=-1, det_size=(640, 640))

    print("ArcFace model initialized.")

    return app


def largest_face(faces):
    if not faces:
        return None

    return max(
        faces,
        key=lambda f: max(0, f.bbox[2] - f.bbox[0])
        * max(0, f.bbox[3] - f.bbox[1])
    )


def extract_single_face_embedding(
    app: FaceAnalysis,
    image_path: Path
) -> Optional[np.ndarray]:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[SKIP] Cannot read image: {image_path}")
        return None

    faces = app.get(image)

    if len(faces) == 0:
        print(f"[SKIP] No face detected: {image_path}")
        return None

    if len(faces) > 1:
        print(
            f"[SKIP] Multiple faces found; "
            f"refusing ambiguous enrollment: {image_path}"
        )
        return None

    face = faces[0]

    if not hasattr(face, "embedding"):
        print(f"[SKIP] No ArcFace embedding: {image_path}")
        return None

    return normalize_embedding(face.embedding)


def build_gallery(
    app: FaceAnalysis,
    dataset_dir: Path,
    gallery_path: Path
) -> None:

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Create data/Faces/<identity_name>/ "
            "and place that person's images inside."
        )

    embeddings_by_identity: Dict[str, List[np.ndarray]] = {}

    for identity_dir in sorted(
        p for p in dataset_dir.iterdir() if p.is_dir()
    ):

        image_paths = sorted(
            p for p in identity_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            continue

        vectors = []

        for image_path in tqdm(
            image_paths,
            desc=f"Enrolling {identity_dir.name}"
        ):
            embedding = extract_single_face_embedding(
                app, image_path
            )

            if embedding is not None:
                vectors.append(embedding)

        if vectors:
            embeddings_by_identity[identity_dir.name] = vectors

    templates, counts = build_identity_templates(
        embeddings_by_identity
    )

    if not templates:
        raise RuntimeError(
            "No identity templates created. "
            "Check the dataset and face images."
        )

    gallery_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dimension = len(next(iter(templates.values())))

    fieldnames = (
        ["identity", "enrollment_images"]
        + [f"emb_{i}" for i in range(dimension)]
    )

    with gallery_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for identity, template in templates.items():

            row = {
                "identity": identity,
                "enrollment_images": counts[identity]
            }

            row.update({
                f"emb_{i}": float(value)
                for i, value in enumerate(template)
            })

            writer.writerow(row)

    print(
        f"Saved {len(templates)} identity templates "
        f"to: {gallery_path}"
    )

    for identity, count in counts.items():
        print(
            f"  {identity}: aggregated "
            f"{count} enrollment image(s)"
        )


def load_gallery(gallery_path: Path):

    if not gallery_path.exists():
        raise FileNotFoundError(
            f"Gallery not found: {gallery_path}. "
            "Run with --build-gallery first."
        )

    with gallery_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as handle:

        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Gallery CSV is empty")

    embedding_columns = sorted(
        (c for c in rows[0] if c.startswith("emb_")),
        key=lambda c: int(c.split("_")[1])
    )

    templates = {
        row["identity"]: normalize_embedding(
            np.array(
                [float(row[c]) for c in embedding_columns],
                dtype=np.float32
            )
        )
        for row in rows
    }

    return templates


def recognize_image(
    app,
    image_path: Path,
    templates,
    threshold: float,
    use_margin: bool,
    margin_threshold: float,
    min_blur_score: Optional[float] = None
) -> None:

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    start_time = time.perf_counter()
    faces = app.get(image)
    inference_ms = (
        time.perf_counter() - start_time
    ) * 1000

    print(f"Inference time: {inference_ms:.2f} ms")

    if not faces:
        print("No face detected.")
        return

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        crop = image[
            max(0, y1):max(0, y2),
            max(0, x1):max(0, x2)
        ]

        blur = (
            laplacian_blur_score(crop)
            if crop.size else 0.0
        )

        if (
            min_blur_score is not None
            and blur < min_blur_score
        ):
            print(
                f"Rejected low-quality face "
                f"(blur score={blur:.2f})"
            )
            continue

        scores = cosine_scores(
            face.embedding,
            templates
        )

        identity, best, margin, reason = decide_identity(
            scores,
            threshold,
            use_margin,
            margin_threshold
        )

        margin_text = (
            "n/a" if margin is None
            else f"{margin:.3f}"
        )

        print(
            f"Result={identity} | "
            f"best_similarity={best:.3f} | "
            f"top_two_margin={margin_text} | "
            f"blur={blur:.2f} | "
            f"reason={reason}"
        )


def run_webcam(
    app,
    templates,
    threshold,
    use_margin,
    margin_threshold,
    min_blur_score
):

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW if os.name == "nt" else 0
    )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions "
            "and close other camera apps."
        )

    print("Webcam running. Press Q to quit.")

    try:
        while True:

            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            # Measure face detection and embedding inference time
            start_time = time.perf_counter()

            detected_faces = app.get(frame)

            inference_ms = (
                time.perf_counter() - start_time
            ) * 1000

            for face in detected_faces:

                x1, y1, x2, y2 = face.bbox.astype(int)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]

                blur = (
                    laplacian_blur_score(crop)
                    if crop.size else 0.0
                )

                if (
                    min_blur_score is not None
                    and blur < min_blur_score
                ):

                    identity = "Low quality"
                    best = float("nan")
                    margin = None
                    reason = "quality_gate"

                else:

                    scores = cosine_scores(
                        face.embedding,
                        templates
                    )

                    identity, best, margin, reason = (
                        decide_identity(
                            scores,
                            threshold,
                            use_margin,
                            margin_threshold
                        )
                    )

                color = (
                    (0, 200, 0)
                    if identity not in ("Unknown", "Low quality")
                    else (0, 0, 255)
                )

                score_text = (
                    ""
                    if not np.isfinite(best)
                    else f" {best:.2f}"
                )

                margin_text = (
                    ""
                    if margin is None
                    else f" d={margin:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{identity}{score_text}{margin_text}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2
                )

            # Show inference time on webcam window
            cv2.putText(
                frame,
                f"Inference: {inference_ms:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "ArcFace - Research Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args():

    parser = argparse.ArgumentParser(
        description="ArcFace multi-image enrollment and recognition"
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--gallery",
        type=Path,
        default=DEFAULT_GALLERY
    )

    parser.add_argument(
        "--build-gallery",
        action="store_true",
        help="Build one normalized template per identity"
    )

    parser.add_argument(
        "--image",
        type=Path,
        help="Recognize one image"
    )

    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run live webcam recognition"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine threshold; calibrate on validation data"
    )

    parser.add_argument(
        "--use-margin",
        action="store_true",
        help="Also require a sufficient top-two score margin"
    )

    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.05,
        help="Top-two margin; calibrate on validation data"
    )

    parser.add_argument(
        "--min-blur-score",
        type=float,
        default=None,
        help="Enable Laplacian blur gate; calibrate for your camera"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if not (
        args.build_gallery
        or args.image
        or args.webcam
    ):
        print(
            "Choose --build-gallery, --image PATH, "
            "or --webcam. Use --help for options."
        )
        return

    app = initialize_arcface()

    if args.build_gallery:
        build_gallery(
            app,
            args.dataset_dir,
            args.gallery
        )

    if args.image or args.webcam:

        templates = load_gallery(args.gallery)

        if args.image:
            recognize_image(
                app,
                args.image,
                templates,
                args.threshold,
                args.use_margin,
                args.margin_threshold,
                args.min_blur_score
            )

        if args.webcam:
            run_webcam(
                app,
                templates,
                args.threshold,
                args.use_margin,
                args.margin_threshold,
                args.min_blur_score
            )


if __name__ == "__main__":
    main()
