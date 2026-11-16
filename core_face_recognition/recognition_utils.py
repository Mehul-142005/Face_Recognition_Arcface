"""Research utilities for ArcFace template construction and open-set decisions.

All embeddings are L2-normalized. Similarity is therefore a dot product.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple
import cv2
import numpy as np


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding has zero or invalid norm")
    return vector / norm


def build_identity_templates(embeddings_by_identity: Mapping[str, Iterable[np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Average each identity's normalized enrollment embeddings, then normalize."""
    templates: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for identity, embeddings in embeddings_by_identity.items():
        vectors = [normalize_embedding(e) for e in embeddings]
        if not vectors:
            continue
        dimensions = {v.shape[0] for v in vectors}
        if len(dimensions) != 1:
            raise ValueError(f"Inconsistent embedding dimensions for {identity}")
        templates[str(identity)] = normalize_embedding(np.mean(np.stack(vectors), axis=0))
        counts[str(identity)] = len(vectors)
    return templates, counts


def cosine_scores(query_embedding: np.ndarray, templates: Mapping[str, np.ndarray]) -> Dict[str, float]:
    query = normalize_embedding(query_embedding)
    scores = {}
    for identity, template in templates.items():
        candidate = normalize_embedding(template)
        if candidate.shape != query.shape:
            raise ValueError(f"Embedding dimension mismatch for {identity}")
        scores[identity] = float(np.dot(query, candidate))
    return scores


def decide_identity(scores: Mapping[str, float], threshold: float = 0.5,
                     use_margin: bool = False, margin_threshold: float = 0.05
                     ) -> Tuple[str, float, Optional[float], str]:
    """Return (identity-or-Unknown, best_score, margin, reason).

    Threshold and margin are experimental parameters; calibrate them on a
    validation set before reporting performance.
    """
    if not scores:
        return "Unknown", float("nan"), None, "empty_gallery"
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_identity, best_score = ranked[0]
    margin = (best_score - ranked[1][1]) if len(ranked) > 1 else None
    if best_score < threshold:
        return "Unknown", best_score, margin, "below_similarity_threshold"
    if use_margin:
        if margin is None:
            return "Unknown", best_score, None, "margin_requires_two_identities"
        if margin < margin_threshold:
            return "Unknown", best_score, margin, "ambiguous_top_two_scores"
    return best_identity, best_score, margin, "accepted"


def laplacian_blur_score(face_bgr: np.ndarray) -> float:
    """Variance of Laplacian; larger values often indicate sharper images.

    This is a simple proxy, not a universal perceptual-quality metric.
    """
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Empty face crop")
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
