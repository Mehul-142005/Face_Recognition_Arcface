
import unittest
import numpy as np

from core_face_recognition.recognition_utils import (
    build_identity_templates,
    cosine_scores,
    decide_identity,
    normalize_embedding
)


class RecognitionUtilsTests(unittest.TestCase):

    def test_template_is_normalized_average(self):
        templates, counts = build_identity_templates({
            "A": [np.array([1., 0.]), np.array([0.9, 0.1])],
            "B": [np.array([0., 1.])]
        })

        self.assertEqual(counts["A"], 2)
        self.assertAlmostEqual(
            float(np.linalg.norm(templates["A"])), 1.0, places=6
        )
        self.assertGreater(templates["A"][0], templates["A"][1])

    def test_margin_rejects_ambiguous_match(self):
        identity, best, margin, reason = decide_identity(
            {"A": 0.78, "B": 0.76},
            threshold=0.5,
            use_margin=True,
            margin_threshold=0.05
        )

        self.assertEqual(identity, "Unknown")
        self.assertEqual(reason, "ambiguous_top_two_scores")
        self.assertAlmostEqual(margin, 0.02, places=6)

    def test_margin_accepts_clear_match(self):
        identity, _, margin, reason = decide_identity(
            {"A": 0.88, "B": 0.42},
            threshold=0.5,
            use_margin=True,
            margin_threshold=0.05
        )

        self.assertEqual(identity, "A")
        self.assertEqual(reason, "accepted")
        self.assertAlmostEqual(margin, 0.46, places=6)

    def test_threshold_rejects_low_score(self):
        identity, _, _, reason = decide_identity(
            {"A": 0.31}, threshold=0.5
        )

        self.assertEqual(identity, "Unknown")
        self.assertEqual(reason, "below_similarity_threshold")

    def test_scores_use_cosine_similarity(self):
        scores = cosine_scores(
            np.array([1., 0.]),
            {
                "A": np.array([1., 0.]),
                "B": np.array([0., 1.])
            }
        )

        self.assertAlmostEqual(scores["A"], 1.0, places=6)
        self.assertAlmostEqual(scores["B"], 0.0, places=6)

    def test_empty_gallery_returns_unknown(self):
        identity, best, margin, reason = decide_identity({})

        self.assertEqual(identity, "Unknown")
        self.assertTrue(np.isnan(best))
        self.assertIsNone(margin)
        self.assertEqual(reason, "empty_gallery")

    def test_zero_embedding_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_embedding(np.array([0., 0.]))

    def test_invalid_embedding_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_embedding(np.array([np.nan, 0.]))

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_scores(
                np.array([1., 0.]),
                {"A": np.array([1., 0., 0.])}
            )

    def test_empty_enrollment_is_skipped(self):
        templates, counts = build_identity_templates({
            "A": [],
            "B": [np.array([1., 0.])]
        })

        self.assertNotIn("A", templates)
        self.assertIn("B", templates)
        self.assertEqual(counts["B"], 1)

    def test_single_identity_margin_is_rejected(self):
        identity, _, margin, reason = decide_identity(
            {"A": 0.9},
            threshold=0.5,
            use_margin=True
        )

        self.assertEqual(identity, "Unknown")
        self.assertIsNone(margin)
        self.assertEqual(reason, "margin_requires_two_identities")

    def test_exact_threshold_is_accepted(self):
        identity, _, _, reason = decide_identity(
            {"A": 0.5}, threshold=0.5
        )

        self.assertEqual(identity, "A")
        self.assertEqual(reason, "accepted")


if __name__ == "__main__":
    unittest.main()
