"""
Unit Tests for Bayesian Knowledge Tracing (BKT) and Root-Gap Logic.
Verifies:
1. Probabilistic updates upon correct answers (mastery increase).
2. Probabilistic updates upon incorrect answers (mastery decrease / slip adjustment).
3. Convergence to mastery (>= 0.85) after consecutive correct answers.
4. Correct threshold classification (mastered >= 0.85, weak < 0.50).
5. Recursive Root-Gap graph traversal through multiple hops (e.g. simp_eq_solve -> simp_eq_intro -> rat_numbers -> int_mul_div -> int_add_sub -> int_basic).
"""

import unittest
from backend.app.core.bkt import BKTModel, BKTParameters, ConceptGraph, RootGapAnalyzer


class TestBKTAndRootGap(unittest.TestCase):

    def setUp(self):
        self.bkt = BKTModel(BKTParameters(p_init=0.15, p_learn=0.20, p_guess=0.20, p_slip=0.10))
        self.graph = ConceptGraph()
        self.analyzer = RootGapAnalyzer(self.graph, self.bkt)

    def test_bkt_correct_answer_increases_mastery(self):
        prior = 0.15
        updated = self.bkt.update_mastery(prior, is_correct=True)
        self.assertGreater(updated, prior, "A correct answer should increase mastery probability")
        self.assertAlmostEqual(updated, 0.554, delta=0.02)

    def test_bkt_incorrect_answer_decreases_or_bounds_mastery(self):
        prior = 0.50
        updated = self.bkt.update_mastery(prior, is_correct=False)
        self.assertLess(updated, prior, "An incorrect answer should decrease mastery probability")

    def test_bkt_convergence_to_mastery(self):
        # Starting with low prior, consecutive correct answers should achieve mastery >= 0.85
        p = 0.15
        for _ in range(5):
            p = self.bkt.update_mastery(p, is_correct=True)
        self.assertTrue(self.bkt.is_mastered(p), f"Expected mastery >= 0.85, got {p}")
        self.assertGreaterEqual(p, 0.85)

    def test_bkt_repeated_incorrect_leads_to_weak(self):
        p = 0.40
        for _ in range(3):
            p = self.bkt.update_mastery(p, is_correct=False)
        self.assertTrue(self.bkt.is_weak(p), f"Expected weak < 0.50, got {p}")

    def test_root_gap_single_hop(self):
        # If student fails frac_add_sub, and its prerequisite frac_proper_improper is weak
        mastery = {
            "frac_add_sub": 0.20,
            "frac_proper_improper": 0.25
        }
        res = self.analyzer.find_deepest_root_gap("frac_add_sub", mastery)
        self.assertIsNotNone(res)
        root_id, path = res
        self.assertEqual(root_id, "frac_proper_improper")
        self.assertEqual(path, ["frac_add_sub", "frac_proper_improper"])

    def test_root_gap_multi_hop(self):
        # Multi-hop prerequisite chain:
        # simp_eq_solve -> int_add_sub -> int_basic
        mastery = {
            "simp_eq_solve": 0.18,
            "int_add_sub": 0.25,
            "int_basic": 0.22,
            "rat_numbers": 0.90,
            "simp_eq_intro": 0.90
        }
        res = self.analyzer.find_deepest_root_gap("simp_eq_solve", mastery)
        self.assertIsNotNone(res)
        root_id, path = res
        self.assertEqual(root_id, "int_basic", "Should reach the deepest weak ancestor in the graph")
        self.assertIn("int_basic", path)
        self.assertIn("simp_eq_solve", path)

    def test_root_gap_when_prereqs_are_mastered(self):
        # Target concept is weak, but all prerequisites are mastered
        mastery = {
            "int_add_sub": 0.30,
            "int_basic": 0.95
        }
        res = self.analyzer.find_deepest_root_gap("int_add_sub", mastery)
        self.assertIsNotNone(res)
        root_id, path = res
        # The concept itself is the root gap since prerequisites are mastered
        self.assertEqual(root_id, "int_add_sub")
        self.assertEqual(path, ["int_add_sub"])

    def test_analyze_student_gaps_explanations(self):
        mastery = {
            "dec_ops": 0.30,
            "frac_proper_improper": 0.25,
            "dec_place_value": 0.90
        }
        results = self.analyzer.analyze_student_gaps(mastery)
        self.assertTrue(len(results) > 0)
        first_gap = results[0]
        self.assertIn("explanation", first_gap)
        self.assertIn("en", first_gap["explanation"])
        self.assertIn("hi", first_gap["explanation"])
        self.assertIn("mr", first_gap["explanation"])


if __name__ == "__main__":
    unittest.main()
