"""Shape checks on the golden set. Importing it must never run a model."""

import unittest


class EvalDatasetTests(unittest.TestCase):
    def test_dataset_is_well_formed(self):
        from evals.dataset import GOLDEN_SET

        self.assertGreaterEqual(len(GOLDEN_SET), 10)
        for case in GOLDEN_SET:
            self.assertTrue(case["question"])
            self.assertIn("expected_source", case)

    def test_dataset_covers_the_unanswerable_case(self):
        """A RAG agent that never says 'not found' is a hallucination machine."""
        from evals.dataset import GOLDEN_SET

        self.assertTrue(any(c.get("unanswerable") for c in GOLDEN_SET))

    def test_case_ids_are_unique(self):
        """Per-case scores are reported by id; duplicates would silently merge."""
        from evals.dataset import GOLDEN_SET

        ids = [case["id"] for case in GOLDEN_SET]
        self.assertEqual(len(ids), len(set(ids)))

    def test_answerable_cases_name_a_fixture_and_an_accepted_spelling(self):
        """Correctness is scored by containment, so every case needs a token."""
        import os

        from evals.dataset import GOLDEN_SET

        fixtures_dir = os.path.join(os.path.dirname(__file__), os.pardir, "evals", "fixtures")
        for case in GOLDEN_SET:
            if case.get("unanswerable"):
                self.assertIsNone(case["expected_source"])
                continue
            self.assertTrue(case["accept"], f"{case['id']} has no accepted answer spelling")
            self.assertTrue(
                os.path.exists(os.path.join(fixtures_dir, case["expected_source"])),
                f"{case['id']} names a fixture that does not exist",
            )

    def test_dataset_needs_a_two_document_case(self):
        """One question no single fixture can answer, or retrieval depth is untested."""
        from evals.dataset import GOLDEN_SET

        self.assertTrue(any(c.get("also_requires_source") for c in GOLDEN_SET))
