"""Scalar risk, step judgement, compare report and pipeline wiring (no network, no LLM)."""
import types
import unittest

from evidence import controller as C
from evidence.audit import risk
from evidence.compare import report, summarize


def ep(**w):
    st = types.SimpleNamespace()
    args = types.SimpleNamespace(model="m", search_url="x", max_searches=10, max_steps=5)
    return C.Episode(st, "q", None, args, C.Weights(**w))


class Risk(unittest.TestCase):
    def test_formula(self):
        # C=0.5, R=1, X=2 conflicts, lam_x=0.5, lam_r=0.5 -> 0.5 + 1.0 + 0.0
        self.assertAlmostEqual(risk((0.5, 1.0, 0.0, -2.0), 0.5, 0.5), 1.5)
        self.assertAlmostEqual(risk((1.0, 1.0, 0.0, 0.0)), 0.0)

    def test_scalar_mode_judges_by_risk(self):
        e = ep(risk_mode="scalar")
        # coverage up, robustness down a little: pareto says no, scalar risk says yes
        before, after = (0.5, 1.0, 0.0, 0.0), (1.0, 0.8, 0.0, 0.0)
        self.assertTrue(e.better(before, after, "FillGap"))
        self.assertFalse(ep(risk_mode="pareto").better(before, after, "FillGap"))

    def test_discovering_a_conflict_counts_for_diversify(self):
        before, after = (1.0, 0.5, 0.0, 0.0), (1.0, 0.5, 0.0, -1.0)
        for mode in ("pareto", "scalar"):
            self.assertTrue(ep(risk_mode=mode).better(before, after, "Diversify"))
            self.assertFalse(ep(risk_mode=mode).better(before, after, "FillGap"))


class Compare(unittest.TestCase):
    def test_report_has_certificate_and_baseline_tables(self):
        ctrl = [{"question": "a", "em": 1.0, "f1": 1.0, "certified": True, "searches": 4, "llm_calls": 6},
                {"question": "b", "em": 0.0, "f1": 0.0, "certified": False, "searches": 6, "llm_calls": 8}]
        base = [{"question": "a", "em": 1.0, "f1": 1.0, "grounded": True, "searches": 2, "turns": 3},
                {"question": "b", "em": 1.0, "f1": 1.0, "grounded": False, "searches": 3, "turns": 4}]
        s = summarize("c", ctrl)
        self.assertEqual((s["n_cert"], s["em_cert"], s["selective_em"]), (1, 1.0, 0.5))
        text = report({"controller": ctrl, "baseline": base})
        self.assertIn("selective EM", text)
        self.assertIn("answer found in retrieved text", text)
        self.assertIn("| 0 | 1c | 1 |", text)


class Pipeline(unittest.TestCase):
    def test_module_imports_and_parses(self):
        import evidence.pipeline as P
        self.assertTrue(callable(P.main))


if __name__ == "__main__":
    unittest.main()
