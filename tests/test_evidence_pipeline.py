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


class PlanConformance(unittest.TestCase):
    def test_naive_provenance_counts_every_passage_as_independent(self):
        import evidence.state as S
        from tests.test_evidence_guards import hold, mk
        st = mk([{"id": "o1", "text": "The director of F is ?D", "depends_on": [], "kind": "retrieve"}], "o1")
        for k in ("a", "b"):
            hold(st, k, "Same Article", "Anna Berg directed F", "o1", {"?D": "Anna Berg"})
        try:
            S.UNIT_MODE = "article"
            self.assertEqual(st.kappa("o1"), 1)       # two passages of one article: one unit
            S.UNIT_MODE = "passage"
            self.assertEqual(st.kappa("o1"), 2)       # naive: independent
        finally:
            S.UNIT_MODE = "article"

    def test_every_executed_step_records_the_plan_reward(self):
        import asyncio
        from evidence.audit import audit
        from tests.test_evidence_guards import FakeClient, hold, mk
        st = mk([{"id": "o1", "text": "The director of F is ?D", "depends_on": [], "kind": "retrieve"}], "o1")
        quote = "Anna Berg directed the film F"
        hold(st, "a", "F", quote, "o1", {"?D": "Anna Berg"})
        client = FakeClient([])
        client.reply = '{"supported": true, "quote": "%s"}' % quote
        args = types.SimpleNamespace(model="m", search_url="x", max_searches=10, max_steps=5)
        e = C.Episode(st, "q", client, args, C.Weights())
        act = e.price(C.Action("Verify", "o1", [], None, 0.15, "single point of failure"))
        cert = audit(st)
        asyncio.run(e.execute(act, 1, cert))
        rec = e.steps[0]
        self.assertEqual(rec["verify"], "supported")
        # robustness 0 -> 1 (+0.5 * 1), cost 0.02 (one LLM call): r = 0.5 - 0.02
        self.assertAlmostEqual(rec["reward"], 0.48, places=3)


class Pipeline(unittest.TestCase):
    def test_module_imports_and_parses(self):
        import evidence.pipeline as P
        self.assertTrue(callable(P.main))

    def test_preflight_rejects_a_dead_api_container(self):      # docker inspect printed "invalid IP"
        import evidence.pipeline as P
        with self.assertRaises(SystemExit) as cm:
            P.preflight("http://invalid IP:8001/search", "http://127.0.0.1:9/v1", "m")
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
