"""
Tests for the fixes after the first real runs: learn_p double counting, one flag per question,
the "ungrounded" stop label, computed date comparisons (the Exit Smiling / Shapath failure),
held-out sampling and the Verify summary.   (no network, no LLM)

  python3 -m unittest tests.test_evidence_fixes -v
"""
import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from evidence import controller
from evidence.controller import Episode, Weights
from evidence.derive import derive_answer, direction, earlier, parse_date, question_options
from evidence.obligations import excluded_questions
from evidence.provenance import ProvenanceIndex, norm
from evidence.state import EvidenceState
from tests.test_evidence_controller import CHAIN, FULL, QUESTION, FakeLLM, args, make_search, prov


def ob(i, text, deps=(), kind="retrieve"):
    return {"id": i, "text": text, "depends_on": list(deps), "kind": kind}


# --------------------------------------------------------------------------- learn_p
class LearnP(unittest.IsolatedAsyncioTestCase):
    def test_no_double_counting_within_a_question(self):
        W = Weights(learn_p=True)
        ep = Episode(EvidenceState(json.loads(json.dumps(CHAIN)), prov()), QUESTION, None, args(), W)
        ep.stats["Diversify"][1] += 1              # what execute() does for one failed attempt ...
        W.seen.setdefault("Diversify", [0, 0])[1] += 1   # ... it updates both counters
        self.assertAlmostEqual(ep.p("Diversify"), 1 / 5)          # (0 + 1) / (1 + 4), not 1 / 6
        ep2 = Episode(EvidenceState(json.loads(json.dumps(CHAIN)), prov()), QUESTION, None, args(), Weights())
        ep2.stats["Diversify"][1] += 1
        self.assertAlmostEqual(ep2.p("Diversify"), ep.p("Diversify"))   # same evidence, same p, learn_p or not

    async def test_run_wide_counter_equals_the_sum_of_the_questions(self):
        W = Weights(learn_p=True)
        total = {}
        for _ in range(2):
            rec = {"question": QUESTION, "gold": ["Mary Lee"], "obligations": json.loads(json.dumps(CHAIN))}
            with mock.patch.object(controller, "retrieve_sync", make_search(FULL)), \
                    mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
                r = await controller.solve(FakeLLM(answer="Mary Lee"), args(), prov(), rec, W)
            for k, v in r["action_stats"].items():
                t = total.setdefault(k, [0, 0])
                t[0] += v["success"]
                t[1] += v["attempts"]
        self.assertEqual({k: list(v) for k, v in W.seen.items()}, total)


# --------------------------------------------------------------------------- monitor flags / stop label
class Monitor(unittest.IsolatedAsyncioTestCase):
    def test_flag_is_counted_once_per_question(self):
        ep = Episode(EvidenceState(json.loads(json.dumps(CHAIN)), prov()), QUESTION, None, args(), Weights())
        ep.stall = 5
        for _ in range(4):
            ep.monitor(controller.audit(ep.state))
        self.assertEqual(ep.flags_seen["retrieval_loop"], 1)

    async def test_certified_flag_and_stop_label_when_the_answer_is_not_backed(self):
        rec = {"question": QUESTION, "gold": ["Mary Lee"], "obligations": json.loads(json.dumps(CHAIN))}
        with mock.patch.object(controller, "retrieve_sync", make_search(FULL)), \
                mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
            r = await controller.solve(FakeLLM(answer="Harvard"), args(), prov(), rec)
        self.assertTrue(r["certificate"]["reasons"])
        self.assertFalse(r["certified"])
        self.assertEqual(r["stop"], "ungrounded")           # not "certified" while certified=False
        self.assertIn("not grounded", " ".join(r["certificate"]["reasons"]))

    async def test_backed_answer_keeps_the_certified_label(self):
        rec = {"question": QUESTION, "gold": ["Mary Lee"], "obligations": json.loads(json.dumps(CHAIN))}
        with mock.patch.object(controller, "retrieve_sync", make_search(FULL)), \
                mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
            r = await controller.solve(FakeLLM(answer="Mary Lee"), args(), prov(), rec)
        self.assertTrue(r["certified"])
        self.assertEqual(r["stop"], "certified")


# --------------------------------------------------------------------------- date parsing and comparison
class DateParsing(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_date("18 November 1888"), (1888, 11, 18))
        self.assertEqual(parse_date("May 5, 1925"), (1925, 5, 5))
        self.assertEqual(parse_date("March 1888"), (1888, 3, 0))
        self.assertEqual(parse_date("a 1926 comedy"), (1926, 0, 0))
        self.assertIsNone(parse_date("unknown"))

    def test_comparison_uses_the_shared_precision(self):
        self.assertTrue(earlier((1926, 0, 0), (1997, 5, 5)))
        self.assertFalse(earlier((1997, 0, 0), (1926, 0, 0)))
        self.assertIsNone(earlier((1888, 11, 18), (1888, 0, 0)))      # same year, one has no month: cannot tell
        self.assertTrue(earlier((1888, 3, 0), (1888, 11, 18)))

    def test_question_forms(self):
        cases = [("Which film was released more recently, Exit Smiling or Shapath?", ("Exit Smiling", "Shapath"), "max"),
                 ("Who was born first out of Otto Schimek and Yip Ho Nung?", ("Otto Schimek", "Yip Ho Nung"), "min"),
                 ("Which film has the director who is older, Fratelli D'Italia (1989 Film) or Lapland Calendar?",
                  ("Fratelli D'Italia (1989 Film)", "Lapland Calendar"), "min"),
                 ("Which film whose director is younger, Swaroopam or Swedenhielms?", ("Swaroopam", "Swedenhielms"), "max")]
        for q, opts, way in cases:
            self.assertEqual(question_options(q), opts, q)
            self.assertEqual(direction(q), way, q)

    def test_yes_no_questions_are_left_to_the_llm(self):
        q = "Do both films The Company of Wolves and Down with Love have the directors from the same country?"
        self.assertIsNone(direction(q))


# comparison state ---------------------------------------------------------------------------------
CMP = {"type": "comparison", "answer_id": "o3", "obligations": [
    ob("o1", "Exit Smiling was released on ?T1"),
    ob("o2", "Shapath was released on ?T2"),
    ob("o3", "Compare ?T1 and ?T2 to find the more recently released film", ["o1", "o2"], "derive")]}
CMPQ = "Which film was released more recently, Exit Smiling or Shapath?"
BRIDGE = {"type": "bridge_comparison", "answer_id": "o5", "obligations": [
    ob("o1", "The director of Fratelli D'Italia (1989 Film) is ?D1"),
    ob("o2", "The director of Lapland Calendar is ?D2"),
    ob("o3", "?D1 was born on ?T1", ["o1"]),
    ob("o4", "?D2 was born on ?T2", ["o2"]),
    ob("o5", "Compare ?T1 and ?T2 to determine which director is older, then name the film", ["o3", "o4"], "derive")]}
BRIDGEQ = "Which film has the director who is older, Fratelli D'Italia (1989 Film) or Lapland Calendar?"


def held(spec, facts):
    """EvidenceState with support entries; facts = [(oid, passage text, {var: value})]."""
    st = EvidenceState(json.loads(json.dumps(spec)), ProvenanceIndex({}))
    for oid, text, b in facts:
        k = st.add_items([{"text": text, "coherence": 0.9}], 1, "q")[0]
        st.apply_match(oid, k, b)
    return st


class DeriveAnswer(unittest.TestCase):
    def test_release_dates(self):
        st = held(CMP, [("o1", "Exit Smiling is a 1926 comedy film.", {"?T1": "1926"}),
                        ("o2", "Shapath is a 1997 Indian film.", {"?T2": "1997"})])
        d = derive_answer(st, CMPQ)
        self.assertEqual(d["answer"], "Shapath")
        self.assertEqual(d["dates"], {"Exit Smiling": "1926", "Shapath": "1997"})

    def test_direction_flips_the_answer(self):
        st = held(CMP, [("o1", "Exit Smiling is a 1926 comedy film.", {"?T1": "1926"}),
                        ("o2", "Shapath is a 1997 Indian film.", {"?T2": "1997"})])
        self.assertEqual(derive_answer(st, "Which film came out first, Exit Smiling or Shapath?")["answer"], "Exit Smiling")

    def test_dates_reach_the_option_through_the_chain(self):          # bridge_comparison: film -> director -> birth
        st = held(BRIDGE, [("o1", "Fratelli D'Italia was directed by Neri Parenti.", {"?D1": "Neri Parenti"}),
                           ("o2", "Lapland Calendar was directed by Per Host.", {"?D2": "Per Host"}),
                           ("o3", "Neri Parenti was born 26 April 1950.", {"?T1": "26 April 1950"}),
                           ("o4", "Per Host was born 5 December 1907.", {"?T2": "5 December 1907"})])
        d = derive_answer(st, BRIDGEQ)
        self.assertEqual(d["answer"], "Lapland Calendar")             # born 1907 -> older
        self.assertEqual(d["dates"]["Fratelli D'Italia (1989 Film)"], "26 April 1950")

    def test_falls_back_to_the_llm_when_it_cannot_be_sure(self):
        one = held(CMP, [("o1", "Exit Smiling is a 1926 comedy film.", {"?T1": "1926"})])
        self.assertIsNone(derive_answer(one, CMPQ), "second date missing")
        tie = held(CMP, [("o1", "Exit Smiling is a 1990 film.", {"?T1": "1990"}),
                         ("o2", "Shapath is a 1990 film.", {"?T2": "1990"})])
        self.assertIsNone(derive_answer(tie, CMPQ), "same year: cannot tell")
        nodate = held(CMP, [("o1", "Exit Smiling is a silent film.", {"?T1": "silent"}),
                            ("o2", "Shapath is a 1997 Indian film.", {"?T2": "1997"})])
        self.assertIsNone(derive_answer(nodate, CMPQ))
        st = held(CMP, [("o1", "Exit Smiling is a 1926 comedy film.", {"?T1": "1926"}),
                        ("o2", "Shapath is a 1997 Indian film.", {"?T2": "1997"})])
        self.assertIsNone(derive_answer(st, "Do both films Exit Smiling and Shapath have the same release year?"))
        self.assertIsNone(derive_answer(st, "Which film was released more recently, Exit Smiling or Another Film?"),
                          "an option that matches no obligation")

    def test_retrieve_answers_are_not_touched(self):
        st = held(CHAIN, [])
        self.assertIsNone(derive_answer(st, QUESTION))


class ComparisonEndToEnd(unittest.IsolatedAsyncioTestCase):
    """The Exit Smiling / Shapath case: certified evidence, wrong free-text comparison by the LLM."""
    PASS = {"Exit Smiling is a 1926 comedy film.": "Exit Smiling", "Shapath is a 1997 Indian film.": "Shapath"}

    def world(self):
        data = {norm(t): {"titles": [a], "chunks": [], "how": "exact"} for t, a in self.PASS.items()}
        table = {"Exit Smiling was released on": [{"<knowledge>": "Exit Smiling is a 1926 comedy film.", "<coherence>": 0.9}],
                 "Shapath was released on": [{"<knowledge>": "Shapath is a 1997 Indian film.", "<coherence>": 0.9}]}

        class LLM(FakeLLM):
            def match(self, user):
                obs_part, pas_part = user.split("\n\nPassages:\n")
                obs = dict(re.findall(r"^(o\d+): (.+)$", obs_part.split("Obligations:\n")[1], re.M))
                pas = {int(i): t for i, t in re.findall(r"^\[(\d+)\] (.+)$", pas_part, re.M)}
                out = []
                for oid, text in obs.items():
                    m = re.match(r"(.+) was released on (\?T\d)$", text)
                    for i, p in pas.items():
                        g = m and re.match(rf"{re.escape(m.group(1))} is a (\d{{4}}) ", p)
                        if g:
                            out.append({"passage": i, "obligation": oid, "bindings": {m.group(2): g.group(1)},
                                        "quote": p.rstrip(".")})
                return out
        return ProvenanceIndex(data), table, LLM

    async def run_q(self, a, **kw):
        pv, table, LLM = self.world()
        rec = {"question": CMPQ, "gold": ["Shapath"], "obligations": json.loads(json.dumps(CMP))}
        with mock.patch.object(controller, "retrieve_sync", make_search(table)), \
                mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
            return await controller.solve(LLM(answer="Exit Smiling"), a, pv, rec)

    async def test_computed_answer_replaces_the_wrong_llm_answer(self):
        r = await self.run_q(args())
        self.assertTrue(r["certified"], r["certificate"]["reasons"])
        self.assertEqual(r["answer_llm"], "Exit Smiling")
        self.assertEqual(r["answer"], "Shapath")
        self.assertEqual(r["em"], 1.0)
        self.assertEqual(r["pathologies"].get("derive_mismatch"), 1)
        self.assertEqual(r["derived"]["dates"], {"Exit Smiling": "1926", "Shapath": "1997"})

    async def test_switch_off_keeps_the_llm_answer(self):
        r = await self.run_q(args(det_compare=False))
        self.assertEqual(r["answer"], "Exit Smiling")
        self.assertEqual(r["em"], 0.0)
        self.assertIsNone(r["derived"])
        self.assertNotIn("derive_mismatch", r["pathologies"])


# --------------------------------------------------------------------------- held-out sampling, summary
class HeldOut(unittest.TestCase):
    def test_excluded_questions(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "o.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"question": " Who is A? ", "ok": True}) + "\n\n" + json.dumps({"question": "Who is B?"}) + "\n")
            self.assertEqual(excluded_questions([p]), {"Who is A?", "Who is B?"})
            self.assertEqual(excluded_questions([]), set())


class Summary(unittest.TestCase):
    def test_verify_outcomes_and_derive_counts_are_reported(self):
        base = {"certified": True, "abstained": False, "em": 1.0, "f1": 1.0, "searches": 2, "llm_calls": 3,
                "stop": "certified", "action_stats": {}, "pathologies": {}, "steps": [], "derived": None}
        rs = [dict(base, steps=[{"verify": "supported"}, {"verify": "rejected"}], pathologies={"confirmation_bias": 1}),
              dict(base, steps=[{"verify": "supported"}], derived={"answer": "X"}, pathologies={"derive_mismatch": 1})]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            controller.summary(rs, "m")
        out = buf.getvalue()
        self.assertIn("Verify outcomes: {'supported': 2, 'rejected': 1}", out)
        self.assertIn("computed date comparisons: 1 | the LLM's own comparison differed in 1", out)
        self.assertIn("questions flagged by the monitor", out)


if __name__ == "__main__":
    unittest.main()
