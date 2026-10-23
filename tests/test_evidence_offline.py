"""
Offline tests for the evidence pipeline: no LLM server, no /search API, no graph needed.

  python3 -m unittest tests.test_evidence_offline -v      (from the project root)
"""
import asyncio
import importlib.util
import json
import os
import random
import re
import unittest
import urllib.error
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

from evidence import run_mvp
from evidence.metrics import em_f1
from evidence.obligations import validate
from evidence.provenance import ProvenanceIndex
from evidence.state import EvidenceState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ob(i, text, deps=(), kind="retrieve"):
    return {"id": i, "text": text, "depends_on": list(deps), "kind": kind}


def spec(obs, answer_id, typ="compositional"):
    return {"type": typ, "obligations": obs, "answer_id": answer_id}


class ValidateVariables(unittest.TestCase):
    def test_chain_and_comparison_are_valid(self):
        chain = spec([ob("o1", "The director of Blue River is ?D"),
                      ob("o2", "The mother of ?D is ?M", ["o1"])], "o2")
        self.assertEqual(validate(chain, "Who is the mother of the director of Blue River?"), [])
        comp = spec([ob("o1", "Blue River was released on ?T1"),
                     ob("o2", "Red Lake was released on ?T2"),
                     ob("o3", "Compare ?T1 and ?T2", ["o1", "o2"], "derive")], "o3", "comparison")
        self.assertEqual(validate(comp), [])

    def test_same_name_for_two_unknowns_is_rejected(self):
        bad = spec([ob("o1", "The director of Blue River is ?D"),
                    ob("o2", "The director of Red Lake is ?D")], "o2", "comparison")
        errs = validate(bad)
        self.assertTrue(any("?D" in e and "independently" in e for e in errs), errs)

    def test_missing_depends_on_is_rejected(self):
        bad = spec([ob("o1", "The director of Blue River is ?D"),
                    ob("o2", "The mother of ?D is ?M")], "o2")  # uses ?D but does not depend on o1
        self.assertTrue(any("?D" in e for e in validate(bad)))

    def test_variable_not_in_direct_dependency_is_rejected(self):
        # o3 needs ?D, but depends only on o2, whose text does not mention ?D:
        # EvidenceState inherits values from DIRECT dependencies only, so this must be flagged.
        bad = spec([ob("o1", "The director of Blue River is ?D"),
                    ob("o2", "The producer of Blue River is ?P", ["o1"]),
                    ob("o3", "?D and ?P studied in the same city ?C", ["o2"])], "o3")
        self.assertTrue(any("?D" in e for e in validate(bad)), validate(bad))

    def test_variable_inherited_through_the_dependency_text_is_valid(self):
        ok = spec([ob("o1", "The director of Blue River is ?D"),
                   ob("o2", "The mother of ?D is ?M", ["o1"]),
                   ob("o3", "?D and ?M lived in the same city ?C", ["o2"])], "o3")
        self.assertEqual(validate(ok), [])

    def test_existing_obligation_files_stay_valid(self):
        for name in ("obligations_dev30.jsonl", "obligations_bonsai30.jsonl"):
            path = os.path.join(ROOT, name)
            if not os.path.exists(path):
                self.skipTest(f"{name} not present")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["ok"]:
                        self.assertEqual(validate(r["obligations"], r["question"]), [], r["question"])


class Metrics(unittest.TestCase):
    def test_same_as_repo_evaluation(self):
        try:
            import numpy  # noqa: F401  (evaluation/eval.py needs it)
        except ImportError:
            self.skipTest("numpy not installed")
        path = os.path.join(ROOT, "evaluation", "eval.py")
        sp = importlib.util.spec_from_file_location("repo_eval", path)
        repo = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(repo)
        rnd = random.Random(0)
        words = ["The", "a", "Lapland", "Calendar", "Anna", "lee", "1989", "Film", "yes", "no", "Mary,", "O'Neil"]
        for _ in range(300):
            gold = [" ".join(rnd.choices(words, k=rnd.randint(1, 4))) for _ in range(rnd.randint(1, 3))]
            pred = " ".join(rnd.choices(words, k=rnd.randint(1, 5)))
            em, f1 = em_f1(pred, gold)
            self.assertAlmostEqual(em, repo.cal_em([gold], [pred]), msg=(pred, gold))
            self.assertAlmostEqual(f1, repo.cal_f1([gold], [pred]), msg=(pred, gold))


class StateBookkeeping(unittest.TestCase):
    def make(self):
        return EvidenceState(spec([ob("o1", "The director of Blue River is ?D"),
                                   ob("o2", "The mother of ?D is ?M", ["o1"])], "o2"), ProvenanceIndex({}))

    def test_conflicts_and_unchecked(self):
        st = self.make()
        keys = st.add_items([{"text": "Blue River was directed by Ann Lee.", "coherence": 0.9},
                             {"text": "Blue River was directed by Bob Roe.", "coherence": 0.8}], 1, "q")
        self.assertEqual(st.conflicts(), {})
        st.apply_match("o1", keys[0], {"?D": "Ann Lee"})
        self.assertEqual(st.conflicts(), {})
        st.apply_match("o1", keys[1], {"?D": "Bob Roe"})
        self.assertEqual(sorted(st.conflicts()["?D"]), ["Ann Lee", "Bob Roe"])
        self.assertEqual(len(st.unchecked("o2")), 2)
        st.mark_checked(keys[:1], ["o2"])
        self.assertEqual(st.unchecked("o2"), [keys[1]])


# ---------------------------------------------------------------- fakes for the loop
H_DIRECTOR = "Blue River was directed by Ann Lee."
H_MOTHER = "Ann Lee's mother is Mary Lee."


def fake_search(url, query):
    table = {"The director of Blue River is": [H_DIRECTOR, H_MOTHER],  # one query returns BOTH facts
             "The mother of Ann Lee is": [H_MOTHER]}                   # the 2nd query only repeats it
    return [{"<knowledge>": t, "<coherence>": 0.9} for t in table.get(query, [])]


class FakeLLM:
    """Stands in for the matcher / answer LLM with simple regex rules."""

    def __init__(self):
        self.calls = 0
        self.completions = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=self.completions)

    async def create(self, model, messages, **kw):
        self.calls += 1
        sys_, user = messages[0]["content"], messages[1]["content"]
        if sys_.startswith("You check whether"):
            out = {"matches": self.match(user)}
        else:
            out = {"answer": "guess"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(out)))])

    @staticmethod
    def match(user):
        obs_part, pas_part = user.split("\n\nPassages:\n")
        obs = dict(re.findall(r"^(o\d+): (.+)$", obs_part.split("Obligations:\n")[1], re.M))
        pas = {int(i): t for i, t in re.findall(r"^\[(\d+)\] (.+)$", pas_part, re.M)}
        out = []
        for oid, text in obs.items():
            m1 = re.match(r"The director of (.+) is \?D$", text)
            m2 = re.match(r"The mother of (.+) is \?M$", text)
            for i, p in pas.items():
                if m1 and (g := re.search(rf"{re.escape(m1.group(1))} was directed by ([^.]+)\.", p)):
                    out.append({"passage": i, "obligation": oid, "bindings": {"?D": g.group(1)},
                                "quote": g.group(0).rstrip(".")})
                if m2 and (g := re.search(rf"{re.escape(m2.group(1))}'s mother is ([^.]+)\.", p)):
                    out.append({"passage": i, "obligation": oid, "bindings": {"?M": g.group(1)},
                                "quote": g.group(0).rstrip(".")})
        return out


REC = {"question": "Who is the mother of the director of film Blue River?", "gold": ["Mary Lee"],
       "obligations": spec([ob("o1", "The director of Blue River is ?D"),
                            ob("o2", "The mother of ?D is ?M", ["o1"])], "o2")}


def args(**kw):
    return Namespace(max_rounds=6, model="fake", search_url="fake://", abstain_incomplete=False, **kw)


class SolveLoop(unittest.IsolatedAsyncioTestCase):
    async def run_solve(self, search, **kw):
        llm = FakeLLM()
        with mock.patch.object(run_mvp, "retrieve_sync", search):
            r = await run_mvp.solve(llm, args(**kw), ProvenanceIndex({}), json.loads(json.dumps(REC)))
        return r, llm

    async def test_fact_already_retrieved_for_an_earlier_hop_is_not_lost(self):
        r, _ = await self.run_solve(fake_search)
        self.assertEqual(r["coverage"], 1.0, r["log"])
        self.assertEqual(r["stop"], "all_covered")
        self.assertEqual(r["bindings"], {"?D": "Ann Lee", "?M": "Mary Lee"})
        self.assertEqual([e["phase"] for e in r["log"]], ["query", "recheck"])
        self.assertEqual(r["exhausted"], [])
        self.assertFalse(r["answered_from_incomplete"])

    async def test_truly_missing_fact_is_exhausted_and_reported(self):
        def only_director(url, q):
            return [{"<knowledge>": H_DIRECTOR, "<coherence>": 0.9}] if "director" in q else []
        r, _ = await self.run_solve(only_director)
        self.assertEqual(r["uncovered"], ["o2"])
        self.assertEqual(r["exhausted"], ["o2"])
        self.assertEqual(r["stop"], "no_ready_targets")
        self.assertTrue(r["answered_from_incomplete"])
        self.assertFalse(r["abstained"])
        self.assertEqual(r["answer"], "guess")

    async def test_abstain_flag_returns_empty_answer(self):
        llm = FakeLLM()
        with mock.patch.object(run_mvp, "retrieve_sync", lambda u, q: []):
            r = await run_mvp.solve(llm, Namespace(max_rounds=6, model="fake", search_url="x",
                                                   abstain_incomplete=True),
                                    ProvenanceIndex({}), json.loads(json.dumps(REC)))
        self.assertTrue(r["abstained"])
        self.assertEqual(r["answer"], "")
        self.assertEqual(r["em"], 0.0)
        self.assertFalse(r["answered_from_incomplete"])

    async def test_failed_matcher_call_is_retried_not_forgotten(self):
        class Flaky(FakeLLM):
            async def create(self, model, messages, **kw):
                if messages[0]["content"].startswith("You check") and self.calls < 3:
                    self.calls += 1
                    raise RuntimeError("server down")  # all 3 attempts of the first matcher call fail
                return await super().create(model, messages, **kw)

        llm = Flaky()
        with mock.patch.object(run_mvp, "retrieve_sync", fake_search), \
                mock.patch.object(run_mvp.asyncio, "sleep", mock.AsyncMock()):
            r = await run_mvp.solve(llm, args(), ProvenanceIndex({}), json.loads(json.dumps(REC)))
        self.assertEqual(r["coverage"], 1.0, r["log"])


class RetrieveRetry(unittest.TestCase):
    def test_retries_then_succeeds(self):
        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = b'{"a": 1}'
        with mock.patch.object(run_mvp.urllib.request, "urlopen",
                               side_effect=[urllib.error.URLError("x"), TimeoutError(), ok]) as m, \
                mock.patch.object(run_mvp.time, "sleep"):
            self.assertEqual(run_mvp.retrieve_sync("http://x", "q"), {"a": 1})
            self.assertEqual(m.call_count, 3)

    def test_gives_up_after_retries(self):
        with mock.patch.object(run_mvp.urllib.request, "urlopen", side_effect=TimeoutError()) as m, \
                mock.patch.object(run_mvp.time, "sleep"):
            with self.assertRaises(TimeoutError):
                run_mvp.retrieve_sync("http://x", "q", retries=2)
            self.assertEqual(m.call_count, 2)


if __name__ == "__main__":
    unittest.main()
