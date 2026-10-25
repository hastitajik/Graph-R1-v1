"""
Matcher gets the question, obligations keep the entity type word, FillGap can use an INFORMED LLM query
(it reads what was retrieved so far), a rejected Verify is recorded with its evidence.   (no network)

  python3 -m unittest tests.test_evidence_informed -v
"""
import unittest

from evidence.obligations import validate
import json
from unittest import mock

from evidence import controller
from tests.test_evidence_controller import ControllerTests, FULL, TEXT, FakeLLM, hit


def obl(texts, answer="o1"):
    return {"type": "compositional", "answer_id": answer,
            "obligations": [{"id": f"o{i + 1}", "text": t, "depends_on": [], "kind": "retrieve"} for i, t in enumerate(texts)]}


class TypeWord(unittest.TestCase):
    Q = "Who is the director of film Well Wishes?"

    def test_dropped_type_word_is_reported(self):
        errs = validate(obl(["The director of Well Wishes is ?D"]), self.Q)
        self.assertTrue(any("'film'" in e for e in errs), errs)

    def test_kept_type_word_is_valid(self):
        self.assertEqual(validate(obl(["The director of the film Well Wishes is ?D"]), self.Q), [])

    def test_stored_files_can_skip_the_check(self):
        self.assertEqual(validate(obl(["The director of Well Wishes is ?D"]), self.Q, check_types=False), [])

    def test_type_word_without_a_name_is_not_required(self):
        self.assertEqual(validate(obl(["Who directed ?F is ?D"]), "Which film was directed by ann?"), [])


class Informed(unittest.IsolatedAsyncioTestCase):
    run_ep = ControllerTests.run_ep

    async def test_matcher_prompt_contains_the_question(self):
        r, llm = await self.run_ep(FULL)
        self.assertTrue(llm.match_prompts)
        self.assertTrue(all(p.startswith("Question: Who is the mother of the director") for p in llm.match_prompts))

    async def test_informed_queries_read_the_retrieved_passages(self):
        # the plain forms find a passage about something else; only the informed query (built from the
        # title seen in that passage) finds the director
        llm = FakeLLM(reform={"paraphrase": ["unhelpful words"], "informed": ["Blue River (1999) director"]},
                      answer="Mary Lee")
        table = dict(FULL)
        table["The director of Blue River is"] = hit("mom_a")      # retrieved, but not the fact
        table["director Blue River"] = []
        table["Blue River (1999) director"] = hit("dir_a")
        r, llm = await self.run_ep(table, llm=llm, max_steps=20, max_searches=20)
        self.assertEqual(r["bindings"].get("?D"), "Ann Lee")
        self.assertTrue(llm.informed_prompts, "the informed mode must have been used")
        self.assertIn("Retrieved so far:", llm.informed_prompts[0])
        self.assertIn("Ann Lee", llm.informed_prompts[0])        # title of the passage that was retrieved
        step = next(s for s in r["steps"] if "Blue River (1999) director" in s["queries"])
        self.assertEqual(step["llm_queries"], ["Blue River (1999) director"])

    async def test_informed_rounds_are_limited(self):
        llm = FakeLLM(reform={"informed": ["x one", "x two"]}, answer="guess")
        table = {"The director of Blue River is": hit("mom_a")}
        r, llm = await self.run_ep(table, llm=llm, max_steps=30, max_searches=40)
        self.assertLessEqual(len(llm.informed_prompts), 2 * 2)        # two obligations, two rounds each

    async def test_rejected_verify_keeps_fact_and_passage(self):
        llm = FakeLLM(reject=[TEXT["dir_a"]], answer="guess")
        table = {k: v for k, v in FULL.items() if k != "The director of Blue River is Ann Lee"}
        r, _ = await self.run_ep(table, llm=llm)
        rej = next(s for s in r["steps"] if s.get("verify") == "rejected")
        self.assertIn("Blue River", rej["verify_rejected"]["fact"])
        self.assertEqual(rej["verify_rejected"]["passage"], TEXT["dir_a"])


if __name__ == "__main__":
    unittest.main()


class CompareStats(unittest.TestCase):
    def test_wilson_and_mcnemar(self):
        from evidence.compare import mcnemar_exact, paired, wilson
        lo, hi = wilson(23, 30)
        self.assertTrue(0.58 < lo < 0.6 and 0.88 < hi < 0.9, (lo, hi))
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(9, 1), 0.021484375)
        ref = [{"question": str(i), "em": 1.0 if i < 20 else 0.0} for i in range(30)]
        oth = [{"question": str(i), "em": 1.0 if i < 10 else 0.0} for i in range(30)]
        c = paired(ref, oth)
        self.assertEqual((c["wins"], c["losses"], c["n"]), (0, 10, 30))
        self.assertLess(c["hi"], 0.0)

    def test_report_has_all_sections(self):
        from evidence.compare import report
        mk = lambda em, cert: {"question": "q", "em": em, "f1": em, "certified": cert, "coverage": 1.0, "searches": 3,
                               "llm_calls": 4, "steps": [{"verify_rejected": {"fact": "f", "passage": "p"}}]}
        text = report({"a": [mk(1.0, True)], "b": [mk(0.0, False)]}, rejected=True)
        for part in ("EM 95% CI", "Selective prediction", "Paired comparison", "Verify rejections"):
            self.assertIn(part, text)


class SurnameAlias(unittest.TestCase):
    def test_bare_surname_is_the_same_entity(self):
        from evidence.state import same_entity
        self.assertTrue(same_entity("Annakin", "Ken Annakin"))
        self.assertTrue(same_entity("Ken Annakin", "Annakin"))
        self.assertFalse(same_entity("Lee", "Ann Lee"))            # too short to be safe
        self.assertFalse(same_entity("Ann Lee", "Bob Roe"))
        self.assertFalse(same_entity("1999", "Ann 1999"))
