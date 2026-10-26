"""
Soundness fixes found by an independent review: false certificates, double-counted sources, aliases,
search failures, learned statistics.   (no network)

  python3 -m unittest tests.test_evidence_soundness -v
"""
import json
import unittest
from unittest import mock

from evidence import controller
from evidence.audit import audit
from evidence.controller import Episode, Weights, grounding
from evidence.obligations import validate
from evidence.provenance import ProvenanceIndex, norm
from evidence.state import EvidenceState, same_entity
from tests.test_evidence_controller import CHAIN, FULL, QUESTION, FakeLLM, args, make_search, prov


def ob(i, text, deps=(), kind="retrieve"):
    return {"id": i, "text": text, "depends_on": list(deps), "kind": kind}


def state_with(obs, answer, titles_by_text, question=""):
    idx = ProvenanceIndex({norm(t): {"titles": ts, "chunks": [], "how": "exact"} for t, ts in titles_by_text.items()})
    st = EvidenceState({"obligations": obs, "answer_id": answer}, idx)
    st.question = question
    return st


def hold(st, oid, text, bindings, rnd=1):
    new = st.add_items([{"text": text, "coherence": 0.9}], rnd, "q")
    k = new[0] if new else st.prov.key(text)
    st.apply_match(oid, k, bindings)
    return k


class Grounding(unittest.TestCase):
    def chain(self):
        st = state_with([ob("o1", "The director of B is ?D"), ob("o2", "The mother of ?D is ?M", ["o1"])], "o2",
                        {"B was directed by Ann Lee": ["B"], "Ann Lee's mother is Mary Lee": ["Ann Lee"]})
        hold(st, "o1", "B was directed by Ann Lee", {"?D": "Ann Lee"})
        hold(st, "o2", "Ann Lee's mother is Mary Lee", {"?M": "Mary Lee"})
        return st

    def test_a_bridge_value_is_not_the_answer(self):
        st = self.chain()
        self.assertTrue(grounding(st, "Mary Lee")[0])
        self.assertFalse(grounding(st, "Ann Lee")[0])      # the director, not the mother
        self.assertFalse(grounding(st, "e")[0])            # a fragment
        self.assertFalse(grounding(st, "Ann")[0])

    def test_derive_answer_must_be_computed_or_an_option(self):
        st = state_with([ob("o1", "A was released on ?T1"), ob("o2", "B was released on ?T2"),
                         ob("o3", "Compare ?T1 and ?T2", ["o1", "o2"], "derive")], "o3", {})
        q = "Which film came out first, A or B?"
        self.assertFalse(grounding(st, "banana", q)[0])
        self.assertEqual(grounding(st, "B", q)[2], "option")
        self.assertEqual(grounding(st, "A", q, {"answer": "A"})[2], "computed")
        self.assertEqual(grounding(st, "yes", "Are A and B the same?")[2], "yes/no")

    def test_whole_pipeline_does_not_certify_a_stray_derive_answer(self):
        chain = [ob("o1", "The director of Blue River is ?D"), ob("o2", "The mother of ?D is ?M", ["o1"]),
                 ob("o3", "Compare ?D and ?M", ["o1", "o2"], "derive")]
        spec = {"type": "other", "obligations": chain, "answer_id": "o3"}
        rec = {"question": "Which is older, Ann Lee or Mary Lee?", "gold": ["Mary Lee"], "obligations": spec}
        import asyncio
        with mock.patch.object(controller, "retrieve_sync", make_search(FULL)), \
                mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
            r = asyncio.run(controller.solve(FakeLLM(answer="banana"), args(), prov(), rec))
        self.assertFalse(r["certified"])


class NoRetrieveInChain(unittest.TestCase):
    def test_validation_rejects_an_answer_without_retrieve_ancestor(self):
        obj = {"type": "other", "answer_id": "o2", "obligations": [ob("o1", "X is ?A"), ob("o2", "Compare", [], "derive")]}
        self.assertTrue(any("depends on no retrieve" in e for e in validate(obj)))

    def test_audit_never_certifies_nothing(self):
        st = state_with([ob("o1", "X is ?A"), ob("o2", "Compare", [], "derive")], "o2", {})
        c = audit(st)
        self.assertFalse(c.complete)
        self.assertTrue(any("no retrieve" in r for r in c.reasons))


class Sources(unittest.TestCase):
    TEXT1, TEXT2, TEXT3 = "B was directed by Ann Lee", "Ann Lee directed B", "The film B has a director, Ann Lee"

    def test_passages_sharing_a_title_are_one_source(self):
        st = state_with([ob("o1", "The director of B is ?D")], "o1",
                        {self.TEXT1: ["B", "Ann Lee"], self.TEXT2: ["B"]})
        hold(st, "o1", self.TEXT1, {"?D": "Ann Lee"})
        hold(st, "o1", self.TEXT2, {"?D": "Ann Lee"})
        self.assertEqual(st.kappa("o1"), 1)
        self.assertEqual([c["oid"] for c in audit(st).critical], ["o1"])

    def test_an_unknown_passage_is_not_independent_of_a_known_one(self):
        st = state_with([ob("o1", "The director of B is ?D")], "o1", {self.TEXT1: ["B"]})
        hold(st, "o1", self.TEXT1, {"?D": "Ann Lee"})
        hold(st, "o1", self.TEXT3, {"?D": "Ann Lee"})            # title unknown
        c = audit(st)
        self.assertEqual(c.kappa["o1"], 1)
        self.assertTrue(c.critical and not c.complete)
        self.assertEqual(c.d_prov, 0.0)

    def test_only_unknown_still_counts_once(self):
        st = state_with([ob("o1", "The director of B is ?D")], "o1", {})
        hold(st, "o1", self.TEXT3, {"?D": "Ann Lee"})
        self.assertEqual(st.kappa("o1"), 1)


class Aliases(unittest.TestCase):
    def test_distinct_entities_stay_distinct(self):
        for a, b in [("King Henry I", "King Henry V"), ("Prince Henry of Wales", "Prince Henry of Prussia"),
                     ("John Smith (footballer)", "John Smith (politician)"),
                     ("Anna Maria of Saxony", "Anna Maria of Prussia")]:
            self.assertFalse(same_entity(a, b), (a, b))
        self.assertTrue(same_entity("John Smith (footballer)", "John Smith"))
        self.assertTrue(same_entity("Infanta Elena of Spain", "Infanta Elena, Duchess of Lugo"))

    def test_alias_source_counts_for_the_chosen_value(self):
        st = state_with([ob("o1", "The director of B is ?D")], "o1",
                        {"B was directed by Ken Annakin": ["B"], "Annakin directed B": ["Annakin"]})
        hold(st, "o1", "B was directed by Ken Annakin", {"?D": "Ken Annakin"})
        hold(st, "o1", "Annakin directed B", {"?D": "Annakin"})
        self.assertEqual(len(st.live_support("o1")), 2)
        self.assertEqual(st.kappa("o1"), 2)


class TypeGuardByQuestion(unittest.TestCase):
    def test_year_answers_born_in_when_the_question_asks_for_a_year(self):
        obs = [ob("o1", "Ann Lee was born in ?Y")]
        text = "Ann Lee was born in 1950 in Leeds."
        for q, expect in (("In which year was Ann Lee born?", True), ("Where was Ann Lee born?", False)):
            st = state_with(obs, "o1", {text: ["Ann Lee"]}, q)
            it = st.items[st.add_items([{"text": text, "coherence": 1.0}], 1, "q")[0]]
            self.assertEqual(st.validate_match("o1", it, {"?Y": "1950"}, "Ann Lee was born in 1950")[0], expect, q)


class Verified(unittest.TestCase):
    def test_a_rematch_under_another_context_must_be_audited_again(self):
        st = state_with([ob("o1", "The director of B is ?D"), ob("o2", "?D was born on ?T", ["o1"])], "o2",
                        {"B was directed by Ann Lee": ["B"], "Bob Roe was born on 1 May 1900": ["Bob Roe"]})
        hold(st, "o1", "B was directed by Ann Lee", {"?D": "Ann Lee"})
        k = hold(st, "o2", "Bob Roe was born on 1 May 1900", {"?T": "1 May 1900"})
        st.verified[("o2", k)] = True
        st.support["o1"].clear()
        st.context = {kk: v for kk, v in st.context.items() if kk[0] != "o1"}
        hold(st, "o1", "B was directed by Ann Lee", {"?D": "Bob Roe"})
        st.apply_match("o2", k, {"?T": "1 May 1900"})
        self.assertNotEqual(st.verified.get(("o2", k)), True)


class LearnedStatistics(unittest.TestCase):
    def ep(self, W):
        st = EvidenceState(json.loads(json.dumps(CHAIN)), prov())
        return Episode(st, QUESTION, None, args(), W)

    def test_essential_actions_keep_their_prior(self):
        ep = self.ep(Weights(learn_p=True, seen={"FillGap": [0, 40], "Verify": [0, 40], "Diversify": [0, 40]}))
        self.assertAlmostEqual(ep.p("FillGap"), 0.25)
        self.assertAlmostEqual(ep.p("Verify"), 0.25)
        self.assertLess(ep.p("Diversify"), 0.05)               # an optional action that never pays is dropped

    def test_conflict_diversify_learns_separately(self):
        ep = self.ep(Weights(learn_p=True, seen={"Diversify": [0, 40]}))
        a = controller.Action("Diversify", "o1", [], None, 0.8, "tied conflict")
        self.assertGreater(ep.p(a.stat), 0.2)


class SearchFailure(unittest.IsolatedAsyncioTestCase):
    async def test_a_failed_search_does_not_use_up_the_query(self):
        calls = []

        def flaky(url, q):
            calls.append(q)
            if len(calls) == 1:
                raise OSError("boom")
            return FULL.get(q, [])

        rec = {"question": QUESTION, "gold": ["Mary Lee"], "obligations": json.loads(json.dumps(CHAIN))}
        with mock.patch.object(controller, "retrieve_sync", flaky), \
                mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
            r = await controller.solve(FakeLLM(answer="Mary Lee"), args(), prov(), rec)
        self.assertEqual(calls[0], "The director of Blue River is")
        self.assertIn("The director of Blue River is", calls[1:], "the query that failed must be tried again")
        self.assertTrue(r["bindings"].get("?D"))


class CompareDuplicates(unittest.TestCase):
    def test_duplicate_question_strings_are_paired_by_occurrence(self):
        from evidence.compare import paired
        ref = [{"question": "q", "em": 1.0}, {"question": "q", "em": 0.0}]
        oth = [{"question": "q", "em": 0.0}, {"question": "q", "em": 0.0}]
        c = paired(ref, oth)
        self.assertEqual((c["n"], c["wins"], c["losses"]), (2, 0, 1))


if __name__ == "__main__":
    unittest.main()
