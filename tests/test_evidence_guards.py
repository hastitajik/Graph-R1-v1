"""Regression tests for wrong evidence seen in real 2WikiMultiHopQA traces (no network, no LLM).

Run:  python3 -m unittest tests.test_evidence_guards -v
"""
import asyncio
import json
import types
import unittest

from evidence import controller as C
from evidence import run_mvp
from evidence.audit import audit
from evidence.state import EvidenceState, Item


class Prov:
    def key(self, t):
        return "".join(c for c in t.lower() if c.isalnum())

    def titles(self, t):
        return []


def mk(obs, answer_id):
    return EvidenceState({"obligations": obs, "answer_id": answer_id}, Prov())


def hold(st, key, title, text, oid=None, bind=None):
    it = Item(key, text, 0.9, [title], 1, "q")
    st.items[key] = it
    if oid:
        st.apply_match(oid, key, bind)
    return it


class FakeClient:
    """Returns a fixed matcher reply, whatever the prompt is."""

    def __init__(self, matches):
        self.reply = json.dumps({"matches": matches})
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        msg = types.SimpleNamespace(content=self.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class TypeGuard(unittest.TestCase):
    def test_year_is_not_a_place(self):                      # question 4: "died in ?P" = 1515
        st = mk([{"id": "o1", "text": "The father of B is ?F", "depends_on": [], "kind": "retrieve"},
                 {"id": "o2", "text": "?F died in ?P", "depends_on": ["o1"], "kind": "retrieve"}], "o2")
        hold(st, "a", "B", "Andrzej Kościelecki is the father of B", "o1", {"?F": "Andrzej Kościelecki"})
        it = hold(st, "b", "Andrzej Kościelecki", "Andrzej Kościelecki died in 1515 and was buried")
        ok, why = st.validate_match("o2", it, {"?P": "1515"}, "Andrzej Kościelecki died in 1515")
        self.assertFalse(ok)
        self.assertIn("type mismatch", why)

    def test_place_with_a_date_is_fine(self):               # question 9: 'Monfort, October 9, 1326'
        st = mk([{"id": "o1", "text": "X died in ?L", "depends_on": [], "kind": "retrieve"}], "o1")
        it = hold(st, "b", "X", "X died in Monfort, October 9, 1326")
        self.assertTrue(st.validate_match("o1", it, {"?L": "Monfort, October 9, 1326"}, "X died in Monfort")[0])

    def test_date_variable_accepts_a_bare_year(self):       # Exit Smiling: "released on" = 1926
        st = mk([{"id": "o1", "text": "Exit Smiling was released on ?T1", "depends_on": [], "kind": "retrieve"}], "o1")
        it = hold(st, "b", "Exit Smiling", "Exit Smiling is a 1926 comedy film")
        self.assertTrue(st.validate_match("o1", it, {"?T1": "1926"}, "Exit Smiling is a 1926 comedy film")[0])


class SubjectGuard(unittest.TestCase):
    def setUp(self):
        self.st = mk([{"id": "o1", "text": "The director of F is ?D", "depends_on": [], "kind": "retrieve"},
                      {"id": "o2", "text": "?D has nationality ?N", "depends_on": ["o1"], "kind": "retrieve"}], "o2")

    def test_passage_about_someone_else_is_rejected(self):  # Bryllupsfotografen / Tuva Novotny
        hold(self.st, "a", "F", "Johan Bergenstråhle directed F", "o1", {"?D": "Johan Bergenstråhle"})
        it = hold(self.st, "t", "Tuva Novotny", "Tuva Novotny is a Swedish actress and director")
        ok, why = self.st.validate_match("o2", it, {"?N": "Swedish"}, "Tuva Novotny is a Swedish actress")
        self.assertFalse(ok)
        self.assertIn("subject mismatch", why)

    def test_name_variants_are_not_rejected(self):          # 'Robert N. Bradbury' vs article 'Robert North Bradbury'
        hold(self.st, "a", "F", "Robert N. Bradbury directed F", "o1", {"?D": "Robert N. Bradbury"})
        it = hold(self.st, "t", "Robert North Bradbury", "He was an American film director")
        self.assertTrue(self.st.validate_match("o2", it, {"?N": "American"}, "He was an American film director")[0])

    def test_pronoun_passage_with_matching_title(self):     # 'He was half-brother to ...' under the right article
        st = mk([{"id": "o1", "text": "The father of J is ?F", "depends_on": [], "kind": "retrieve"},
                 {"id": "o2", "text": "The brother of ?F is ?U", "depends_on": ["o1"], "kind": "retrieve"}], "o2")
        hold(st, "a", "J", "Gerald FitzGerald, 3rd Earl of Desmond is the father of J", "o1",
             {"?F": "Gerald FitzGerald, 3rd Earl of Desmond"})
        it = hold(st, "t", "Gerald FitzGerald, 3rd Earl of Desmond", "He was half-brother to Maurice FitzGerald, 2nd Earl")
        self.assertTrue(st.validate_match("o2", it, {"?U": "Maurice FitzGerald, 2nd Earl"},
                                          "He was half-brother to Maurice FitzGerald")[0])


class Retarget(unittest.TestCase):
    def test_wrong_obligation_id_is_corrected(self):        # questions 9 and 17
        st = mk([{"id": "o1", "text": "The director of X is ?D1", "depends_on": [], "kind": "retrieve"},
                 {"id": "o2", "text": "The director of Y is ?D2", "depends_on": [], "kind": "retrieve"},
                 {"id": "o3", "text": "?D1 was born on ?T1", "depends_on": ["o1"], "kind": "retrieve"},
                 {"id": "o4", "text": "?D2 was born on ?T2", "depends_on": ["o2"], "kind": "retrieve"}], "o4")
        hold(st, "a", "X", "A directed X", "o1", {"?D1": "A"})
        hold(st, "b", "Y", "Gustaf Molander directed Y", "o2", {"?D2": "Gustaf Molander"})
        txt = "Gustaf Molander( 18 November 1888 – 19 June 1973) was a Swedish director"
        st.items["m"] = Item("m", txt, 0.9, ["Gustaf Molander"], 2, "q")
        client = FakeClient([{"passage": 0, "obligation": "o2", "bindings": {"?T2": "18 November 1888"},
                              "quote": "Gustaf Molander( 18 November 1888 – 19 June 1973)"}])
        good, _, _, _ = asyncio.run(run_mvp.map_passages(client, "m", st, ["m"], ["o3", "o4"]))
        self.assertEqual([m["obligation"] for m, _ in good], ["o4"])


class DiversifyScope(unittest.TestCase):
    def test_diversify_cannot_cover_another_obligation(self):   # question 19: 'graduated from Apley Park'
        st = mk([{"id": "o1", "text": "The father of H is ?F", "depends_on": [], "kind": "retrieve"},
                 {"id": "o2", "text": "?F graduated from ?S", "depends_on": ["o1"], "kind": "retrieve"}], "o2")
        hold(st, "a", "Henry Whitmore", "Thomas Whitmore is the father of H", "o1", {"?F": "Thomas Whitmore"})
        orig = C.retrieve_sync
        C.retrieve_sync = lambda url, q, **k: [{"<knowledge>": "Thomas Whitmore lived at Apley Park near Bridgnorth",
                                                "<coherence>": 0.8}]
        try:
            client = FakeClient([{"passage": 0, "obligation": "o2", "bindings": {"?S": "Apley Park"},
                                  "quote": "Thomas Whitmore lived at Apley Park"}])
            args = types.SimpleNamespace(model="m", search_url="x", max_searches=10, max_steps=5)
            ep = C.Episode(st, "q", client, args, C.Weights())
            act = ep.price(C.Action("Diversify", "o1", ["Thomas Whitmore is the father of H"], None, 0.3,
                                    "single source", limit=1))
            asyncio.run(ep.execute(act, 1, audit(st)))
        finally:
            C.retrieve_sync = orig
        self.assertFalse(st.covered("o2"))


class SelfReference(unittest.TestCase):
    def test_uncle_cannot_be_the_subject(self):             # question 22: 'uncle of James' bound to James
        st = mk([{"id": "o1", "text": "The father of James Fitzgerald, 6th Earl of Desmond is ?F",
                  "depends_on": [], "kind": "retrieve"},
                 {"id": "o2", "text": "The brother of ?F is ?U", "depends_on": ["o1"], "kind": "retrieve"}], "o2")
        st.question = "Who is the uncle of James Fitzgerald, 6Th Earl Of Desmond?"
        hold(st, "a", "James", "Gerald is the father of James", "o1", {"?F": "Gerald FitzGerald"})
        it = hold(st, "b", "Gerald FitzGerald", "Gerald FitzGerald, brother James FitzGerald, 6th Earl of Desmond")
        ok, why = st.validate_match("o2", it, {"?U": "James FitzGerald, 6th Earl of Desmond"},
                                    "brother James FitzGerald, 6th Earl of Desmond")
        self.assertFalse(ok)
        self.assertIn("self-reference", why)

    def test_other_person_is_fine(self):
        st = mk([{"id": "o1", "text": "The father of B is ?F", "depends_on": [], "kind": "retrieve"}], "o1")
        st.question = "Who is the father of Beata Laska?"
        it = hold(st, "a", "Beata", "Andrzej Koscielecki is the father of Beata Laska")
        ok, _ = st.validate_match("o1", it, {"?F": "Andrzej Koscielecki"}, "Andrzej Koscielecki is the father")
        self.assertTrue(ok)


class Aliases(unittest.TestCase):
    def test_same_person_two_spellings_is_no_conflict(self):    # question 29: Infanta Elena
        from evidence.state import same_entity
        self.assertTrue(same_entity("Infanta Elena of Spain", "Infanta Elena, Duchess of Lugo"))
        self.assertFalse(same_entity("Sir Paul Gore, 1st Baronet", "Sir Paul Rycaut"))
        self.assertFalse(same_entity("Ralph Gore, 2nd Baronet", "Ralph Gore, 3rd Baronet"))
        self.assertFalse(same_entity("Henry II", "Henry III"))
        self.assertFalse(same_entity("1 May 1900", "2 May 1900"))
        st = mk([{"id": "o1", "text": "The mother of F is ?M", "depends_on": [], "kind": "retrieve"}], "o1")
        hold(st, "a", "F", "Infanta Elena of Spain is the mother of F", "o1", {"?M": "Infanta Elena of Spain"})
        hold(st, "b", "Elena", "Infanta Elena, Duchess of Lugo is the mother of F", "o1",
             {"?M": "Infanta Elena, Duchess of Lugo"})
        self.assertEqual(st.conflicts(), {})


class AnswerGrounded(unittest.TestCase):
    def test_not_found_and_stray_answers_are_not_certified(self):
        st = mk([{"id": "o1", "text": "X studied at ?S", "depends_on": [], "kind": "retrieve"}], "o1")
        hold(st, "a", "X", "X studied at Eton College (1796) and Christ Church", "o1",
             {"?S": "Eton College (1796) and Christ Church"})
        self.assertFalse(C.answer_grounded(st, "NOT FOUND")[0])
        self.assertFalse(C.answer_grounded(st, "")[0])
        self.assertFalse(C.answer_grounded(st, "Harvard")[0])
        self.assertTrue(C.answer_grounded(st, "Eton College (1796) and Christ Church")[0])


class Ablations(unittest.TestCase):
    def test_disabled_kinds_are_never_proposed(self):
        st = mk([{"id": "o1", "text": "The director of F is ?D", "depends_on": [], "kind": "retrieve"}], "o1")
        hold(st, "a", "F", "Anna Berg directed F", "o1", {"?D": "Anna Berg"})
        args = types.SimpleNamespace(model="m", search_url="x", max_searches=10, max_steps=5)
        ep = C.Episode(st, "q", None, args, C.Weights(disabled=("Diversify",)))
        kinds = {a.kind for a in ep.propose(audit(st), ep.monitor(audit(st)))}
        self.assertNotIn("Diversify", kinds)
        self.assertIn("Verify", kinds)

    def test_learned_p_drops_after_failures(self):
        st = mk([{"id": "o1", "text": "X is ?A", "depends_on": [], "kind": "retrieve"}], "o1")
        args = types.SimpleNamespace(model="m", search_url="x", max_searches=10, max_steps=5)
        W = C.Weights(learn_p=True, seen={"Diversify": [0, 40]})
        self.assertLess(C.Episode(st, "q", None, args, W).p("Diversify"), 0.05)
        self.assertEqual(C.Episode(st, "q", None, args, C.Weights()).p("Diversify"), 0.25)


class Baseline(unittest.TestCase):
    def test_agent_loop_searches_then_answers(self):
        from evidence import baseline as B
        replies = iter(["<think>x</think>\n<query>\nDirector of F\n", "<think>y</think>\n<answer>\nAnna Berg\n</answer>"])

        async def create(**kw):
            msg = types.SimpleNamespace(content=next(replies))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        orig = B.retrieve_sync
        B.retrieve_sync = lambda url, q, **k: [{"<knowledge>": "Anna Berg directed F", "<coherence>": 0.9}]
        try:
            args = types.SimpleNamespace(model="m", search_url="x", max_turns=4, info_chars=1000, short=False)
            r = asyncio.run(B.run_one(client, args, {"question": "Who directed F?", "gold": ["Anna Berg"]}))
        finally:
            B.retrieve_sync = orig
        self.assertEqual((r["answer"], r["em"], r["searches"], r["turns"]), ("Anna Berg", 1.0, 1, 2))


class ExportGraph(unittest.TestCase):
    def test_graph_has_dependency_and_support_edges(self):
        from evidence.export_graph import build
        ob = {"obligations": {"obligations": [
            {"id": "o1", "text": "The director of A is ?D", "kind": "retrieve", "depends_on": []},
            {"id": "o2", "text": "?D was born on ?T", "kind": "retrieve", "depends_on": ["o1"]}], "answer_id": "o2"}}
        t = {"question": "q", "answer": "1950", "certified": True, "certificate": {"unaudited": []},
             "obligations": [{"id": "o1", "text": "The director of A is Neri", "covered": True, "titles": ["A (film)"]},
                             {"id": "o2", "text": "Neri was born on 1950", "covered": True, "titles": ["Neri Parenti"]}]}
        G = build(t, ob)
        self.assertTrue(G.has_edge("o1", "o2"))
        self.assertTrue(G.has_edge("unit:Neri Parenti", "o2"))
        self.assertEqual(G.number_of_nodes(), 4)


class Round2(unittest.TestCase):
    def test_two_word_name_needs_both_words(self):          # William Clemens vs William I, Count of Nassau-Dillenburg
        from evidence.state import subject_in
        self.assertFalse(subject_in("William Clemens", ["William I, Count of Nassau-Dillenburg"],
                                    "William I, Count of Nassau-Dillenburg was born in Dillenburg"))
        self.assertTrue(subject_in("William Clemens", ["William Clemens"], "William Clemens was born in Saginaw"))

    def test_entity_is_not_its_own_father(self):            # Lewis Watson is the father of Lewis Watson
        st = mk([{"id": "o1", "text": "The father of E is ?F", "depends_on": [], "kind": "retrieve"},
                 {"id": "o2", "text": "The father of ?F is ?G", "depends_on": ["o1"], "kind": "retrieve"}], "o2")
        hold(st, "a", "E", "Lewis Watson is the father of E", "o1", {"?F": "Lewis Watson"})
        it = hold(st, "b", "Lewis Watson", "Lewis Watson, 1st Earl, son of Lewis Watson")
        ok, why = st.validate_match("o2", it, {"?G": "Lewis Watson"}, "Lewis Watson, 1st Earl, son of Lewis Watson")
        self.assertFalse(ok)
        self.assertIn("self-reference", why)


if __name__ == "__main__":
    unittest.main()
