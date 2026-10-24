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


if __name__ == "__main__":
    unittest.main()
