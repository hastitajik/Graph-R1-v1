"""
Offline tests for evidence/audit.py and evidence/controller.py (fake LLM + fake search + fake
provenance, no server, no graph).

  python3 -m unittest tests.test_evidence_controller -v      (from the project root)
"""
import json
import re
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

from evidence import controller
from evidence.audit import (audit, build_graph, counterfactual_fragility, graph_critical_cuts, graph_units,
                            pareto_improves, required_retrieve)
from evidence.controller import Action, Episode, Weights, bridge_queries, compact_query, fillgap_queries
from evidence.provenance import ProvenanceIndex, norm
from evidence.state import EvidenceState


def ob(i, text, deps=(), kind="retrieve"):
    return {"id": i, "text": text, "depends_on": list(deps), "kind": kind}


CHAIN = {"type": "compositional", "answer_id": "o2", "obligations": [
    ob("o1", "The director of Blue River is ?D"),
    ob("o2", "The mother of ?D is ?M", ["o1"])]}
QUESTION = "Who is the mother of the director of film Blue River?"

# passage text -> (article title, coherence)
P = {
    "dir_a": ("Blue River was directed by Ann Lee.", "Blue River", 0.9),
    "dir_b": ("Ann Lee directed Blue River.", "Ann Lee (director)", 0.7),
    "dir_x": ("Blue River was directed by Bob Roe.", "Blue River (1999)", 0.8),
    "mom_a": ("Ann Lee's mother is Mary Lee.", "Ann Lee", 0.9),
    "mom_b": ("Mary Lee is Ann Lee's mother.", "Mary Lee", 0.7),
}
TEXT = {k: v[0] for k, v in P.items()}


def prov():
    return ProvenanceIndex({norm(t): {"titles": [title], "chunks": [], "how": "exact"} for t, title, _ in P.values()})


def hit(*keys):
    return [{"<knowledge>": P[k][0], "<coherence>": P[k][2]} for k in keys]


# ------------------------------------------------------------------------- fake LLM
class FakeLLM:
    PAT = [  # (obligation text regex, [(passage regex template, variable)])
        (r"The director of (.+) is \?D$", [(r"{s} was directed by ([^.]+)\.", "?D"), (r"([^.]+) directed {s}\.", "?D")]),
        (r"The mother of (.+) is \?M$", [(r"{s}'s mother is ([^.]+)\.", "?M"), (r"([^.]+) is {s}'s mother\.", "?M")]),
    ]

    def __init__(self, reject=(), reform=None, answer="guess"):
        self.reject, self.reform, self.answer = set(reject), reform or {}, answer
        self.calls = {"match": 0, "verify": 0, "reform": 0, "answer": 0}
        self.informed_prompts, self.match_prompts = [], []
        self.completions = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=self.completions)

    async def create(self, model, messages, **kw):
        sys_, user = messages[0]["content"], messages[1]["content"]
        if sys_.startswith("You check whether"):
            self.calls["match"] += 1
            self.match_prompts.append(user)
            out = {"matches": self.match(user)}
        elif sys_.startswith("You verify ONE fact"):
            self.calls["verify"] += 1
            passage = user.split("Passage: ", 1)[1]
            out = {"supported": False} if passage in self.reject else {"supported": True, "quote": passage.rstrip(".")}
        elif sys_.startswith(("Write 2 different", "Write 2 short", "Write 2 new")):
            self.calls["reform"] += 1
            mode = "paraphrase" if sys_.startswith("Write 2 different") else "counter" if sys_.startswith("Write 2 short") else "informed"
            if mode == "informed":
                self.informed_prompts.append(user)
            out = {"queries": self.reform.get(mode, [])}
        else:
            self.calls["answer"] += 1
            out = {"answer": self.answer}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(out)))])

    def match(self, user):
        obs_part, pas_part = user.split("\n\nPassages:\n")
        obs = dict(re.findall(r"^(o\d+): (.+)$", obs_part.split("Obligations:\n")[1], re.M))
        pas = {int(i): t for i, t in re.findall(r"^\[(\d+)\] (.+)$", pas_part, re.M)}
        out = []
        for oid, text in obs.items():
            for pat, forms in self.PAT:
                m = re.match(pat, text)
                if not m:
                    continue
                for i, p in pas.items():
                    for tpl, var in forms:
                        g = re.search(tpl.format(s=re.escape(m.group(1))), p)
                        if g:
                            out.append({"passage": i, "obligation": oid, "bindings": {var: g.group(1)},
                                        "quote": g.group(0).rstrip(".")})
        return out


def make_search(table, log=None):
    def search(url, query):
        if log is not None:
            log.append(query)
        return table.get(query, [])
    return search


def args(**kw):
    base = dict(model="fake", search_url="fake://", max_steps=12, max_searches=10, abstain_uncertified=False)
    base.update(kw)
    return Namespace(**base)


def fresh(spec=CHAIN):
    return EvidenceState(json.loads(json.dumps(spec)), prov())


def put(state, oid, key, bindings, verified=False):
    k = state.add_items([{"text": TEXT[key], "coherence": P[key][2]}], 1, "q")[0] if prov().key(TEXT[key]) not in state.items \
        else prov().key(TEXT[key])
    state.apply_match(oid, k, bindings)
    if verified:
        state.verified[(oid, k)] = True
    return k


# ------------------------------------------------------------------------- audit
class AuditTests(unittest.TestCase):
    def test_graph_and_state_agree_on_kappa_and_cuts(self):
        st = fresh()
        put(st, "o1", "dir_a", {"?D": "Ann Lee"})
        put(st, "o1", "dir_b", {"?D": "Ann Lee"})
        put(st, "o2", "mom_a", {"?M": "Mary Lee"})
        G = build_graph(st)
        for o in required_retrieve(st):
            self.assertEqual(len(graph_units(G, o)), st.kappa(o), o)
        self.assertEqual(st.kappa("o1"), 2)
        self.assertEqual(graph_critical_cuts(G, required_retrieve(st)), {"o2": "Ann Lee"})
        c = audit(st)
        self.assertEqual([x["oid"] for x in c.critical], ["o2"])
        self.assertTrue(c.connected)
        self.assertEqual(c.kappa, {"o1": 2, "o2": 1})

    def test_certificate_needs_verified_critical_cuts(self):
        st = fresh()
        k1 = put(st, "o1", "dir_a", {"?D": "Ann Lee"})
        k2 = put(st, "o2", "mom_a", {"?M": "Mary Lee"})
        c = audit(st)
        self.assertFalse(c.complete)
        self.assertEqual(c.unaudited, ["o1", "o2"])
        self.assertIn("critical cut not audited", " ".join(c.reasons))
        st.verified[("o1", k1)] = st.verified[("o2", k2)] = True
        c = audit(st)
        self.assertTrue(c.complete, c.reasons)
        self.assertEqual(c.coverage, 1.0)

    def test_uncovered_blocks_certificate_and_is_marked_blocked(self):
        st = fresh()
        c = audit(st)
        self.assertEqual(c.uncovered, ["o1", "o2"])
        self.assertEqual(c.blocked, ["o2"])      # o2 cannot be searched before ?D is known
        self.assertFalse(c.connected)

    def test_counterfactual_fragility(self):
        st = fresh()
        put(st, "o1", "dir_a", {"?D": "Ann Lee"})
        put(st, "o2", "mom_a", {"?M": "Mary Lee"})
        put(st, "o2", "mom_b", {"?M": "Mary Lee"})
        fr = {f["unit"]: f for f in counterfactual_fragility(st)}
        self.assertEqual(set(fr), {"Blue River"})            # o2 has two sources and survives losing either
        self.assertIn("o1", fr["Blue River"]["uncovers"])
        self.assertIn("?D", fr["Blue River"]["flips"])

    def test_conflict_resolution_rule(self):
        st = fresh()
        put(st, "o1", "dir_a", {"?D": "Ann Lee"})
        put(st, "o1", "dir_x", {"?D": "Bob Roe"})
        c = audit(st)
        self.assertEqual(c.unresolved, ["?D"])               # 1 unit vs 1 unit: arbitrary
        self.assertEqual(st.value("?D"), "Ann Lee")          # higher coherence breaks the tie
        put(st, "o1", "dir_b", {"?D": "Ann Lee"})
        c = audit(st)
        self.assertEqual(c.unresolved, [])                   # 2 units vs 1
        self.assertTrue(c.conflicts[0]["resolved"])

    def test_stale_downstream_support_is_not_live(self):
        st = fresh()
        put(st, "o1", "dir_a", {"?D": "Ann Lee"})
        put(st, "o2", "mom_a", {"?M": "Mary Lee"})
        self.assertTrue(st.covered("o2"))
        put(st, "o1", "dir_b", {"?D": "Bob Roe"})   # two units now make Bob Roe the best-supported director
        put(st, "o1", "dir_x", {"?D": "Bob Roe"})
        self.assertEqual(st.value("?D"), "Bob Roe")
        self.assertFalse(st.covered("o2"), "mother-of-Ann-Lee evidence must not support mother-of-Bob-Roe")
        self.assertTrue(st.ready("o2"))
        self.assertIn("o2", audit(st).uncovered)

    def test_recheck_after_value_change(self):
        st = fresh()
        k = put(st, "o1", "dir_a", {"?D": "Ann Lee"})
        st.mark_checked([k], ["o2"])
        self.assertEqual(st.unchecked("o2"), [])
        put(st, "o1", "dir_b", {"?D": "Bob Roe"})
        put(st, "o1", "dir_x", {"?D": "Bob Roe"})
        self.assertIn(k, st.unchecked("o2"), "after ?D changed the pair must be shown to the matcher again")

    def test_pareto(self):
        self.assertTrue(pareto_improves((0.5, 0, 0, 0), (1.0, 0, 0, 0)))
        self.assertFalse(pareto_improves((0.5, 0, 0, 0), (0.5, 0, 0, 0)))
        self.assertFalse(pareto_improves((0.5, 0.5, 0, 0), (1.0, 0.0, 0, 0)))   # a gain that costs something else


# ------------------------------------------------------------------------- query builders
class QueryTests(unittest.TestCase):
    def test_compact(self):
        self.assertEqual(compact_query("The director of Lapland Calendar is"), "director Lapland Calendar")

    def test_fillgap_gives_distinct_forms_and_skips_tried(self):
        st = fresh()
        self.assertEqual(fillgap_queries(st, "o1"), ["The director of Blue River is", "director Blue River"])
        st.note_query("o1", "The director of Blue River is")
        self.assertEqual(fillgap_queries(st, "o1"), ["director Blue River"])

    def test_bridge_composes_the_chain(self):
        st = fresh()
        qs = bridge_queries(st, "o2", QUESTION)
        self.assertEqual(qs, ["The mother of the director of Blue River is", QUESTION])


# ------------------------------------------------------------------------- controller
FULL = {  # a world where every query a good controller sends finds something
    "The director of Blue River is": hit("dir_a"),
    "The mother of Ann Lee is": hit("mom_a"),
    "The director of Blue River is Ann Lee": hit("dir_b"),
    "The mother of Ann Lee is Mary Lee": hit("mom_b"),
}


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def run_ep(self, table, llm=None, spec=CHAIN, W=None, log=None, **kw):
        llm = llm or FakeLLM(answer="Mary Lee")
        rec = {"question": QUESTION, "gold": ["Mary Lee"], "obligations": json.loads(json.dumps(spec))}
        with mock.patch.object(controller, "retrieve_sync", make_search(table, log)), \
                mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
            r = await controller.solve(llm, args(**kw), prov(), rec, W)
        return r, llm

    async def test_happy_path_reaches_a_certificate(self):
        log = []
        r, llm = await self.run_ep(FULL, log=log)
        self.assertTrue(r["certified"], r["certificate"]["reasons"])
        self.assertEqual(r["stop"], "certified")
        self.assertEqual(r["bindings"], {"?D": "Ann Lee", "?M": "Mary Lee"})
        self.assertEqual(r["em"], 1.0)
        kinds = [s["action"] for s in r["steps"]]
        self.assertEqual(kinds[:2], ["FillGap", "FillGap"])
        self.assertEqual(sum(k == "Verify" for k in kinds), 0 if r["certificate"]["r_struct"] == 1 and all(
            v >= 2 for v in r["certificate"]["kappa"].values()) else kinds.count("Verify"))
        self.assertTrue(all(v >= 1 for v in r["certificate"]["kappa"].values()))
        self.assertLessEqual(r["searches"], 10)
        self.assertEqual(r["pathologies"].get("premature_confidence"), None)

    async def test_diversify_buys_a_second_independent_source(self):
        r, _ = await self.run_ep(FULL)
        diversify = [s for s in r["steps"] if s["action"] == "Diversify" and s["improved"]]
        self.assertTrue(diversify, [s["action"] for s in r["steps"]])
        self.assertTrue(any(v == 2 for v in r["certificate"]["kappa"].values()), r["certificate"]["kappa"])
        self.assertGreater(r["certificate"]["d_prov"], 0)

    async def test_tied_conflict_is_resolved_by_more_evidence(self):
        table = dict(FULL)
        table["The director of Blue River is"] = hit("dir_a", "dir_x")    # two articles, two different directors
        r, _ = await self.run_ep(table)
        self.assertTrue(r["certified"], r["certificate"]["reasons"])
        self.assertEqual(r["bindings"]["?D"], "Ann Lee")
        self.assertEqual(r["conflicts"]["?D"], ["Ann Lee", "Bob Roe"])        # still visible, but resolved 2 vs 1
        first = next(s for s in r["steps"] if s["action"] in ("Diversify", "Challenge") and s["oid"] == "o1")
        self.assertEqual(first["why"], "tied conflict")

    async def test_failed_verify_removes_support_and_reopens_the_gap(self):
        llm = FakeLLM(reject=[TEXT["dir_a"]], reform={"paraphrase": ["who directed Blue River"]}, answer="Mary Lee")
        table = {k: v for k, v in FULL.items() if k != "The director of Blue River is Ann Lee"}  # no 2nd source yet
        table["who directed Blue River"] = hit("dir_b")      # but a paraphrase finds another article
        r, _ = await self.run_ep(table, llm=llm)
        rej = [s for s in r["steps"] if s.get("verify") == "rejected"]
        self.assertTrue(rej, "Verify must reject the unsupported passage")
        self.assertEqual(rej[0]["oid"], "o1")
        # the gap is reopened; the plain query was already tried, so a different one is used
        later = [s for s in r["steps"] if s["step"] > rej[0]["step"] and s["oid"] == "o1" and s["queries"]]
        self.assertTrue(later)
        self.assertNotIn("The director of Blue River is", later[0]["queries"])
        self.assertEqual(r["bindings"].get("?D"), "Ann Lee")
        self.assertEqual(r["coverage"], 1.0)

    async def test_failed_verify_without_alternative_ends_uncertified_not_crashed(self):
        llm = FakeLLM(reject=[TEXT["dir_a"]], answer="guess")
        table = {k: v for k, v in FULL.items() if k != "The director of Blue River is Ann Lee"}
        r, _ = await self.run_ep(table, llm=llm)
        self.assertFalse(r["certified"])
        self.assertIn("o1", r["uncovered"])
        self.assertTrue(r["answered_from_incomplete"])

    async def test_bridge_finds_the_chain_when_plain_queries_fail(self):
        table = {"The mother of the director of Blue River is": hit("dir_a", "mom_a"),
                 "The mother of Ann Lee is": hit("mom_a")}
        r, _ = await self.run_ep(table, llm=FakeLLM(answer="Mary Lee"), max_steps=12)
        kinds = [s["action"] for s in r["steps"]]
        self.assertIn("Bridge", kinds)
        self.assertEqual(r["bindings"].get("?M"), "Mary Lee")
        self.assertEqual(r["coverage"], 1.0)
        bridge = next(s for s in r["steps"] if s["action"] == "Bridge")
        self.assertEqual(bridge["oid"], "o2")
        first_o1 = next(i for i, s in enumerate(r["steps"]) if s["oid"] == "o1")
        self.assertLess(first_o1, kinds.index("Bridge"), "plain FillGap of the upstream hop comes first")

    async def test_llm_paraphrase_is_used_only_after_plain_forms_are_exhausted(self):
        llm = FakeLLM(reform={"paraphrase": ["who directed the film Blue River"]}, answer="Mary Lee")
        table = dict(FULL)
        table["The director of Blue River is"] = []
        table["director Blue River"] = []
        table["who directed the film Blue River"] = hit("dir_a")
        r, llm = await self.run_ep(table, llm=llm)
        self.assertEqual(llm.calls["reform"] >= 1, True)
        self.assertEqual(r["bindings"].get("?D"), "Ann Lee")
        used = [s for s in r["steps"] if "who directed the film Blue River" in s["queries"]]
        self.assertTrue(used)
        plain = [s for s in r["steps"] if s["queries"][:1] == ["The director of Blue River is"]]
        self.assertLess(plain[0]["step"], used[0]["step"])

    async def test_nothing_found_stops_cheaply_and_flags_the_guess(self):
        log = []
        r, _ = await self.run_ep({}, log=log, max_searches=10)
        self.assertFalse(r["certified"])
        self.assertIn(r["stop"], {"stalled", "blocked", "no_positive_vne", "budget"})
        self.assertLessEqual(r["searches"], 10)
        self.assertEqual(r["searches"], len(log))
        self.assertTrue(r["answered_from_incomplete"])
        self.assertEqual(r["pathologies"].get("premature_confidence"), 1)
        self.assertEqual(r["uncovered"], ["o1", "o2"])

    async def test_abstain_when_not_certified(self):
        r, llm = await self.run_ep({}, abstain_uncertified=True)
        self.assertTrue(r["abstained"])
        self.assertEqual(r["answer"], "")
        self.assertEqual(llm.calls["answer"], 0)

    async def test_search_budget_is_a_hard_limit(self):
        log = []
        r, _ = await self.run_ep(FULL, log=log, max_searches=2)
        self.assertLessEqual(len(log), 2)
        self.assertEqual(r["stop"], "budget")

    async def test_search_errors_do_not_crash_the_episode(self):
        def boom(url, q):
            raise OSError("down")
        llm = FakeLLM()
        rec = {"question": QUESTION, "gold": ["Mary Lee"], "obligations": json.loads(json.dumps(CHAIN))}
        with mock.patch.object(controller, "retrieve_sync", boom), \
                mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
            r = await controller.solve(llm, args(), prov(), rec)
        self.assertFalse(r["certified"])
        self.assertTrue(any(s.get("errors") for s in r["steps"]))


class VerifyEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def test_two_passages_of_one_article_are_audited_once_without_looping(self):
        # both passages come from the same article (same unit): kappa = 1, so the cut is critical
        extra = "Blue River was directed by Ann Lee in 1987."
        data = {norm(t): {"titles": [title], "chunks": [], "how": "exact"} for t, title, _ in P.values()}
        data[norm(extra)] = {"titles": ["Blue River"], "chunks": [], "how": "exact"}
        st = EvidenceState(json.loads(json.dumps(CHAIN)), ProvenanceIndex(data))
        k1 = st.add_items([{"text": TEXT["dir_a"], "coherence": 0.9}, {"text": extra, "coherence": 0.5}], 1, "q")
        st.apply_match("o1", k1[0], {"?D": "Ann Lee"})
        st.apply_match("o1", k1[1], {"?D": "Ann Lee"})
        self.assertEqual(st.kappa("o1"), 1)
        ep = Episode(st, QUESTION, FakeLLM(), args(), Weights())
        self.assertEqual(await ep.verify("o1"), "supported")
        self.assertTrue(audit(st).critical[0]["verified"])
        self.assertEqual(audit(st).unaudited, [])

    async def test_verify_without_live_support_is_banned_not_repeated(self):
        ep = Episode(fresh(), QUESTION, FakeLLM(), args(), Weights())
        self.assertEqual(await ep.verify("o1"), "skipped")
        self.assertIn(("Verify", "o1"), ep.banned)


class VneTests(unittest.TestCase):
    def episode(self):
        return Episode(fresh(), QUESTION, None, args(), Weights())

    def test_price_and_probability_update(self):
        ep = self.episode()
        a = ep.price(Action("FillGap", "o1", ["q1", "q2"], gain=1.0))
        self.assertAlmostEqual(a.p, 0.25)
        self.assertAlmostEqual(a.cost, 0.10)
        self.assertAlmostEqual(a.vne, 0.25 * 1.0 - 0.10)
        ep.stats["FillGap"] = [3, 3]
        self.assertAlmostEqual(ep.p("FillGap"), 4 / 7)
        ep.stats["Diversify"] = [0, 6]
        self.assertAlmostEqual(ep.p("Diversify"), 1 / 10)

    def test_failing_action_becomes_not_worth_its_cost(self):
        ep = self.episode()
        d = Action("Diversify", "o1", ["q"], gain=0.3)
        self.assertGreater(ep.price(d).vne, 0)
        ep.stats["Diversify"] = [0, 3]
        self.assertLessEqual(ep.price(d).vne, 0)

    def test_verify_has_no_search_cost(self):
        a = self.episode().price(Action("Verify", "o1", [], gain=0.15))
        self.assertEqual(a.n_search, 0)
        self.assertAlmostEqual(a.cost, 0.02)

    def test_confirmation_bias_flag_picks_the_weakest_and_clears_after_challenge(self):
        ep = self.episode()
        put(ep.state, "o1", "dir_a", {"?D": "Ann Lee"})
        put(ep.state, "o2", "mom_b", {"?M": "Mary Lee"})    # coherence 0.7 < 0.9
        flags = ep.monitor(audit(ep.state))
        self.assertEqual(flags["confirmation_bias"], ["o2"])
        ep.challenged = True
        self.assertEqual(ep.monitor(audit(ep.state))["confirmation_bias"], [])

    def test_retrieval_loop_flag(self):
        ep = self.episode()
        ep.stall = 3
        self.assertTrue(ep.monitor(audit(ep.state))["retrieval_loop"])


if __name__ == "__main__":
    unittest.main()
