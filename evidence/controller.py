"""
Decision-Time Controller (training-free): decides WHICH evidence to buy next, and WHEN the
certificate of audit.py is good enough to answer. The LLM and the retrieval API stay frozen.

Loop (one question):
    sweep    show held-but-unchecked passages to the matcher (obligations that just became ready)
    audit    certificate + state vector (audit.py)
    monitor  reasoning-pathology checks (below)
    propose  every applicable action, each with a gain, a cost and a success probability
    choose   the action with the best Value of Next Evidence, VNE = p * gain - cost
    stop     no action with VNE > 0  ->  Answer (certified if the certificate is complete)

Actions
    FillGap   obligation uncovered and ready      search its fact (plain query; then compact form,
                                                  entity-only form, then LLM paraphrases)
    Bridge    obligation blocked because the      search the chain as ONE query ("mother of the
              upstream hop cannot be found        director of X") and the whole question
    Diversify covered with < 2 independent        search for a SECOND article stating the fact
              known sources (or a tied conflict)
    Challenge conflict, or confirmation bias      search for passages that contradict the chosen value
    Verify    critical cut (kappa = 1)            re-check the sole passage with a stricter prompt;
                                                  no search; removes the support if it does not hold
    Answer    certificate complete, VNE <= 0      (also the fall-back when nothing is worth buying)

VNE
    gain   = Weights.<tier> (x1 + number of blocked descendants for gap actions)
    cost   = c_search * #queries + c_llm * #LLM-query calls
    p      = (successes + prior_succ) / (attempts + prior_att), per action kind and per question;
             a step succeeds when the audit vector (C, R, D, -X) Pareto-improves.
    The tier weights are hyper-parameters (Weights, all on the command line). They only fix an
    ORDER of importance (gap > conflict > challenge > fragile > unverified); the Pareto test is
    what decides whether a step made progress. Ablate them.

Reasoning-pathology monitor (flags go to the trace, the first three also act)
    retrieval_loop       `stall_limit` consecutive search steps with no new item and no Pareto gain
                         -> stop searching
    over_retrieval       an obligation's last `novelty_limit` searches returned nothing new
                         -> no more searches for it
    confirmation_bias    every covered obligation rests on one known article (D = 0) and nothing was
                         challenged -> Challenge becomes allowed for the weakest obligation
    premature_confidence the answer was produced while a required obligation was uncovered
    ungrounded_answer    the answer of a retrieve-type question appears in none of the evidence

Run (from the project root):
  python3 -m evidence.controller --obligations obligations_dev30.jsonl --n 5 --search-url ... --out trace_ctrl.jsonl
"""
import argparse
import asyncio
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace

from openai import AsyncOpenAI

from evidence.audit import audit, pareto_improves, required_retrieve
from evidence.common import LLM_BASE_URL, LLM_MODEL, SEARCH_URL
from evidence.metrics import em_f1
from evidence.obligations import extract_json
from evidence.provenance import ProvenanceIndex, norm
from evidence.run_mvp import (MAP_BATCH, PASSAGE_CHARS, answer_question, make_query, map_and_apply,
                              parse_results, retrieve_sync)
from evidence.state import MIN_QUOTE, UNKNOWN, EvidenceState

SEARCH_KINDS = ("FillGap", "Bridge", "Diversify", "Challenge")
KINDS = SEARCH_KINDS + ("Verify",)
STOP_WORDS = {"the", "a", "an", "of", "is", "was", "were", "are", "be", "been", "in", "on", "at", "by",
              "to", "for", "did", "does", "do", "has", "have", "had", "that", "which", "who"}


@dataclass
class Weights:
    gap: float = 1.0          # an uncovered required obligation
    conflict: float = 0.8     # an unresolved conflict
    challenge: float = 0.2    # confirmation-bias challenge (no conflict known)
    fragile: float = 0.3      # a single-source (kappa = 1) obligation
    unverified: float = 0.15  # a critical cut not re-checked
    c_search: float = 0.05    # one retrieval query
    c_llm: float = 0.02       # one LLM call that writes queries
    prior_succ: float = 1.0   # pseudo-counts of the success probability: p0 = 1/4
    prior_att: float = 4.0
    stall_limit: int = 3
    novelty_limit: int = 2
    max_queries: int = 2      # queries per search action


# ------------------------------------------------------------------ query builders
def compact_query(text: str) -> str:
    """Keywords only: 'The director of Lapland Calendar is' -> 'director Lapland Calendar'."""
    toks = [t for t in text.split() if t.lower().strip(",.?") not in STOP_WORDS]
    return " ".join(toks).strip(" ,.")


def _unique(qs, tried):
    out, seen = [], set(tried)
    for q in qs:
        q = (q or "").strip()
        if q and norm(q) not in seen:
            seen.add(norm(q))
            out.append(q)
    return out


def fillgap_queries(state, oid):
    base = make_query(state.instantiate(oid))
    deps = [state.value(v) for v in sorted(state.dep_vars(oid)) if state.value(v)]
    return _unique([base, compact_query(base), " ".join(deps)], state.tried(oid))


def bridge_queries(state, oid, question):
    qs = []
    deps = state.obs[oid]["depends_on"]
    dv = sorted(state.dep_vars(oid))
    if len(deps) == 1 and len(dv) == 1 and state.is_retrieve(deps[0]):
        phrase = re.sub(r"^the\s+", "", make_query(state.instantiate(deps[0])), flags=re.I)
        phrase = re.sub(r"\s+(is|was|were|are)$", "", phrase)
        composed = re.sub(re.escape(dv[0]) + r"(?!\w)", "the " + phrase, state.obs[oid]["text"])
        qs.append(make_query(composed))
    qs.append(question)
    return _unique(qs, state.tried(oid))


def diversify_queries(state, oid):
    inst = state.instantiate(oid)
    return _unique([inst, compact_query(inst)], state.tried(oid))


def challenge_queries(state, oid):
    vals = [state.value(v) for v in state.own_vars(oid) if state.value(v)]
    return _unique(vals[:2], state.tried(oid))


# ------------------------------------------------------------------ LLM helpers
REFORM_SYSTEM = {
    "paraphrase": "Write 2 different short search queries that would retrieve a passage stating the "
                  "fact below, with different wording (synonyms, other relation names). Keep entity "
                  'names exactly as written. Return ONLY JSON: {"queries": ["...", "..."]}',
    "counter": "Write 2 short search queries that would retrieve passages that CONTRADICT the fact "
               "below or give a DIFFERENT value for it (other candidates, other sources). Keep entity "
               'names exactly as written. Return ONLY JSON: {"queries": ["...", "..."]}',
}

VERIFY_SYSTEM = """You verify ONE fact against ONE passage. Use only the passage, never outside knowledge.
Answer supported=true only if the passage EXPLICITLY states the fact; mentioning the entities, or a \
fact that is merely implied, is not enough. If true, copy the exact words that state it.
Return ONLY JSON: {"supported": true, "quote": "exact words"} or {"supported": false}"""


async def llm_json(client, model, system, user):
    err = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return extract_json(resp.choices[0].message.content or ""), None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
    return None, err


async def llm_queries(client, model, fact, mode):
    obj, err = await llm_json(client, model, REFORM_SYSTEM[mode], f"Fact: {fact}")
    qs = (obj or {}).get("queries") if isinstance(obj, dict) else None
    return [str(q) for q in qs if isinstance(q, str)][:2] if isinstance(qs, list) else []


# ------------------------------------------------------------------ the controller
@dataclass
class Action:
    kind: str
    oid: str
    det: list                 # deterministic queries (already filtered against what was tried)
    llm_mode: str = None      # "paraphrase" | "counter": ask the LLM for more queries when executed
    gain: float = 0.0
    why: str = ""
    p: float = 0.0
    cost: float = 0.0
    vne: float = 0.0
    n_search: int = 0
    limit: int = 0            # max queries this action may send (0 = Weights.max_queries)


class Episode:
    def __init__(self, state, question, client, args, W: Weights):
        self.state, self.question, self.client, self.args, self.W = state, question, client, args, W
        self.stats = {k: [0, 0] for k in KINDS}     # kind -> [successes, attempts]
        self.banned, self.dead, self.llm_used = set(), set(), set()
        self.zero_novelty = Counter()
        self.challenged = False
        self.stall = 0
        self.n_search = self.n_llm = 0
        self.steps = []
        self.flags_seen = Counter()

    # ---- probabilities, costs
    def p(self, kind):
        s, a = self.stats[kind]
        return (s + self.W.prior_succ) / (a + self.W.prior_att)

    def price(self, a: Action):
        a.n_search = 0 if a.kind == "Verify" else min(a.limit or self.W.max_queries,
                                                      len(a.det) + (2 if a.llm_mode else 0))
        a.cost = self.W.c_search * a.n_search + self.W.c_llm * (1 if a.llm_mode or a.kind == "Verify" else 0)
        a.p = self.p(a.kind)
        a.vne = a.p * a.gain - a.cost
        return a

    # ---- LLM matcher wrapper (counts calls)
    async def match(self, keys, oids, texts=None):
        self.n_llm += math.ceil(len(keys) / MAP_BATCH) if keys else 0
        return await map_and_apply(self.client, self.args.model, self.state, keys, oids, texts)

    async def sweep(self):
        S = self.state
        for oid in required_retrieve(S):
            if S.ready(oid) and not S.covered(oid):
                pend = S.unchecked(oid)
                if pend:
                    await self.match(pend, [oid])

    # ---- monitor
    def monitor(self, cert):
        S = self.state
        flags = {"retrieval_loop": self.stall >= self.W.stall_limit,
                 "over_retrieval": sorted(self.dead), "confirmation_bias": []}
        covered = [o for o in required_retrieve(S) if o not in cert.uncovered]
        if covered and not cert.uncovered and cert.d_prov == 0 and not self.challenged:
            weakest = min(covered, key=lambda o: min(S.items[k].coherence for k in S.live_support(o)))
            flags["confirmation_bias"] = [weakest]
        for k in ("retrieval_loop",):
            if flags[k]:
                self.flags_seen[k] += 1
        for k in ("over_retrieval", "confirmation_bias"):
            if flags[k]:
                self.flags_seen[k] += 1
        return flags

    # ---- proposal
    def _stuck(self, oid, seen=None):
        """True if nothing more can be searched for this obligation (so its dependents need a Bridge)."""
        S, seen = self.state, (seen or set())
        if oid in seen:
            return True
        seen.add(oid)
        if oid in self.dead or ("FillGap", oid) in self.banned:
            return True
        if S.ready(oid):
            return not fillgap_queries(S, oid) and ("FillGap", oid, "paraphrase") in self.llm_used
        ups = [d for d in S.obs[oid]["depends_on"] if S.is_retrieve(d) and not S.covered(d)]
        return all(self._stuck(d, seen) for d in ups)

    def propose(self, cert, flags):
        S, W = self.state, self.W
        req_ids = S.required_ids()
        acts = []
        conflicted = set(cert.unresolved)
        for oid in required_retrieve(S):
            if oid in cert.uncovered:
                blocked_n = len([d for d in S.descendants(oid) if d in req_ids])
                gain = W.gap * (1 + blocked_n)
                if S.ready(oid):
                    if ("FillGap", oid) not in self.banned and oid not in self.dead:
                        det = fillgap_queries(S, oid)
                        mode = "paraphrase" if (not det and ("FillGap", oid, "paraphrase") not in self.llm_used) else None
                        if det or mode:
                            acts.append(Action("FillGap", oid, det, mode, gain, "uncovered"))
                else:
                    ups = [d for d in S.obs[oid]["depends_on"] if S.is_retrieve(d) and not S.covered(d)]
                    if ups and all(self._stuck(d) for d in ups) and ("Bridge", oid) not in self.banned \
                            and oid not in self.dead:
                        det = bridge_queries(S, oid, self.question)
                        if det:
                            acts.append(Action("Bridge", oid, det, None, gain, "upstream hop not found"))
                continue

            own_conflict = bool(set(S.own_vars(oid)) & conflicted)
            if oid in cert.unaudited and ("Verify", oid) not in self.banned:
                acts.append(Action("Verify", oid, [], None, W.unverified, "single point of failure"))
            if oid in self.dead:
                continue
            if (S.kappa_known(oid) < 2 or own_conflict) and ("Diversify", oid) not in self.banned:
                det = diversify_queries(S, oid)
                mode = "paraphrase" if (not det and ("Diversify", oid, "paraphrase") not in self.llm_used) else None
                if det or mode:
                    # one query: the full sentence with the value in it. A second source needs no more,
                    # and at two queries the cost would exceed p * gain for the default weights.
                    acts.append(Action("Diversify", oid, det, mode, W.conflict if own_conflict else W.fragile,
                                       "tied conflict" if own_conflict else "single source", limit=1))
            if (own_conflict or oid in flags["confirmation_bias"]) and ("Challenge", oid) not in self.banned:
                det = challenge_queries(S, oid)
                mode = "counter" if ("Challenge", oid, "counter") not in self.llm_used else None
                if det or mode:
                    acts.append(Action("Challenge", oid, det, mode, W.conflict if own_conflict else W.challenge,
                                       "conflict" if own_conflict else "confirmation bias"))
        return [self.price(a) for a in acts]

    # ---- execution
    async def verify(self, oid):
        S = self.state
        live = S.live_support(oid)
        if not live:
            self.banned.add(("Verify", oid))
            return "skipped"
        # a critical cut has ONE unit but may hold several passages of it: checking the best one
        # is enough to audit the cut (is_verified() accepts any verified live passage)
        key = live[0]
        self.n_llm += 1
        obj, err = await llm_json(self.client, self.args.model, VERIFY_SYSTEM,
                                  f"Fact: {S.instantiate(oid)}\n\nPassage: {S.items[key].text[:PASSAGE_CHARS]}")
        if err or not isinstance(obj, dict):
            self.banned.add(("Verify", oid))
            return "error"
        quote = norm(str(obj.get("quote", "")))
        if obj.get("supported") is True and len(quote) >= MIN_QUOTE and quote in norm(S.items[key].text):
            S.verified[(oid, key)] = True
            return "supported"
        S.remove_support(oid, key)
        return "rejected"

    async def execute(self, a: Action, step: int, cert):
        S, W = self.state, self.W
        rec = {"step": step, "action": a.kind, "oid": a.oid, "why": a.why, "gain": round(a.gain, 3),
               "p": round(a.p, 3), "cost": round(a.cost, 3), "vne": round(a.vne, 3), "queries": [], "new": 0}
        before = cert.vector
        new_keys = []
        if a.kind == "Verify":
            rec["verify"] = await self.verify(a.oid)
        else:
            qs = list(a.det)
            if a.llm_mode:
                self.llm_used.add((a.kind, a.oid, a.llm_mode))
                self.n_llm += 1
                extra = await llm_queries(self.client, self.args.model, S.instantiate(a.oid), a.llm_mode)
                qs = _unique(qs + extra, S.tried(a.oid))
            qs = qs[:a.limit or W.max_queries]
            if a.kind == "Challenge":
                self.challenged = True
            if not qs:
                self.banned.add((a.kind, a.oid))
                rec["note"] = "no query left"
            for q in qs:
                try:
                    res = parse_results(await asyncio.to_thread(retrieve_sync, self.args.search_url, q))
                except Exception as e:
                    rec.setdefault("errors", []).append(f"{type(e).__name__}: {e}"[:150])
                    S.note_query(a.oid, q)
                    continue
                self.n_search += 1
                S.note_query(a.oid, q)
                new_keys += S.add_items(res, step, q)
                rec["queries"].append(q)
            rec["new"] = len(new_keys)
            if new_keys:
                texts = {a.oid: S.instantiate(a.oid, keep_own=True)} if a.kind in ("Diversify", "Challenge") else None
                # Diversify / Challenge look for another value of THEIR obligation only; letting them
                # cover a different obligation as a side effect produced wrong evidence
                ready = [a.oid] if texts else [o for o in S.order if S.ready(o)]
                matched, rejected, dbg = await self.match(new_keys, ready, texts)
                rec["matched"], rec["rejected"], rec["llm_error"] = matched, rejected, dbg["error"]
            # novelty bookkeeping (over-retrieval)
            if a.kind != "Verify" and rec["queries"]:
                if new_keys:
                    self.zero_novelty[a.oid] = 0
                else:
                    self.zero_novelty[a.oid] += 1
                    if self.zero_novelty[a.oid] >= W.novelty_limit:
                        self.dead.add(a.oid)
        after = audit(S)
        improved = pareto_improves(before, after.vector)
        self.stats[a.kind][1] += 1
        self.stats[a.kind][0] += int(improved)
        rec["improved"] = improved
        rec["vector"] = [round(x, 3) for x in after.vector]
        if a.kind != "Verify":
            self.stall = 0 if (improved or new_keys) else self.stall + 1
        self.steps.append(rec)
        return after

    # ---- main loop
    async def run(self):
        S = self.state
        cert, stop = None, "max_steps"
        for step in range(1, self.args.max_steps + 1):
            await self.sweep()
            cert = audit(S)
            flags = self.monitor(cert)
            if flags["retrieval_loop"]:
                stop = "stalled"
                break
            acts = self.propose(cert, flags)
            best = max(acts, key=lambda a: a.vne, default=None)
            if best is None or best.vne <= 0:
                stop = "certified" if cert.complete else ("no_positive_vne" if acts else "blocked")
                break
            if self.n_search + best.n_search > self.args.max_searches:
                stop = "budget"
                break
            cert = await self.execute(best, step, cert)
        if stop in ("stalled", "max_steps", "budget"):  # a last look at held passages before answering
            await self.sweep()
            cert = audit(S)
        return stop, cert


# ------------------------------------------------------------------ one question
async def solve(client, args, prov, rec, W=None):
    W = W or Weights()
    state = EvidenceState(rec["obligations"], prov)
    ep = Episode(state, rec["question"], client, args, W)
    stop, cert = await ep.run()

    gold = rec.get("gold") or []
    uncovered = list(cert.uncovered)
    abstained = bool(getattr(args, "abstain_uncertified", False)) and not cert.complete
    if abstained:
        ans, ans_err = "", None
    else:
        ans, ans_err = await answer_question(client, args.model, state, rec["question"])
        ep.n_llm += 1
    em, f1 = em_f1(ans, gold)

    pathologies = {k: v for k, v in ep.flags_seen.items()}
    if uncovered and ans:
        pathologies["premature_confidence"] = 1
    if ans and state.obs[state.answer_id]["kind"] == "retrieve":
        texts = " ".join(norm(it.text) for it in state.items.values())
        if norm(ans) and norm(ans) not in texts:
            pathologies["ungrounded_answer"] = 1

    vals = state.all_values()
    return {
        "answer": ans, "answer_error": ans_err, "em": em, "f1": f1,
        "question": rec["question"], "gold": gold, "coverage": cert.coverage, "stop": stop,
        "certified": cert.complete, "certificate": cert.to_dict(), "abstained": abstained,
        "answered_from_incomplete": bool(uncovered) and not abstained,
        "uncovered": uncovered, "pathologies": pathologies,
        "searches": ep.n_search, "llm_calls": ep.n_llm,
        "action_stats": {k: {"success": s, "attempts": a} for k, (s, a) in ep.stats.items() if a},
        "steps": ep.steps, "bindings": vals, "conflicts": state.conflicts(), "n_items": len(state.items),
        "obligations": [{"id": o, "kind": state.obs[o]["kind"], "text": state.instantiate(o),
                         "covered": state.covered(o), "kappa": state.kappa(o),
                         "titles": sorted({t for k in state.live_support(o) for t in (state.items[k].titles or [UNKNOWN])})}
                        for o in state.order],
    }


def show(r):
    print("\nQ:", r["question"], "| gold:", str(r["gold"])[:50])
    for s in r["steps"]:
        print(f"  #{s['step']} {s['action']:9s} {s['oid']} vne={s['vne']:+.2f} (p={s['p']:.2f} gain={s['gain']:.2f} "
              f"cost={s['cost']:.2f}) {s['why']} | q={s['queries']} new={s['new']} "
              f"{'verify=' + s['verify'] if 'verify' in s else ''} improved={s['improved']} vec={s['vector']}")
    for o in r["obligations"]:
        mark = "OK" if o["covered"] else ("--" if o["kind"] == "retrieve" else "dv")
        print(f"  [{mark}] {o['id']} k={o['kappa']} {o['text']}  {o['titles']}")
    c = r["certificate"]
    print(f"  stop={r['stop']} certified={r['certified']} reasons={c['reasons']} D={c['d_prov']:.2f} R={c['r_struct']:.2f}")
    print(f"  pathologies={r['pathologies']} searches={r['searches']} llm={r['llm_calls']} bindings={r['bindings']}")


def compact(i, r):
    tag = "CERT" if r["certified"] else ("ABSTAIN" if r["abstained"] else "UNCERT")
    print(f"[{i}] {tag} stop={r['stop']} EM={r['em']:.0f} s={r['searches']} llm={r['llm_calls']} "
          f"ans='{r['answer'][:30]}' gold={[str(g)[:20] for g in (r['gold'] or [])[:2]]} | {r['question'][:60]}")


def summary(rs, model):
    avg = lambda xs, k: (sum(x[k] for x in xs) / len(xs)) if xs else float("nan")
    cert = [r for r in rs if r["certified"]]
    unc = [r for r in rs if not r["certified"]]
    print(f"\n==== {len(rs)} questions | model={model}")
    print(f"EM={avg(rs, 'em'):.3f} F1={avg(rs, 'f1'):.3f} | avg searches={avg(rs, 'searches'):.1f} "
          f"avg LLM calls={avg(rs, 'llm_calls'):.1f}")
    print(f"certified: {len(cert)}/{len(rs)}  EM={avg(cert, 'em'):.3f}  |  not certified: {len(unc)}  EM={avg(unc, 'em'):.3f}"
          f"  (abstained {sum(r['abstained'] for r in unc)})")
    print("stop reasons:", dict(Counter(r["stop"] for r in rs)))
    acts = Counter()
    succ = Counter()
    for r in rs:
        for k, v in r["action_stats"].items():
            acts[k] += v["attempts"]
            succ[k] += v["success"]
    print("actions (success/attempts):", {k: f"{succ[k]}/{acts[k]}" for k in acts})
    print("pathologies:", dict(sum((Counter(r["pathologies"]) for r in rs), Counter())))
    errs = sum(1 for r in rs for s in r["steps"] if s.get("llm_error") or s.get("errors")) + sum(1 for r in rs if r.get("answer_error"))
    if errs:
        print(f"!! {errs} LLM/search call(s) failed - check these results before using them")


async def main():
    d = Weights()
    ap = argparse.ArgumentParser()
    ap.add_argument("--obligations", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-searches", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--search-url", default=SEARCH_URL, help="Graph-R1 /search API (or EVIDENCE_SEARCH_URL)")
    ap.add_argument("--api-base", default=LLM_BASE_URL, help="LLM server (or EVIDENCE_LLM_BASE_URL)")
    ap.add_argument("--model", default=LLM_MODEL, help="LLM name (or EVIDENCE_LLM_MODEL)")
    ap.add_argument("--abstain-uncertified", action="store_true",
                    help="empty answer when the certificate is not complete (default: answer anyway, flagged)")
    ap.add_argument("--out", default="trace_ctrl.jsonl")
    ap.add_argument("--show", action="store_true")
    for f in ("gap", "conflict", "challenge", "fragile", "unverified", "c_search", "c_llm", "prior_succ", "prior_att"):
        ap.add_argument(f"--w-{f.replace('_', '-')}", type=float, default=getattr(d, f))
    for f in ("stall_limit", "novelty_limit", "max_queries"):
        ap.add_argument(f"--{f.replace('_', '-')}", type=int, default=getattr(d, f))
    args = ap.parse_args()
    if not args.search_url:
        ap.error("--search-url (or EVIDENCE_SEARCH_URL) is required")
    W = replace(d, **{f: getattr(args, f"w_{f}") for f in ("gap", "conflict", "challenge", "fragile", "unverified",
                                                          "c_search", "c_llm", "prior_succ", "prior_att")},
                **{f: getattr(args, f) for f in ("stall_limit", "novelty_limit", "max_queries")})

    recs = [json.loads(l) for l in open(args.obligations, encoding="utf-8")]
    recs = [r for r in recs if r["ok"]][args.start: args.start + args.n]
    prov = ProvenanceIndex.load()
    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")
    try:
        await client.chat.completions.create(
            model=args.model, max_tokens=5, temperature=0, messages=[{"role": "user", "content": "ping"}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    except Exception as e:
        print(f"LLM server unreachable at {args.api_base}: {type(e).__name__}: {e}")
        return

    sem = asyncio.Semaphore(1 if args.show else args.concurrency)

    async def one(i, rec):
        async with sem:
            try:
                return i, await solve(client, args, prov, rec, W)
            except Exception as e:
                return i, {"question": rec["question"], "gold": rec.get("gold"), "coverage": 0.0,
                           "stop": f"crash: {type(e).__name__}: {e}"[:120], "answer": "", "answer_error": None,
                           "em": 0.0, "f1": 0.0, "certified": False, "certificate": {"reasons": ["crash"], "d_prov": 0, "r_struct": 0},
                           "abstained": False, "answered_from_incomplete": False, "uncovered": [], "pathologies": {},
                           "searches": 0, "llm_calls": 0, "action_stats": {}, "steps": [], "bindings": {},
                           "conflicts": {}, "n_items": 0, "obligations": []}

    results = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for fut in asyncio.as_completed([asyncio.create_task(one(i, r)) for i, r in enumerate(recs)]):
            i, r = await fut
            results[i] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if args.show:
                show(r)
            compact(i, r)
    summary([results[i] for i in sorted(results)], args.model)


if __name__ == "__main__":
    asyncio.run(main())
