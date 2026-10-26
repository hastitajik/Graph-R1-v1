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
                                                  entity-only form, one blind LLM paraphrase, then up to
                                                  two INFORMED LLM queries that read the passages retrieved
                                                  so far)
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
    p      = (successes + prior_succ) / (attempts + prior_att), per action kind; by default the counts are
             carried over the questions of the run (--no-learn-p: per question), so an optional action that
             never pays (Diversify on a corpus with one article per fact) is dropped. FillGap and Verify
             never fall below the prior: a certificate needs them. A Diversify that resolves a tied conflict
             keeps its own counts. A step succeeds when the audit vector (C, R, D, -X) Pareto-improves.
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
from dataclasses import dataclass, field, replace

from openai import AsyncOpenAI

from evidence.audit import audit, pareto_improves, required_retrieve, risk
from evidence.common import LLM_BASE_URL, LLM_MODEL, SEARCH_URL
from evidence.derive import derive_answer, question_options
from evidence.metrics import em_f1
from evidence.obligations import extract_json
from evidence.provenance import ProvenanceIndex, norm
from evidence.run_mvp import (MAP_BATCH, PASSAGE_CHARS, answer_question, make_query, map_and_apply,
                              parse_results, retrieve_sync)
from evidence.state import MIN_QUOTE, UNKNOWN, EvidenceState, same_entity

SEARCH_KINDS = ("FillGap", "Bridge", "Diversify", "Challenge")
KINDS = SEARCH_KINDS + ("Verify",)
ESSENTIAL = ("FillGap", "Verify")   # their success rate is never allowed to fall below the prior
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
    risk_mode: str = "pareto"  # "pareto": a step must not worsen any component; "scalar": Risk(s) must drop
    lam_x: float = 0.5         # weight of unresolved conflicts in the scalar risk
    lam_r: float = 0.5         # weight of (1 - robustness) in the scalar risk
    alpha: float = 0.5         # step reward r = dC + alpha dR + beta dD - gamma dX - lam_cost * cost
    beta: float = 0.5
    gamma: float = 0.5
    lam_cost: float = 1.0
    learn_p: bool = False     # carry the success counts of each action kind over to the next questions
    disabled: tuple = ()      # action kinds switched off (ablations)
    seen: dict = field(default_factory=lambda: {})  # kind -> [successes, attempts] over all questions


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

REFORM_SYSTEM["informed"] = (
    "Write 2 new search queries for a retrieval system. The earlier queries did not find a passage that "
    "states the missing fact. Read the passages retrieved so far (titles and openings): they may name the "
    "right entity under another title, or show what the corpus calls the relation. Write queries that use "
    "the exact names and words you see there. Do not repeat an earlier query. Keep entity names exactly as "
    'written. Return ONLY JSON: {"queries": ["...", "..."]}')

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


def context_snippets(state, n=6, chars=160):
    """Titles and openings of the best passages retrieved so far (what an informed query may build on)."""
    best = sorted(state.items.values(), key=lambda it: -it.coherence)[:n]
    return [f"[{' | '.join(it.titles) or '?'}] {it.text[:chars]}" for it in best]


async def informed_queries(client, model, question, fact, tried, snippets):
    user = (f"Question: {question}\nMissing fact: {fact}\nEarlier queries: {json.dumps(list(tried))}\n\n"
            "Retrieved so far:\n" + "\n".join(snippets))
    obj, err = await llm_json(client, model, REFORM_SYSTEM["informed"], user)
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

    @property
    def stat(self):
        """Statistics key: a Diversify that resolves a tied conflict is a different bet from one that looks for a
        second source of an unchallenged fact, so each learns its own success rate."""
        return "Diversify/conflict" if self.kind == "Diversify" and self.why == "tied conflict" else self.kind


class Episode:
    def __init__(self, state, question, client, args, W: Weights):
        self.state, self.question, self.client, self.args, self.W = state, question, client, args, W
        self.stats = {k: [0, 0] for k in KINDS}     # kind -> [successes, attempts]
        self.banned, self.dead, self.llm_used = set(), set(), set()
        self.zero_novelty = Counter()
        self.search_errors = Counter()
        self.challenged = False
        self.stall = 0
        self.n_search = self.n_llm = 0
        self.steps = []
        self.flags_seen = Counter()

    def better(self, before, after, kind):
        """Did the step help? A search that DISCOVERS a conflicting value (Diversify / Challenge) also counts:
        it makes the state look worse on paper (X grows) but it is exactly the evidence the audit needs."""
        if self.W.risk_mode == "scalar":
            ok = risk(before, self.W.lam_x, self.W.lam_r) - risk(after, self.W.lam_x, self.W.lam_r) > 1e-9
        else:
            ok = pareto_improves(before, after)
        return ok or (kind in ("Diversify", "Challenge") and after[3] < before[3])

    # ---- probabilities, costs
    def p(self, kind):
        # with learn_p the run-wide counter W.seen already contains the attempts of THIS question
        # (execute() updates both), so it replaces the per-question counter instead of being added to it
        s, a = self.W.seen.get(kind, (0, 0)) if self.W.learn_p else self.stats.get(kind, (0, 0))
        p = (s + self.W.prior_succ) / (a + self.W.prior_att)
        if kind in ESSENTIAL:   # a certificate needs them: bad luck on earlier questions must not switch them off
            p = max(p, self.W.prior_succ / self.W.prior_att)
        return p

    def price(self, a: Action):
        a.n_search = 0 if a.kind == "Verify" else min(a.limit or self.W.max_queries,
                                                      len(a.det) + (2 if a.llm_mode else 0))
        a.cost = self.W.c_search * a.n_search + self.W.c_llm * (1 if a.llm_mode or a.kind == "Verify" else 0)
        a.p = self.p(a.stat)
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
        for k in ("retrieval_loop", "over_retrieval", "confirmation_bias"):
            if flags[k]:
                self.flags_seen[k] = 1   # a question is flagged once, however many steps it stays true
        return flags

    def fillgap_mode(self, oid):
        """Next LLM help for an obligation whose plain queries are used up: one blind paraphrase, then up to
        two INFORMED rounds that read what was retrieved so far (only if something was retrieved)."""
        if ("FillGap", oid, "paraphrase") not in self.llm_used:
            return "paraphrase"
        if self.state.items and sum(("FillGap", oid, f"informed{i}") in self.llm_used for i in (1, 2)) < 2:
            return "informed"
        return None

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
            return not fillgap_queries(S, oid) and self.fillgap_mode(oid) is None
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
                        mode = self.fillgap_mode(oid) if not det else None
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
        return [self.price(a) for a in acts if a.kind not in self.W.disabled]

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
        self.reject_note = {"fact": S.instantiate(oid), "passage": S.items[key].text[:200]}
        return "rejected"

    async def execute(self, a: Action, step: int, cert):
        S, W = self.state, self.W
        rec = {"step": step, "action": a.kind, "oid": a.oid, "why": a.why, "gain": round(a.gain, 3),
               "p": round(a.p, 3), "cost": round(a.cost, 3), "vne": round(a.vne, 3), "queries": [], "new": 0}
        before = cert.vector
        new_keys = []
        if a.kind == "Verify":
            self.reject_note = None
            rec["verify"] = await self.verify(a.oid)
            if self.reject_note:
                rec["verify_rejected"] = self.reject_note
        else:
            qs = list(a.det)
            if a.llm_mode:
                mode = a.llm_mode
                if mode == "informed":
                    mode = "informed1" if (a.kind, a.oid, "informed1") not in self.llm_used else "informed2"
                self.llm_used.add((a.kind, a.oid, mode))
                self.n_llm += 1
                if a.llm_mode == "informed":
                    extra = await informed_queries(self.client, self.args.model, self.question, S.instantiate(a.oid),
                                                   S.tried(a.oid), context_snippets(S))
                else:
                    extra = await llm_queries(self.client, self.args.model, S.instantiate(a.oid), a.llm_mode)
                rec["llm_queries"] = extra
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
                    # a failed search says nothing about the query: it is NOT marked as tried, and it does not
                    # count as a stalled step; after repeated failures the obligation is given up
                    rec.setdefault("errors", []).append(f"{type(e).__name__}: {e}"[:150])
                    self.search_errors[a.oid] += 1
                    if self.search_errors[a.oid] >= 3:
                        self.dead.add(a.oid)
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
        improved = self.better(before, after.vector, a.kind)
        st = self.stats.setdefault(a.stat, [0, 0])
        st[1] += 1
        st[0] += int(improved)
        g = self.W.seen.setdefault(a.stat, [0, 0])
        g[1] += 1
        g[0] += int(improved)
        rec["improved"] = improved
        d = [x - y for x, y in zip(after.vector, before)]   # (dC, dR, dD, -dX)
        rec["reward"] = round(d[0] + W.alpha * d[1] + W.beta * d[2] + W.gamma * d[3] - W.lam_cost * a.cost, 4)
        rec["vector"] = [round(x, 3) for x in after.vector]
        if a.kind != "Verify" and not (rec.get("errors") and not rec["queries"]):
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
def _toks(s):
    return re.findall(r"\w+", (s or "").lower())


def value_match(a, b):
    """Do two surface forms name the same value? Exact, alias, or token F1 >= 0.6 ('Eton College' ~ 'Eton')."""
    if not norm(a) or not norm(b):
        return False
    if norm(a) == norm(b) or same_entity(a, b):
        return True
    ta, tb = set(_toks(a)), set(_toks(b))
    hit = len(ta & tb)
    if not hit:
        return False
    p, r = hit / len(ta), hit / len(tb)
    return 2 * p * r / (p + r) >= 0.6


def grounding(state, ans, question="", derived=None):
    """(ok, why, basis). A certificate covers the ANSWER, so the answer must come out of the certified evidence:
      retrieved  - the answer obligation yields a value and the answer IS that value (or is quoted from the
                   passages that support it); a value bound for another obligation does not count
      computed   - a date comparison computed from the certified dates (derive.py)
      option     - a derive answer that names one of the two options of the question; the facts behind it are
                   certified, the comparison itself is the LLM's
      yes/no     - a yes/no derive answer; same remark"""
    a = norm(ans or "")
    if not a or a == "notfound":
        return False, "no answer from the evidence", None
    if state.obs[state.answer_id]["kind"] != "retrieve":
        if derived and norm(derived.get("answer", "")) == a:
            return True, "", "computed"
        opts = question_options(question or "") or ()
        if any(value_match(ans, o) for o in opts):
            return True, "", "option"
        if a in ("yes", "no"):
            return True, "", "yes/no"
        return False, "derived answer is neither computed nor an option named in the question", None
    vals = [state.value(v) for v in state.own_vars(state.answer_id)]
    if any(v and value_match(ans, v) for v in vals):
        return True, "", "retrieved"
    own = set(state.own_vars(state.answer_id))
    others = [state.value(v) for v in state.all_values() if v not in own]
    if any(o and value_match(ans, o) for o in others):
        return False, "answer is not grounded: it is the value of another obligation, not of the answer", None
    if len(a) >= 3:
        pat = re.compile(r"(?<!\w)" + re.escape((ans or "").strip().lower()) + r"(?!\w)")
        if any(pat.search(state.items[k].text.lower()) for k in state.live_support(state.answer_id)):
            return True, "", "retrieved"
    return False, "answer is not grounded: it is not the value the certified evidence gives", None


def answer_grounded(state, ans, question="", derived=None):
    ok, why, _ = grounding(state, ans, question, derived)
    return ok, why


async def solve(client, args, prov, rec, W=None):
    W = W or Weights()
    state = EvidenceState(rec["obligations"], prov)
    state.question = rec["question"]
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
    ans_llm, derived = ans, None
    if ans and getattr(args, "det_compare", True):
        # the last step of a date comparison needs no LLM: compute it from the certified values
        derived = derive_answer(state, rec["question"])
        if derived:
            ans = derived["answer"]
    em, f1 = em_f1(ans, gold)
    grounded, why, basis = grounding(state, ans, rec["question"], derived)
    certified = cert.complete and (grounded or abstained)
    cert_d = cert.to_dict()
    cert_d["risk"] = round(risk(cert.vector, W.lam_x, W.lam_r), 4)
    if cert.complete and not certified:
        cert_d["reasons"] = list(cert_d.get("reasons", [])) + [why]
        if stop == "certified":
            stop = "ungrounded"   # the audit passed but the answer is not backed by it: do not call it certified

    pathologies = {k: v for k, v in ep.flags_seen.items()}
    if derived and norm(ans_llm) != norm(derived["answer"]):
        pathologies["derive_mismatch"] = 1   # the LLM's comparison disagreed with the computed one
    if uncovered and ans:
        pathologies["premature_confidence"] = 1
    if ans and state.obs[state.answer_id]["kind"] == "retrieve":
        texts = " ".join(norm(it.text) for it in state.items.values())
        if norm(ans) and norm(ans) not in texts:
            pathologies["ungrounded_answer"] = 1

    vals = state.all_values()
    return {
        "answer": ans, "answer_llm": ans_llm, "derived": derived, "answer_error": ans_err, "em": em, "f1": f1,
        "question": rec["question"], "gold": gold, "coverage": cert.coverage, "stop": stop,
        "certified": certified, "certificate": cert_d, "abstained": abstained,
        "answer_basis": basis if certified or grounded else None,
        "answered_from_incomplete": bool(uncovered) and not abstained,
        "uncovered": uncovered, "pathologies": pathologies,
        "unknown_source_rate": unknown_rate(state),
        "searches": ep.n_search, "llm_calls": ep.n_llm,
        "action_stats": {k: {"success": s, "attempts": a} for k, (s, a) in ep.stats.items() if a},
        "steps": ep.steps, "bindings": vals, "conflicts": state.conflicts(), "n_items": len(state.items),
        "obligations": [{"id": o, "kind": state.obs[o]["kind"], "text": state.instantiate(o),
                         "covered": state.covered(o), "kappa": state.kappa(o),
                         "titles": sorted({t for k in state.live_support(o) for t in (state.items[k].titles or [UNKNOWN])})}
                        for o in state.order],
    }


def unknown_rate(state):
    """Share of the supporting passages whose article is unknown. A high value means the provenance index does not
    fit the corpus (chunk layout, titles): independence then cannot be judged and D / kappa are meaningless."""
    keys = [k for o in required_retrieve(state) for k in state.live_support(o)]
    return round(sum(not state.items[k].titles for k in keys) / len(keys), 3) if keys else 0.0


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
    print("questions flagged by the monitor:", dict(sum((Counter(r["pathologies"]) for r in rs), Counter())))
    rew = {}
    for r in rs:
        for s in r["steps"]:
            if "reward" in s:
                rew.setdefault(s["action"], []).append(s["reward"])
    if rew:
        print("mean step reward (dC + a dR + b dD - g dX - lam cost):",
              {k: round(sum(v) / len(v), 3) for k, v in rew.items()})
    ver = Counter(s["verify"] for r in rs for s in r["steps"] if "verify" in s)
    if ver:
        print("Verify outcomes:", dict(ver), "(rejected = support removed; check these by hand for false rejections)")
    dm = [r for r in rs if r.get("derived")]
    if dm:
        bad = [r for r in dm if r["pathologies"].get("derive_mismatch")]
        print(f"computed date comparisons: {len(dm)} | the LLM's own comparison differed in {len(bad)}")
    basis = Counter(r.get("answer_basis") for r in rs if r["certified"])
    if basis:
        print("certified answers by basis:", dict(basis), "(computed / retrieved = checked against the evidence; "
              "option / yes-no = the facts are certified, the final comparison is the LLM's)")
    unk = sum(r.get("unknown_source_rate", 0) for r in rs) / max(len(rs), 1)
    if unk > 0.5:
        print(f"!! {unk:.0%} of the supporting passages have no known article: the provenance index does not fit this "
              "corpus, independence (D, kappa, critical cuts) is not meaningful")
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
    ap.add_argument("--no-det-compare", dest="det_compare", action="store_false",
                    help="let the LLM answer date-comparison questions (default: compute them from the certified dates)")
    ap.add_argument("--no-learn-p", dest="learn_p", action="store_false",
                    help="estimate p per question only (default: the success counts of each optional action are "
                         "carried over to the next questions of the run, so an action that never pays is dropped; "
                         "with --concurrency > 1 the order of those updates, and so the run, is not bit-reproducible)")
    ap.add_argument("--disable", default="", help="comma list of action kinds to switch off, e.g. Diversify,Verify")
    ap.add_argument("--risk-mode", choices=("pareto", "scalar"), default="pareto",
                    help="how a step is judged: Pareto vector (default) or the scalar Risk of the plan")
    ap.add_argument("--lam-x", type=float, default=0.5, help="scalar risk: weight of unresolved conflicts")
    ap.add_argument("--lam-r", type=float, default=0.5, help="scalar risk: weight of (1 - robustness)")
    ap.add_argument("--unit", choices=("article", "passage"), default="article",
                    help="independence unit: Wikipedia article (provenance-aware) or every passage (naive ablation)")
    for f in ("alpha", "beta", "gamma", "lam_cost"):
        ap.add_argument(f"--r-{f.replace('_', '-')}", type=float, default=getattr(d, f), help="step reward weight")
    ap.add_argument("--out", default="trace_ctrl.jsonl")
    ap.add_argument("--show", action="store_true")
    for f in ("gap", "conflict", "challenge", "fragile", "unverified", "c_search", "c_llm", "prior_succ", "prior_att"):
        ap.add_argument(f"--w-{f.replace('_', '-')}", type=float, default=getattr(d, f))
    for f in ("stall_limit", "novelty_limit", "max_queries"):
        ap.add_argument(f"--{f.replace('_', '-')}", type=int, default=getattr(d, f))
    args = ap.parse_args()
    if not args.search_url:
        ap.error("--search-url (or EVIDENCE_SEARCH_URL) is required")
    import evidence.state as state_mod
    state_mod.UNIT_MODE = args.unit
    W = replace(d, seen={}, alpha=args.r_alpha, beta=args.r_beta, gamma=args.r_gamma, lam_cost=args.r_lam_cost,
                learn_p=args.learn_p, risk_mode=args.risk_mode, lam_x=args.lam_x, lam_r=args.lam_r, disabled=tuple(k for k in args.disable.split(",") if k),
                **{f: getattr(args, f"w_{f}") for f in ("gap", "conflict", "challenge", "fragile", "unverified",
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
