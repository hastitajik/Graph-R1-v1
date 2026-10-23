"""
MVP loop: Layers 1+2 with a single action (FillGap) and the simple stop rule
"all retrieve-obligations covered". Retrieval = the existing /search API (unchanged).

Run from the project root (inside the container):
  python3 -m evidence.run_mvp --obligations obligations_dev30.jsonl --n 5 \
      --search-url http://<API_IP>:8001/search \
      --api-base http://192.168.4.114:5579/v1 --model alpha-Wolf --out trace_mvp.jsonl
"""
import argparse
import asyncio
import json
import re
import string
import urllib.request

from openai import AsyncOpenAI

from evidence.obligations import extract_json
from evidence.audit import audit
from evidence.controller import plan
from evidence.provenance import ProvenanceIndex, clean_hyperedge, norm
from evidence.state import EvidenceState

PASSAGE_CHARS = 1500
UNK = "UNKNOWN"
VERBOSE = False

MAP_SYSTEM = """You check whether evidence passages support obligations.

For each obligation (a fact to establish) and each numbered passage, decide whether the \
passage EXPLICITLY states that fact. Use only the passage, never outside knowledge.
If it does, return the value of every ?variable in the obligation exactly as written in \
the passage, and copy a short quote (the exact words) that states the fact.
Skip passages that merely mention the entity without stating the fact.
Each passage is labelled with the title of its source article. If the passage uses a pronoun \
(He/She/It/The film), it is about that article's subject.
Reading "(born date, place - died date, place)" parentheses is allowed. Close synonyms count \
(educated at = graduated from, half-brother = brother).
Return bindings ONLY for the ?variables of the obligation you are matching. A year alone \
(for example "1926") is a valid value for a date variable.

Return ONLY a JSON object:
{"matches":[{"passage":0,"obligation":"o1","bindings":{"?D":"value"},"quote":"exact words"}]}
Use {"matches":[]} if nothing is supported."""


def make_query(text: str) -> str:
    """Instantiated obligation -> search query: drop the unknown ?variables."""
    return re.sub(r"\s*\?[A-Za-z]\w*", "", text).strip(" ,.")


def parse_results(raw):
    """Tolerant parser for the /search response -> [{"text", "coherence"}]."""
    out = []

    def walk(x):
        if isinstance(x, str):
            try:
                walk(json.loads(x))
            except Exception:
                pass
        elif isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            if "<knowledge>" in x:
                try:
                    c = float(x.get("<coherence>", 0) or 0)
                except (TypeError, ValueError):
                    c = 0.0
                out.append({"text": clean_hyperedge(str(x["<knowledge>"])), "coherence": c})
            else:
                for v in x.values():
                    walk(v)

    walk(raw)
    return out


def retrieve_sync(url, query, timeout=180):
    req = urllib.request.Request(url, data=json.dumps({"queries": [query]}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


async def map_passages(client, model, state, new_keys, ready_oids):
    """One LLM call: which new passages support which ready obligations. Returns validated matches."""
    items = [state.items[k] for k in new_keys]
    obs = "\n".join(f"{oid}: {state.instantiate_open(oid)}" for oid in ready_oids)
    pas = "\n".join(f"[{i}] (article: {', '.join(it.titles) or 'unknown'}) {it.text[:PASSAGE_CHARS]}"
                    for i, it in enumerate(items))
    user = f"Obligations:\n{obs}\n\nPassages:\n{pas}"
    raw = None
    dbg = {"error": None, "raw_text": None, "proposed": 0}
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": MAP_SYSTEM}, {"role": "user", "content": user}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            dbg["raw_text"] = (resp.choices[0].message.content or "")[:800]
            raw = extract_json(resp.choices[0].message.content)
            dbg["error"] = None
            break
        except Exception as e:
            dbg["error"] = f"{type(e).__name__}: {e}"[:300]
            raw = None
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
    dbg["proposed"] = len((raw or {}).get("matches", []))
    good, rejected = [], []
    for m in (raw or {}).get("matches", []):
        try:
            item = items[int(m["passage"])]
            b = m.get("bindings") or {}
            # the model often writes the wrong obligation id: the variable decides who owns the match
            owners = {state.owner_of(v) for v in b} - {None}
            if len(owners) == 1:
                m["obligation"] = next(iter(owners))
            if m["obligation"] not in ready_oids:
                ok, why = False, f"{m['obligation']} was not requested (already covered or not ready)"
            else:
                ok, why = state.validate_match(m["obligation"], item, b, m.get("quote", ""))
        except Exception as e:
            ok, why = False, f"bad match: {e}"
        (good if ok else rejected).append((m, why))
    return good, rejected, items, dbg


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred: str, golds):
    """Max over gold aliases (SQuAD-style). Quick signal only - verify against the repo's eval before comparing with the baseline."""
    best_em, best_f1 = 0.0, 0.0
    p = normalize_answer(pred).split()
    for g in golds or []:
        gt = normalize_answer(str(g)).split()
        best_em = max(best_em, float(p == gt))
        common = sum((__import__("collections").Counter(p) & __import__("collections").Counter(gt)).values())
        if common and p and gt:
            pr, rc = common / len(p), common / len(gt)
            best_f1 = max(best_f1, 2 * pr * rc / (pr + rc))
    return best_em, best_f1


ANSWER_SYSTEM = """Answer the question using ONLY the established facts below.
Compare values when the question asks for it (earlier/later, older/younger, same country).
Give a SHORT answer: a name, date, place, or yes/no. For "which film/person" questions answer \
with the name exactly as written in the question. Every fact below is established; do not use \
outside knowledge.
Return ONLY JSON: {"answer": "..."}"""


async def answer_question(client, model, state, question):
    lines = []
    for oid in state.order:
        text = state.instantiate(oid)
        if state.is_retrieve(oid) and state.covered(oid):
            best = max((state.items[k] for k in state.support[oid]), key=lambda it: it.coherence)
            lines.append(f"- {text}\n  evidence: {best.text[:300]}")
        elif state.is_retrieve(oid):
            lines.append(f"- {text}  (NOT FOUND)")
        else:
            lines.append(f"- (to derive) {text}")
    user = f"Question: {question}\n\nFacts:\n" + "\n".join(lines)
    err = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            raw = resp.choices[0].message.content or ""
            return str(extract_json(raw).get("answer", "")).strip(), None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
    return "", err


ALT_SYSTEM = """You write ONE alternative search query that should retrieve a DIFFERENT source \
stating the same fact (use synonyms, the full name, or a closely related entity). Do not answer \
it and do not repeat a previous query.
Return ONLY JSON: {"query": "..."}"""


async def alt_query(client, model, text, used):
    user = f"Fact to find: {text}\nQueries already used: {json.dumps(used, ensure_ascii=False)}"
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": ALT_SYSTEM}, {"role": "user", "content": user}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        q = str(extract_json(resp.choices[0].message.content or "").get("query", "")).strip()
    except Exception:
        q = ""
    return q if q and q not in used else ""


async def vne_loop(client, args, state, log):
    """Decision-time controller: re-plan after every action, stop when no action has VNE > 0."""
    diversified, used = set(), {}
    for step in range(1, args.max_actions + 1):
        before = audit(state, args.lam_x, args.lam_r)
        acts = plan(state, args.lam_r, args.cost, diversified)
        if not acts:
            return "vne_stop"
        a = acts[0]
        oid = a["oid"]
        text = state.instantiate(oid)
        entry = {"round": step, "action": a["action"], "obligation": oid, "vne": round(a["vne"], 3),
                 "risk_before": before["risk"]}
        if a["action"] == "FillGap":
            q = make_query(text)
        else:
            diversified.add(oid)          # one attempt per obligation in v0
            q = await alt_query(client, args.model, text, used.get(oid, []))
            if not q:
                entry["note"] = "no alternative query"
                log.append(entry)
                continue
        used.setdefault(oid, []).append(q)
        res = parse_results(await asyncio.to_thread(retrieve_sync, args.search_url, q))
        new = state.add_items(res, step, q)
        k_before = state.kappa(oid)
        entry.update({"query": q, "returned": len(res), "new": len(new)})
        if new:
            targets = list(dict.fromkeys([oid] + [o for o in state.order if state.ready(o) and not state.covered(o)]))
            good, rejected, items, dbg = await map_passages(client, args.model, state, new, targets)
            for m, _ in good:
                state.apply_match(m["obligation"], new[int(m["passage"])], m.get("bindings") or {})
            entry["matched"] = [(m["obligation"], m.get("bindings")) for m, _ in good]
            entry["rejected"] = [(m.get("obligation"), why) for m, why in rejected]
            entry["llm"] = dbg
        if a["action"] == "FillGap" and not state.covered(oid):
            state.exhausted.add(oid)       # the same query would return the same passages
        entry["kappa_gain"] = state.kappa(oid) - k_before
        entry["risk_after"] = audit(state, args.lam_x, args.lam_r)["risk"]
        log.append(entry)
    return "max_actions"


async def solve(client, args, prov, rec):
    state = EvidenceState(rec["obligations"], prov)
    log, stop = [], "max_rounds"
    use_vne = getattr(args, "mode", "fillgap") == "vne"
    if use_vne:
        stop = await vne_loop(client, args, state, log)
    for rnd in (() if use_vne else range(1, args.max_rounds + 1)):
        targets = [o for o in state.order if state.is_retrieve(o) and not state.covered(o)
                   and state.ready(o) and o not in state.exhausted]
        if not targets:
            stop = "all_covered" if state.coverage() == 1 else "no_ready_targets"
            break
        for oid in targets:
            if state.covered(oid):
                continue
            q = make_query(state.instantiate(oid))
            res = parse_results(await asyncio.to_thread(retrieve_sync, args.search_url, q))
            new = state.add_items(res, rnd, q)
            entry = {"round": rnd, "obligation": oid, "query": q, "returned": len(res), "new": len(new)}
            if not new:
                state.exhausted.add(oid)
                entry["note"] = "no new evidence"
                log.append(entry)
                continue
            ready = [o for o in state.order if state.ready(o) and not state.covered(o)]
            good, rejected, items, dbg = await map_passages(client, args.model, state, new, ready)
            for m, _ in good:
                state.apply_match(m["obligation"], new[int(m["passage"])], m.get("bindings") or {})
            entry["matched"] = [(m["obligation"], m.get("bindings")) for m, _ in good]
            entry["rejected"] = [(m.get("obligation"), why) for m, why in rejected]
            entry["llm"] = dbg
            entry["passages"] = [(state.items[k].titles, state.items[k].text[:110]) for k in new]
            log.append(entry)
    else:
        if not use_vne and state.coverage() == 1:
            stop = "all_covered"

    vals = state.all_values()
    gold = rec.get("gold") or []
    ans_kind = state.obs[state.answer_id]["kind"]
    gold_hit = None
    if ans_kind == "retrieve":
        vn = [norm(v) for v in vals.values()]
        gold_hit = any(norm(g) and any(norm(g) in v or v in norm(g) for v in vn if v) for g in gold)
    unmet = [o for o in state.order if state.is_retrieve(o) and not state.covered(o)]
    abstain = bool(unmet)          # evidence incomplete -> do NOT guess
    if abstain:
        ans, ans_err = "", None
    else:
        ans, ans_err = await answer_question(client, args.model, state, rec["question"])
    em, f1 = em_f1(ans, gold)
    return {
        "answer": ans, "answer_error": ans_err, "em": em, "f1": f1, "abstain": abstain, "unmet": unmet,
        "mode": "vne" if use_vne else "fillgap", "n_retrievals": sum(1 for e in log if e.get("query")),
        "audit": audit(state, getattr(args, "lam_x", 0.5), getattr(args, "lam_r", 0.5)),
        "question": rec["question"], "gold": gold, "coverage": state.coverage(), "stop": stop,
        "bindings": vals, "gold_in_bindings": gold_hit, "log": log,
        "obligations": [{"id": o, "kind": state.obs[o]["kind"], "text": state.instantiate(o),
                         "covered": state.covered(o), "kappa": state.kappa(o),
                         "titles": sorted({t for k in state.support[o] for t in (state.items[k].titles or [UNK])})}
                        for o in state.order],
        "n_items": len(state.items),
    }


def show(r):
    print("\nQ:", r["question"], "| gold:", str(r["gold"])[:50])
    for e in r["log"]:
        if "action" in e and "query" not in e:
            print(f"  r{e['round']} {e['action']} {e['obligation']} vne={e['vne']} {e.get('note', '')}")
            continue
        print(f"  r{e['round']} {e.get('action', '')} {e['obligation']} q='{e['query'][:70]}' returned={e['returned']} new={e['new']}"
              f" matched={e.get('matched', [])} {e.get('note', '')}")
        if e.get("llm"):
            print(f"     llm: proposed={e['llm']['proposed']} error={e['llm']['error']}")
        if e.get("rejected"):
            print("     rejected:", e["rejected"])
        if VERBOSE and e.get("passages"):
            for t, x in e["passages"]:
                print("     P", t, "|", x)
            print("     RAW:", (e["llm"]["raw_text"] or "")[:500].replace("\n", " "))
    for o in r["obligations"]:
        mark = "OK" if o["covered"] else ("--" if o["kind"] == "retrieve" else "dv")
        print(f"  [{mark}] {o['id']} k={o['kappa']} {o['text']}  {o['titles']}")
    print(f"  coverage={r['coverage']:.2f} stop={r['stop']} bindings={r['bindings']} gold_in_bindings={r['gold_in_bindings']}")


def compact(i, r):
    print(f"[{i}] cov={r['coverage']:.2f} stop={r['stop']} EM={r['em']:.0f} "
          f"ans='{'<ABSTAIN>' if r.get('abstain') else r['answer'][:40]}' "
          f"gold={[str(g)[:25] for g in (r['gold'] or [])[:2]]} | {r['question'][:70]}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obligations", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--search-url", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="trace_mvp.jsonl")
    ap.add_argument("--mode", choices=["fillgap", "vne"], default="fillgap",
                    help="fillgap = old loop (baseline); vne = decision-time controller")
    ap.add_argument("--max-actions", type=int, default=8)
    ap.add_argument("--lam-x", type=float, default=0.5, help="weight of conflict in Risk")
    ap.add_argument("--lam-r", type=float, default=0.5, help="weight of weak support in Risk")
    ap.add_argument("--cost", type=float, default=0.1, help="cost of one retrieval in VNE")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show", action="store_true", help="full per-question trace without raw passages")
    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    recs = [json.loads(l) for l in open(args.obligations, encoding="utf-8")]
    recs = [r for r in recs if r["ok"]][args.start: args.start + args.n]
    prov = ProvenanceIndex.load()
    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")
    try:  # fail fast instead of producing silently empty results
        await client.chat.completions.create(
            model=args.model, max_tokens=5, temperature=0,
            messages=[{"role": "user", "content": "ping"}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    except Exception as e:
        print(f"LLM server unreachable at {args.api_base}: {type(e).__name__}: {e}")
        return

    sem = asyncio.Semaphore(1 if (args.verbose or args.show) else args.concurrency)

    async def one(i, rec):
        async with sem:
            try:
                return i, await solve(client, args, prov, rec)
            except Exception as e:  # keep the batch alive, record the failure
                return i, {"question": rec["question"], "gold": rec.get("gold"), "coverage": 0.0,
                           "stop": f"crash: {type(e).__name__}: {e}"[:120], "answer": "", "answer_error": None,
                           "em": 0.0, "f1": 0.0, "log": [], "n_items": 0, "bindings": {},
                           "gold_in_bindings": None, "obligations": []}

    results = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for fut in asyncio.as_completed([asyncio.create_task(one(i, r)) for i, r in enumerate(recs)]):
            i, r = await fut
            results[i] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if args.verbose or args.show:
                show(r)
            compact(i, r)

    rs = [results[i] for i in sorted(results)]
    n = max(len(rs), 1)
    full = [r for r in rs if r["coverage"] == 1]
    part = [r for r in rs if r["coverage"] < 1]
    avg = lambda xs, k: (sum(x[k] for x in xs) / len(xs)) if xs else float("nan")
    print(f"\n==== {len(rs)} questions | model={args.model}")
    answered = [r for r in rs if not r.get("abstain")]
    print(f"EM={avg(rs,'em'):.3f} F1={avg(rs,'f1'):.3f} (abstain counted as wrong) | avg items={sum(r['n_items'] for r in rs)/n:.1f}")
    print(f"mode={rs[0].get('mode')} | avg retrievals={sum(r.get('n_retrievals', 0) for r in rs)/n:.2f}"
          f" | avg final risk={sum(r.get('audit', {}).get('risk', 0) for r in rs)/n:.3f}")
    print(f"answered: {len(answered)}/{len(rs)} | abstained: {len(rs)-len(answered)} | EM on answered only={avg(answered,'em'):.3f}")
    print(f"fully covered: {len(full)}/{len(rs)}  EM={avg(full,'em'):.3f}  |  not fully covered: {len(part)}  EM={avg(part,'em'):.3f}")
    print("stop reasons:", dict(__import__("collections").Counter(r["stop"].split(":")[0] for r in rs)))
    errs = sum(1 for r in rs for e in r["log"] if (e.get("llm") or {}).get("error")) + sum(1 for r in rs if r.get("answer_error"))
    if errs:
        print(f"!! {errs} LLM call(s) failed - these results are NOT valid")


if __name__ == "__main__":
    asyncio.run(main())
