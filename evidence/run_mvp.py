"""
MVP loop: Layers 1+2 with a single action (FillGap) and the simple stop rule
"all retrieve-obligations covered". Retrieval = the existing /search API (unchanged).

Run from the project root (inside the container). The endpoints come from the flags or from
EVIDENCE_SEARCH_URL / EVIDENCE_LLM_BASE_URL / EVIDENCE_LLM_MODEL (see evidence/common.py):
  python3 -m evidence.run_mvp --obligations obligations_dev30.jsonl --n 5 \
      --search-url http://<API_IP>:8001/search \
      --api-base http://<LLM_IP>:<PORT>/v1 --model <model-name> --out trace_mvp.jsonl

Per target obligation a round does, in order:
  (a) recheck - show the matcher the items already held that were never checked against this
      obligation (they were retrieved for another obligation while this one was not ready yet);
  (b) query   - only if (a) did not cover it: one new search, new items go to the matcher.
An obligation is "exhausted" only when a query brought nothing new AND nothing is left to recheck.
"""
import argparse
import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter

from openai import AsyncOpenAI

from evidence.common import LLM_BASE_URL, LLM_MODEL, SEARCH_URL
from evidence.metrics import em_f1, normalize_answer  # noqa: F401  (normalize_answer re-exported)
from evidence.obligations import extract_json
from evidence.provenance import ProvenanceIndex, clean_hyperedge, norm
from evidence.state import UNKNOWN as UNK, EvidenceState

PASSAGE_CHARS = 700
MAP_BATCH = 10  # passages per matcher call
VERBOSE = False

MAP_SYSTEM = """You check whether evidence passages support obligations.

For each obligation (a fact to establish) and each numbered passage, decide whether the \
passage EXPLICITLY states that fact. Use only the passage, never outside knowledge.
If it does, return the value of every ?variable in the obligation exactly as written in \
the passage, and copy a short quote (the exact words) that states the fact.
Skip passages that merely mention the entity without stating the fact.

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


def retrieve_sync(url, query, timeout=180, retries=3):
    """POST one query to /search. Network errors and bad JSON are retried with a growing pause."""
    body = json.dumps({"queries": [query]}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (OSError, ValueError, urllib.error.URLError) as e:  # URLError/timeouts are OSError
            last = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last


async def map_passages(client, model, state, new_keys, ready_oids, texts=None):
    """One LLM call: which passages support which ready obligations. Returns validated matches.

    texts: optional {oid: text} that replaces state.instantiate(oid) in the prompt (the controller
    passes state.instantiate(oid, keep_own=True) to look for other values of a fact it already holds).
    """
    items = [state.items[k] for k in new_keys]
    obs = "\n".join(f"{oid}: {(texts or {}).get(oid) or state.instantiate(oid)}" for oid in ready_oids)
    pas = "\n".join(f"[{i}] {it.text[:PASSAGE_CHARS]}" for i, it in enumerate(items))
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
                ok, why = False, f"{m['obligation']} was not requested"
            else:
                ok, why = state.validate_match(m["obligation"], item, b, m.get("quote", ""))
        except Exception as e:
            ok, why = False, f"bad match: {e}"
        (good if ok else rejected).append((m, why))
    return good, rejected, items, dbg


async def map_and_apply(client, model, state, keys, oids, texts=None):
    """Show the items `keys` to the matcher against obligations `oids` and apply the validated matches.

    Returns (matched, rejected, dbg) for the trace. A pair is marked as checked only when the
    LLM call worked, so a failed call is retried in a later round instead of being forgotten.
    """
    matched, rejected, dbgs = [], [], []
    for i in range(0, len(keys), MAP_BATCH):
        batch = keys[i:i + MAP_BATCH]
        good, rej, _, dbg = await map_passages(client, model, state, batch, oids, texts)
        for m, _ in good:
            state.apply_match(m["obligation"], batch[int(m["passage"])], m.get("bindings") or {})
        if not dbg["error"]:
            state.mark_checked(batch, oids)
        matched += [(m["obligation"], m.get("bindings")) for m, _ in good]
        rejected += [(m.get("obligation"), why) for m, why in rej]
        dbgs.append(dbg)
    dbg = {"error": next((d["error"] for d in dbgs if d["error"]), None),
           "raw_text": dbgs[0]["raw_text"] if dbgs else None,
           "proposed": sum(d["proposed"] for d in dbgs)}
    return matched, rejected, dbg


ANSWER_SYSTEM = """Answer the question using ONLY the established facts below.
Compare values when the question asks for it (earlier/later, older/younger, same country).
Give a SHORT answer: a name, date, place, or yes/no. For "which film/person" questions answer \
with the name exactly as written in the question. If a fact is marked NOT FOUND, make your \
best guess from the remaining facts.
Return ONLY JSON: {"answer": "..."}"""


async def answer_question(client, model, state, question):
    lines = []
    for oid in state.order:
        text = state.instantiate(oid)
        if state.is_retrieve(oid) and state.covered(oid):
            best = state.items[state.live_support(oid)[0]]  # live_support is sorted by coherence
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


async def solve(client, args, prov, rec):
    state = EvidenceState(rec["obligations"], prov)
    log, stop = [], "max_rounds"
    for rnd in range(1, args.max_rounds + 1):
        targets = [o for o in state.order if state.is_retrieve(o) and not state.covered(o)
                   and state.ready(o) and o not in state.exhausted]
        if not targets:
            stop = "all_covered" if state.coverage() == 1 else "no_ready_targets"
            break
        for oid in targets:
            if state.covered(oid):
                continue

            # (a) items we already hold but never checked against this obligation
            pending = state.unchecked(oid)
            if pending:
                matched, rejected, dbg = await map_and_apply(client, args.model, state, pending, [oid])
                log.append({"round": rnd, "obligation": oid, "phase": "recheck", "items": len(pending),
                            "matched": matched, "rejected": rejected, "llm": dbg,
                            "passages": [(state.items[k].titles, state.items[k].text[:110]) for k in pending]})
                if state.covered(oid):
                    continue

            # (b) a new query
            q = make_query(state.instantiate(oid))
            res = parse_results(await asyncio.to_thread(retrieve_sync, args.search_url, q))
            state.note_query(oid, q)
            new = state.add_items(res, rnd, q)
            entry = {"round": rnd, "obligation": oid, "phase": "query", "query": q,
                     "returned": len(res), "new": len(new)}
            if not new:
                if not state.unchecked(oid):  # nothing new and nothing left to recheck: really exhausted
                    state.exhausted.add(oid)
                    entry["note"] = "no new evidence"
                log.append(entry)
                continue
            ready = [o for o in state.order if state.ready(o)]
            matched, rejected, dbg = await map_and_apply(client, args.model, state, new, ready)
            entry["matched"] = matched
            entry["rejected"] = rejected
            entry["llm"] = dbg
            entry["passages"] = [(state.items[k].titles, state.items[k].text[:110]) for k in new]
            log.append(entry)
    else:
        if state.coverage() == 1:
            stop = "all_covered"

    vals = state.all_values()
    gold = rec.get("gold") or []
    ans_kind = state.obs[state.answer_id]["kind"]
    gold_hit = None
    if ans_kind == "retrieve":
        vn = [norm(v) for v in vals.values()]
        gold_hit = any(norm(g) and any(norm(g) in v or v in norm(g) for v in vn if v) for g in gold)

    uncovered = [o for o in state.order if state.is_retrieve(o) and not state.covered(o)]
    abstained = bool(uncovered) and args.abstain_incomplete
    if abstained:
        ans, ans_err = "", None
    else:
        ans, ans_err = await answer_question(client, args.model, state, rec["question"])
    em, f1 = em_f1(ans, gold)
    return {
        "answer": ans, "answer_error": ans_err, "em": em, "f1": f1,
        "question": rec["question"], "gold": gold, "coverage": state.coverage(), "stop": stop,
        "uncovered": uncovered, "exhausted": sorted(state.exhausted),
        "abstained": abstained, "answered_from_incomplete": bool(uncovered) and not abstained,
        "bindings": vals, "conflicts": state.conflicts(), "gold_in_bindings": gold_hit, "log": log,
        "obligations": [{"id": o, "kind": state.obs[o]["kind"], "text": state.instantiate(o),
                         "covered": state.covered(o), "kappa": state.kappa(o),
                         "titles": sorted({t for k in state.support[o] for t in (state.items[k].titles or [UNK])})}
                        for o in state.order],
        "n_items": len(state.items),
    }


def show(r):
    print("\nQ:", r["question"], "| gold:", str(r["gold"])[:50])
    for e in r["log"]:
        if e.get("phase") == "recheck":
            print(f"  r{e['round']} {e['obligation']} recheck of {e['items']} held item(s)"
                  f" matched={e.get('matched', [])}")
        else:
            print(f"  r{e['round']} {e['obligation']} q='{e['query'][:70]}' returned={e['returned']} new={e['new']}"
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
    print(f"  coverage={r['coverage']:.2f} stop={r['stop']} bindings={r['bindings']} "
          f"conflicts={r.get('conflicts', {})} gold_in_bindings={r['gold_in_bindings']}")


def compact(i, r):
    tag = " ABSTAIN" if r.get("abstained") else (" GUESS" if r.get("answered_from_incomplete") else "")
    print(f"[{i}] cov={r['coverage']:.2f} stop={r['stop']}{tag} EM={r['em']:.0f} ans='{r['answer'][:40]}' "
          f"gold={[str(g)[:25] for g in (r['gold'] or [])[:2]]} | {r['question'][:70]}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obligations", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--search-url", default=SEARCH_URL, help="Graph-R1 /search API (or EVIDENCE_SEARCH_URL)")
    ap.add_argument("--api-base", default=LLM_BASE_URL, help="LLM server (or EVIDENCE_LLM_BASE_URL)")
    ap.add_argument("--model", default=LLM_MODEL, help="LLM name (or EVIDENCE_LLM_MODEL)")
    ap.add_argument("--abstain-incomplete", action="store_true",
                    help="return an empty answer instead of a best guess when an obligation is not covered")
    ap.add_argument("--out", default="trace_mvp.jsonl")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show", action="store_true", help="full per-question trace without raw passages")
    args = ap.parse_args()
    if not args.search_url:
        ap.error("--search-url (or EVIDENCE_SEARCH_URL) is required")
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
                           "em": 0.0, "f1": 0.0, "log": [], "n_items": 0, "bindings": {}, "conflicts": {},
                           "uncovered": [], "exhausted": [], "abstained": False, "answered_from_incomplete": False,
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
    print(f"EM={avg(rs,'em'):.3f} F1={avg(rs,'f1'):.3f} | avg items={sum(r['n_items'] for r in rs)/n:.1f}")
    print(f"fully covered: {len(full)}/{len(rs)}  EM={avg(full,'em'):.3f}  |  not fully covered: {len(part)}  EM={avg(part,'em'):.3f}")
    print(f"not fully covered: abstained={sum(bool(r.get('abstained')) for r in part)} "
          f"answered-by-guess={sum(bool(r.get('answered_from_incomplete')) for r in part)}")
    print("stop reasons:", dict(Counter(r["stop"].split(":")[0] for r in rs)))
    errs = sum(1 for r in rs for e in r["log"] if (e.get("llm") or {}).get("error")) + sum(1 for r in rs if r.get("answer_error"))
    if errs:
        print(f"!! {errs} LLM call(s) failed - these results are NOT valid")


if __name__ == "__main__":
    asyncio.run(main())
