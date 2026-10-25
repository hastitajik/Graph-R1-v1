"""
Diagnose failed questions from a trace (no LLM calls, only the /search API).

For every question with EM=0 it prints the obligation chain (covered or not), the bindings,
and then re-queries the retrieval API with a few query forms to answer ONE question:
is the gold answer retrievable at all (retrieval miss) or was it retrieved but not used
(mapping miss)? Yes/no questions have no gold span, so they are only summarized.

  python3 -m evidence.diag --trace trace_bonsai30.jsonl --search-url http://<IP>:8001/search
(--search-url defaults to EVIDENCE_SEARCH_URL)
"""
import argparse
import json

from evidence.common import SEARCH_URL
from evidence.provenance import norm
from evidence.run_mvp import make_query, parse_results, retrieve_sync


def first_hit(results, golds):
    for rank, x in enumerate(results, 1):
        tn = norm(x["text"])
        if any(len(norm(g)) >= 3 and norm(g) in tn for g in golds):
            return rank, x["text"][:100]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--search-url", default=SEARCH_URL)
    args = ap.parse_args()
    if not args.search_url:
        ap.error("--search-url (or EVIDENCE_SEARCH_URL) is required")
    recs = [json.loads(l) for l in open(args.trace, encoding="utf-8")]
    cats = {"yes_no": [], "wrong_but_covered": [], "mapping_miss": [], "retrieved_only_by_whole_question": [],
            "retrieval_miss": [], "search_failed": []}
    for i, r in enumerate(recs):
        if r["em"] == 1:
            continue
        golds = [str(g) for g in (r["gold"] or [])]
        head = f"[{i}] cov={r['coverage']:.2f} stop={r['stop']} ans='{r['answer'][:40]}' gold={golds[:2]}"
        if any(g.lower() in ("yes", "no") for g in golds):
            print(head, "| yes/no (no span to check) |", r["question"][:60])
            cats["yes_no"].append(i)
            continue
        print("\n" + head)
        print("   Q:", r["question"])
        for o in r["obligations"]:
            mark = "OK" if o["covered"] else ("--" if o["kind"] == "retrieve" else "dv")
            print(f"   [{mark}] {o['id']} k={o['kappa']} {o['text']}  {o['titles']}")
        print("   bindings:", r["bindings"])
        why = [f"{o}: {w}" for st in r.get("steps", []) for o, w in (st.get("rejected") or [])]
        if why:
            print("   mapper rejections:", "; ".join(dict.fromkeys(why))[:600])
        qs = [(f"{o['id']}:obligation", make_query(o["text"]))
              for o in r["obligations"] if o["kind"] == "retrieve" and not o["covered"]]
        qs.append(("whole question", r["question"]))
        hits = {}
        for label, q in qs:
            try:
                res = parse_results(retrieve_sync(args.search_url, q))
            except Exception as e:
                print(f"   {label}: search failed: {e}")
                hits[label] = None
                continue
            h = first_hit(res, golds)
            hits[label] = bool(h)
            print(f"   {label:18s} q='{q[:55]}' n={len(res)} gold_in_results="
                  + (f"rank {h[0]} | {h[1]}" if h else "NO"))
        if r["coverage"] >= 1.0:
            cats["wrong_but_covered"].append(i)      # every obligation was covered, the answer is still wrong
        elif None in hits.values():
            cats["search_failed"].append(i)
        elif any(v for k, v in hits.items() if k.endswith(":obligation")):
            cats["mapping_miss"].append(i)           # the gold text WAS retrieved by an obligation query
        elif hits.get("whole question"):
            cats["retrieved_only_by_whole_question"].append(i)
        else:
            cats["retrieval_miss"].append(i)
    print("\n==== failure categories (questions with EM=0)")
    for k, v in cats.items():
        print(f"{k:34s} {len(v):3d}  {v}")


if __name__ == "__main__":
    main()
