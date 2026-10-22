"""
Diagnose failed questions from a trace (no LLM calls, only the /search API).

For every question with EM=0 it prints the obligation chain (covered or not), the bindings,
and then re-queries the retrieval API with a few query forms to answer ONE question:
is the gold answer retrievable at all (retrieval miss) or was it retrieved but not used
(mapping miss)? Yes/no questions have no gold span, so they are only summarized.

  python3 -m evidence.diag --trace trace_bonsai30.jsonl --search-url http://<IP>:8001/search
"""
import argparse
import json

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
    ap.add_argument("--search-url", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.trace, encoding="utf-8")]
    for i, r in enumerate(recs):
        if r["em"] == 1:
            continue
        golds = [str(g) for g in (r["gold"] or [])]
        head = f"[{i}] cov={r['coverage']:.2f} stop={r['stop']} ans='{r['answer'][:40]}' gold={golds[:2]}"
        if any(g.lower() in ("yes", "no") for g in golds):
            print(head, "| yes/no (no span to check) |", r["question"][:60])
            continue
        print("\n" + head)
        print("   Q:", r["question"])
        for o in r["obligations"]:
            mark = "OK" if o["covered"] else ("--" if o["kind"] == "retrieve" else "dv")
            print(f"   [{mark}] {o['id']} k={o['kappa']} {o['text']}  {o['titles']}")
        print("   bindings:", r["bindings"])
        qs = [(f"{o['id']}:obligation", make_query(o["text"]))
              for o in r["obligations"] if o["kind"] == "retrieve" and not o["covered"]]
        qs.append(("whole question", r["question"]))
        for label, q in qs:
            try:
                res = parse_results(retrieve_sync(args.search_url, q))
            except Exception as e:
                print(f"   {label}: search failed: {e}")
                continue
            h = first_hit(res, golds)
            print(f"   {label:18s} q='{q[:55]}' n={len(res)} gold_in_results="
                  + (f"rank {h[0]} | {h[1]}" if h else "NO"))


if __name__ == "__main__":
    main()
