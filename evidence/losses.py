"""Questions where the two systems disagree, side by side (no LLM, no search).

  python3 -m evidence.losses --a runs/main/trace_baseline.jsonl --b runs/main/trace_controller.jsonl
  (A right and B wrong first; --both also lists B right and A wrong)

For each such question it prints the gold answer, both answers, and what the controller knew: certified or not, stop
reason, coverage, the reasons the certificate was refused and the actions it took. Read these lines and put every
question in ONE group: wrong obligations, evidence not retrieved, wrong value bound, answer format, or other.
"""
import argparse
import json


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["question"]] = r
    return out


def show(q, a, b):
    print("=" * 100)
    print("Q:", q)
    print("gold:", a.get("gold") or b.get("gold"))
    print(f"A answer: {a.get('answer')!r}   em={a.get('em')}   grounded={a.get('grounded')}")
    cert = b.get("certificate") or {}
    print(f"B answer: {b.get('answer')!r}   em={b.get('em')}   certified={b.get('certified')}   stop={b.get('stop')}"
          f"   coverage={b.get('coverage')}")
    if cert.get("reasons"):
        print("   certificate reasons:", cert["reasons"])
    steps = b.get("steps") or b.get("trace") or []
    if steps:
        print("   actions:", " > ".join(str(s.get("action")) for s in steps if isinstance(s, dict)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reference system (usually the baseline trace)")
    ap.add_argument("--b", required=True, help="the system under study (usually the controller trace)")
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    common = [q for q in A if q in B]
    a_only = [q for q in common if A[q]["em"] >= 1 and B[q]["em"] < 1]
    b_only = [q for q in common if B[q]["em"] >= 1 and A[q]["em"] < 1]
    print(f"{len(common)} common questions: A right and B wrong = {len(a_only)}, B right and A wrong = {len(b_only)}")
    for q in a_only:
        show(q, A[q], B[q])
    if args.both:
        print("\n\n##### B right, A wrong #####")
        for q in b_only:
            show(q, A[q], B[q])


if __name__ == "__main__":
    main()
