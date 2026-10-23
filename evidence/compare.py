"""Question-by-question comparison of two traces, using the repo's official eval.py.

  python3 -m evidence.compare trace_v1.jsonl trace_v2.jsonl
"""
import json
import sys

from eval import cal_em, cal_f1


def load(path):
    return {r["question"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8"))}


def score(r):
    return cal_em([r["gold"] or []], [r["answer"]])


def main(pa, pb):
    a, b = load(pa), load(pb)
    qs = [q for q in a if q in b]
    print(f"common questions: {len(qs)} (v1 {len(a)}, v2 {len(b)})")
    for name, d in (("v1", a), ("v2", b)):
        g = [d[q]["gold"] or [] for q in qs]
        p = [d[q]["answer"] for q in qs]
        ab = sum(bool(d[q].get("abstain")) for q in qs)
        print(f"{name}: EM={cal_em(g, p):.3f} F1={cal_f1(g, p):.3f} abstained={ab}")
    gained = [q for q in qs if score(a[q]) < 1 <= score(b[q])]
    lost = [q for q in qs if score(a[q]) >= 1 > score(b[q])]
    wrong_to_abstain = [q for q in qs if score(a[q]) < 1 and b[q].get("abstain") and not a[q].get("abstain")]
    print(f"\nGAINED ({len(gained)}):")
    for q in gained: print("  +", q[:90], "->", b[q]["answer"])
    print(f"\nLOST ({len(lost)})  <- must be empty or explained:")
    for q in lost: print("  -", q[:90], "|", a[q]["answer"], "->", b[q]["answer"] or "<ABSTAIN>")
    print(f"\nWRONG GUESS -> ABSTAIN ({len(wrong_to_abstain)}):")
    for q in wrong_to_abstain: print("  ~", q[:90], "| was:", a[q]["answer"][:30])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
