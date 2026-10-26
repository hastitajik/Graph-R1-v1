"""Side-by-side report of controller runs and baseline runs on the same question set.

  python3 -m evidence.compare --run controller=trace_ctrl.jsonl --run no_diversify=trace_nodiv.jsonl \
      --run baseline=trace_base_short.jsonl --out report.md

Every trace is one json object per question. A trace with a "certified" field is a controller run, any other
is a baseline run. Prints (and writes) a markdown report:
  * overall EM / F1, searches, LLM calls per system
  * for controller runs: certified coverage, EM on certified / uncertified answers, and the selective EM
    (answer only certified questions; an abstained question counts as wrong) with its answered precision
  * per question: which systems got it right
"""
import argparse
import json
import math
import random


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(name, rs):
    row = {"name": name, "n": len(rs), "em": mean(r["em"] for r in rs), "f1": mean(r["f1"] for r in rs),
           "searches": mean(r.get("searches", 0) for r in rs), "llm": mean(r.get("llm_calls", r.get("turns", 0)) for r in rs)}
    if rs and "certified" in rs[0]:
        cert = [r for r in rs if r["certified"]]
        unc = [r for r in rs if not r["certified"]]
        row.update(coverage=len(cert) / len(rs), em_cert=mean(r["em"] for r in cert),
                   em_unc=mean(r["em"] for r in unc), selective_em=sum(r["em"] for r in cert) / len(rs),
                   n_cert=len(cert))
    elif rs and "grounded" in rs[0]:
        g = [r for r in rs if r["grounded"]]
        row.update(grounded=len(g), em_grounded=mean(r["em"] for r in g))
    return row


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion k/n."""
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact p-value for b wins vs c losses (ties ignored)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(ref, other, boots=2000, seed=0):
    """Per-question comparison on the questions both runs answered: wins / losses of `other` against `ref`,
    exact McNemar p and a bootstrap 95% interval of the EM difference (other - ref)."""
    def keyed(rs):      # a question that occurs twice is matched occurrence by occurrence
        seen, out = {}, {}
        for r in rs:
            n = seen[r["question"]] = seen.get(r["question"], 0) + 1
            out[(r["question"], n)] = r["em"]
        return out

    a, b = keyed(ref), keyed(other)
    pairs = [(a[k], b[k]) for k in b if k in a]
    if not pairs:
        return None
    wins = sum(1 for x, y in pairs if y > x)
    losses = sum(1 for x, y in pairs if y < x)
    rng = random.Random(seed)
    diffs = sorted(sum(y - x for x, y in (rng.choice(pairs) for _ in pairs)) / len(pairs) for _ in range(boots))
    return {"n": len(pairs), "wins": wins, "losses": losses, "p": mcnemar_exact(wins, losses),
            "diff": sum(y - x for x, y in pairs) / len(pairs),
            "lo": diffs[int(0.025 * boots)], "hi": diffs[int(0.975 * boots) - 1]}


def fmt(x):
    return "-" if x is None or x != x else f"{x:.3f}" if isinstance(x, float) else str(x)


def report(runs, ref=None, rejected=False):
    rows = [summarize(n, rs) for n, rs in runs.items()]
    out = ["# Results", "", "| system | n | EM | EM 95% CI | F1 | searches | LLM calls / turns |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = wilson(round(r["em"] * r["n"]), r["n"])
        out.append(f"| {r['name']} | {r['n']} | {fmt(r['em'])} | [{fmt(lo)}, {fmt(hi)}] | {fmt(r['f1'])} | "
                   f"{fmt(r['searches'])} | {fmt(r['llm'])} |")
    ctrl = [r for r in rows if "coverage" in r]
    if ctrl:
        out += ["", "## Certificates (controller runs)", "",
                "| system | certified | coverage | EM certified | EM uncertified | selective EM (abstain on uncertified) |",
                "|---|---|---|---|---|---|"]
        for r in ctrl:
            out.append(f"| {r['name']} | {r['n_cert']}/{r['n']} | {fmt(r['coverage'])} | {fmt(r['em_cert'])} | "
                       f"{fmt(r['em_unc'])} | {fmt(r['selective_em'])} |")
    base = [r for r in rows if "grounded" in r]
    if base:
        out += ["", "## Baselines: answer found in retrieved text", "", "| system | grounded | EM grounded |", "|---|---|---|"]
        for r in base:
            out.append(f"| {r['name']} | {r['grounded']}/{r['n']} | {fmt(r['em_grounded'])} |")
    # selective prediction: which signal tells right answers from wrong ones
    sel = [(n_, rs) for n_, rs in runs.items() if rs and ("certified" in rs[0] or "grounded" in rs[0])]
    if sel:
        out += ["", "## Selective prediction (answer only when the signal holds)", "",
                "| system | signal | answered | EM when answered | EM when not |", "|---|---|---|---|---|"]
        for n_, rs in sel:
            sigs = []
            if "certified" in rs[0]:
                sigs += [("certified", lambda r: r["certified"])]
            if "coverage" in rs[0]:
                sigs += [("coverage = 1", lambda r: r["coverage"] >= 1.0)]
            if "grounded" in rs[0]:
                sigs += [("answer found in retrieved text", lambda r: r["grounded"])]
            for label, f in sigs:
                yes = [r for r in rs if f(r)]
                no = [r for r in rs if not f(r)]
                out.append(f"| {n_} | {label} | {len(yes)}/{len(rs)} | {fmt(mean(r['em'] for r in yes))} | "
                           f"{fmt(mean(r['em'] for r in no))} |")
    names = list(runs)
    ref = ref or names[0]
    if len(runs) > 1 and ref in runs:
        out += ["", f"## Paired comparison against `{ref}` (EM; + = better than {ref})", "",
                "| system | questions | wins | losses | EM diff | 95% bootstrap CI | McNemar p |", "|---|---|---|---|---|---|---|"]
        for n_ in names:
            if n_ == ref:
                continue
            c = paired(runs[ref], runs[n_])
            if c:
                out.append(f"| {n_} | {c['n']} | {c['wins']} | {c['losses']} | {c['diff']:+.3f} | "
                           f"[{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p']:.3f} |")
        out.append("")
        out.append("A difference whose interval contains 0 (or p > 0.05) cannot be told from noise at this sample size.")
    if rejected:
        out += ["", "## Verify rejections (check by hand: a wrongly rejected fact lowers recall)", ""]
        for n_, rs in runs.items():
            for r in rs:
                for st in r.get("steps", []):
                    if "verify_rejected" in st:
                        v = st["verify_rejected"]
                        out.append(f"- {n_} | EM={int(r['em'])} | {r['question'][:60]} | fact: {v['fact']} | passage: {v['passage'][:140]}")
    if len({len(v) for v in runs.values()}) > 1:
        out += ["", "WARNING: the runs have different numbers of questions " +
                str({k: len(v) for k, v in runs.items()}) + "; the per-question table is cut to the shortest."]
    n = min(len(v) for v in runs.values())
    keyed = {k: {r["question"]: r for r in v} for k, v in runs.items()}
    qs = [r["question"] for r in next(iter(runs.values()))][:n]
    out += ["", "## Per question (1 = exact match)", "", "| # | " + " | ".join(names) + " | question |",
            "|---|" + "---|" * len(names) + "---|"]
    for i, q in enumerate(qs):
        cells = []
        for k in names:
            r = keyed[k].get(q)
            cells.append("?" if r is None else f"{int(r['em'])}" + ("c" if r.get("certified") else ""))
        out.append(f"| {i} | " + " | ".join(cells) + f" | {q[:70]} |")
    out += ["", "c = certified answer."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=TRACE.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--ref", default="", help="run that the paired comparison uses as reference (default: the first)")
    ap.add_argument("--rejected", action="store_true", help="list the facts Verify rejected")
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = load(path)
    text = report(runs, args.ref, args.rejected)
    print(text)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
