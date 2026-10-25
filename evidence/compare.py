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


def fmt(x):
    return "-" if x is None or x != x else f"{x:.3f}" if isinstance(x, float) else str(x)


def report(runs):
    rows = [summarize(n, rs) for n, rs in runs.items()]
    out = ["# Results", "", "| system | n | EM | F1 | searches | LLM calls / turns |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['n']} | {fmt(r['em'])} | {fmt(r['f1'])} | {fmt(r['searches'])} | {fmt(r['llm'])} |")
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
    names = list(runs)
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
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = load(path)
    text = report(runs)
    print(text)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
