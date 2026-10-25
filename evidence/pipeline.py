"""One command for the whole evidence experiment, after the hypergraph is built and the /search API is up.

  python3 -m evidence.pipeline --dataset datasets/2WikiMultiHopQA/raw/qa_test.json --n 100 \
      --search-url http://<api-host>:8001/search --api-base http://<llm-host>/v1 --model <name> --out-dir runs/main

Steps (each is skipped when its output already exists, unless --force):
  1. provenance index   (python3 -m evidence.provenance build)
  2. obligations        (python3 -m evidence.obligations)            -> <out-dir>/obligations.jsonl
  3. controller         (python3 -m evidence.controller)             -> <out-dir>/trace_controller.jsonl
  4. ablations          (--ablations: no Diversify, no Verify, scalar risk, no computed comparison)
  5. baseline           (python3 -m evidence.baseline --short)       -> <out-dir>/trace_baseline.jsonl
  6. report             (python3 -m evidence.compare)                -> <out-dir>/report.md
  7. support graphs     (python3 -m evidence.export_graph)           -> <out-dir>/graphs/

The same frozen LLM and the same /search API are used for every system, so the numbers are comparable.
"""
import argparse
import os
import subprocess
import sys

from evidence import provenance
from evidence.common import DATA_DIR, LLM_BASE_URL, LLM_MODEL, SEARCH_URL


def run(args_list, log_path):
    cmd = [sys.executable, "-m"] + args_list
    print("+", " ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"step failed (exit {p.returncode}); see {log_path}", flush=True)
        sys.exit(p.returncode)
    tail = open(log_path, encoding="utf-8").read().strip().splitlines()[-8:]
    print("\n".join("    " + t for t in tail), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="question file, e.g. datasets/2WikiMultiHopQA/raw/qa_test.json")
    ap.add_argument("--key", default="question")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="obligations_*.jsonl whose questions must not be sampled")
    ap.add_argument("--obligations", default="", help="use this existing obligations file instead of building one")
    ap.add_argument("--out-dir", default="runs/main")
    ap.add_argument("--search-url", default=SEARCH_URL)
    ap.add_argument("--api-base", default=LLM_BASE_URL)
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="recompute steps whose output exists")
    args = ap.parse_args()
    if not args.search_url:
        ap.error("--search-url (or EVIDENCE_SEARCH_URL) is required")

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    llm = ["--api-base", args.api_base, "--model", args.model]
    srch = ["--search-url", args.search_url]
    todo = lambda path: args.force or not os.path.exists(path)

    if todo(provenance.INDEX):
        run(["evidence.provenance", "build"], f"{out}/log_provenance.txt")
    else:
        print(f"provenance index exists: {provenance.INDEX}")

    obl = args.obligations or f"{out}/obligations.jsonl"
    if not args.obligations and todo(obl):
        cmd = ["evidence.obligations", "--input", args.dataset, "--key", args.key, "--n", str(args.n),
               "--seed", str(args.seed), "--out", obl] + llm
        if args.exclude:
            cmd += ["--exclude"] + args.exclude
        run(cmd, f"{out}/log_obligations.txt")
    n = sum(1 for _ in open(obl, encoding="utf-8"))
    ctrl = ["evidence.controller", "--obligations", obl, "--n", str(n), "--concurrency", str(args.concurrency)] + srch + llm

    runs = {}
    systems = {"controller": []}
    if args.ablations:
        systems.update({"no_diversify": ["--disable", "Diversify"], "no_verify": ["--disable", "Verify"],
                        "scalar_risk": ["--risk-mode", "scalar"], "llm_compare": ["--no-det-compare"]})
    for name, extra in systems.items():
        trace = f"{out}/trace_{name}.jsonl"
        if todo(trace):
            run(ctrl + extra + ["--out", trace], f"{out}/log_{name}.txt")
        runs[name] = trace
    if not args.skip_baseline:
        trace = f"{out}/trace_baseline.jsonl"
        if todo(trace):
            run(["evidence.baseline", "--obligations", obl, "--n", str(n), "--short",
                 "--concurrency", str(args.concurrency), "--out", trace] + srch + llm, f"{out}/log_baseline.txt")
        runs["baseline"] = trace

    cmp_cmd = ["evidence.compare", "--out", f"{out}/report.md"]
    for k, v in runs.items():
        cmp_cmd += ["--run", f"{k}={v}"]
    run(cmp_cmd, f"{out}/log_report.txt")
    run(["evidence.export_graph", "--trace", runs["controller"], "--obligations", obl, "--out", f"{out}/graphs"],
        f"{out}/log_graphs.txt")
    print(f"\ndone. report: {out}/report.md   graphs: {out}/graphs/   traces: {out}/trace_*.jsonl")


if __name__ == "__main__":
    main()
