# Evidence-Certificate GraphRAG on top of Graph-R1

A training-free layer on top of the Graph-R1 knowledge hypergraph. No RL policy is trained: a frozen LLM plus a
decision-time controller. The answer comes with an Evidence Certificate that says what supports it.

## Architecture

```
question
  -> obligations      evidence/obligations.py   Evidence Obligations O(q): retrieve / derive steps with ?variables
  -> state            evidence/state.py         evidence items, bindings, support per obligation, guards
  -> audit            evidence/audit.py         coverage C, robustness R, conflicts X, provenance independence D,
                                                critical cuts, counterfactual fragility, Pareto vector, scalar Risk
  -> controller       evidence/controller.py    actions FillGap, Bridge, Diversify, Challenge, Verify, Answer
                                                chosen by VNE = p * gain - cost
  -> answer + certificate (evidence/derive.py computes date comparisons from the certified dates)
```

- Provenance unit = Wikipedia article title (`evidence/provenance.py`); an unknown source counts once, never as independent.
- Guards against wrong evidence (all in `evidence/state.py`): quote must occur in the passage, binding must occur in the
  passage, type check (date vs place), subject check, self-reference check, alias-aware conflicts.
- A certificate is issued only if the audit passes AND the final answer is grounded in the certified evidence.
- Model confidence is not used in the stop rule.
- Risk(s) = (1 - C) + lam_x * X + lam_r * (1 - R), hand-set weights (`--risk-mode scalar --lam-x --lam-r`).
  The default judges a step by the Pareto vector (C, R, D, -X). A search that discovers a conflicting value counts
  as a successful step.

## Full rerun from scratch

1. Build the hypergraph and start the retrieval API exactly as in `README.md` (`script_process.py`, `script_build.py`,
   `script_api.py`). The LLM used for graph construction is read from `GRAPHR1_LLM_BASE_URL`
   (default `http://localhost:8000/v1`).
2. Evidence image (adds `openai<3`, `networkx`, `matplotlib` to the Graph-R1 image):
   `docker build -f Dockerfile.evidence -t graphr1:evidence-v1 .`
3. Endpoints (flags win over variables):
   ```
   EVIDENCE_LLM_BASE_URL   OpenAI-compatible server of the frozen LLM
   EVIDENCE_LLM_MODEL      model name served there
   EVIDENCE_SEARCH_URL     http://<api-host>:8001/search
   EVIDENCE_DATA_DIR       expr/2WikiMultiHopQA (graphml, chunk store, provenance index)
   ```
4. Everything in one command (provenance index, obligations, controller, baseline, report, graphs):
   ```
   python3 -m evidence.pipeline --dataset datasets/2WikiMultiHopQA/raw/qa_test.json --n 100 \
       --search-url http://<api-host>:8001/search --api-base http://<llm-host>/v1 --model <name> \
       --out-dir runs/main --ablations
   ```
   Add `--exclude obligations_dev30.jsonl obligations_bonsai30.jsonl` to sample a held-out set that does not
   overlap the questions the guards were tuned on.

Outputs in `--out-dir`: `obligations.jsonl`, `trace_<system>.jsonl`, `report.md`, `graphs/` (graphml, dot, png of the
evidence support graph of each question), one `log_<step>.txt` per step.

## Single steps

```
python3 -m evidence.provenance build
python3 -m evidence.obligations --input <qa.json> --n 100 --out obligations.jsonl
python3 -m evidence.controller --obligations obligations.jsonl --n 100 --search-url ... --api-base ... --model ...
python3 -m evidence.baseline   --obligations obligations.jsonl --n 100 --short --search-url ... --api-base ... --model ...
python3 -m evidence.compare --run controller=trace_ctrl.jsonl --run baseline=trace_base.jsonl --out report.md
python3 -m evidence.export_graph --trace trace_ctrl.jsonl --obligations obligations.jsonl --out graphs/
python3 -m evidence.diag --trace trace_ctrl.jsonl      # why did a failed question fail: retrieval or mapping
python3 -m unittest discover -s tests -t .
```

Controller switches: `--disable Diversify,Verify`, `--learn-p` (use with `--concurrency 1`), `--risk-mode scalar`,
`--no-det-compare`, `--abstain-uncertified`, weights `--w-gap --w-conflict --w-fragile --w-c-search ...`.

## What the report shows

EM / F1, searches and LLM calls per system; for controller runs the share of certified answers, EM on certified
and uncertified answers, and the selective EM (answer only when certified). The baseline is the Graph-R1 agent loop
(think / query / answer) with the same frozen LLM and the same retrieval API, without the RL-trained policy.
Numbers from different LLMs are not comparable; use one model for all systems of a table.

## Known limits

- The certificate covers retrieved facts. The Verify step cannot catch a passage that is about a different entity
  with the same name; the final comparison is computed only for dates, other derive steps are an LLM step.
- Independent second sources are rare in 2WikiMultiHopQA (a fact lives in one article), so D stays near 0 there.
- Retrieval misses on the first hop cannot be repaired by the controller.
