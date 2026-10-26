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
- A certificate is issued only if the audit passes AND the final answer is grounded: a retrieve answer must be the value of the answer obligation itself (a bridge value does not count), a derive answer must be computed, or name an option of the question, or be yes/no (`answer_basis` in the trace says which; only `retrieved` and `computed` check the last step against the evidence).
- Independence: passages that share an article title are one source (union-find); an UNKNOWN passage is independent of nothing; an answer that depends on no retrieve obligation can never be certified.
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
3b. After a graph rebuild delete `$EVIDENCE_DATA_DIR/provenance_index.json` (the pipeline reuses an existing one) or pass `--force`.
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

Controller switches: `--disable Diversify,Verify`, `--no-learn-p` (p per question instead of carried over the run; with `--concurrency > 1` the learned p makes a run not bit-reproducible), `--risk-mode scalar`,
`--no-det-compare`, `--abstain-uncertified`, weights `--w-gap --w-conflict --w-fragile --w-c-search ...`.

## What the report shows

EM / F1, searches and LLM calls per system; for controller runs the share of certified answers, EM on certified
and uncertified answers, and the selective EM (answer only when certified). The baseline is the Graph-R1 agent loop
(think / query / answer) with the same frozen LLM and the same retrieval API, without the RL-trained policy.
Numbers from different LLMs are not comparable; use one model for all systems of a table.

## Conformance with the plan (Evidence-Certificate GraphRAG)

| plan element | status | where |
|---|---|---|
| Evidence Obligations O(q) from the frozen LLM | done | `obligations.py` (validated: ids, dependencies, cycles) |
| Evidence Support Graph G_E(q, a) | done, can be exported | `audit.build_graph`, `export_graph.py` |
| kappa = provenance-independent support paths, unit = article, unknown source counts once | done | `state.unit`, `state.kappa` |
| evidence cuts, counterfactual fragility, conflicts X, provenance diversity D, robustness R, coverage C | done | `audit.py`, all in the certificate |
| actions FillGap, Bridge, Diversify, Challenge, Verify, Answer | done (Answer = the stop rule) | `controller.py` |
| step reward r = dC + alpha dR + beta dD - gamma dX - lam cost | computed and logged for every step (`reward` in the trace, mean per action in the summary); the controller itself is training-free, so it is a diagnostic and not a learning signal | `controller.execute` |
| Pareto evidence state | done (default); scalar Risk = (1-C) + lam_x X + lam_r (1-R) as an option | `audit.risk`, `--risk-mode` |
| VNE(a|s) = E[Risk(s) - Risk(s')] - Cost(a) | approximated: p (success rate per action kind, optionally learned across questions) times a hand-set gain per situation (uncovered obligation, conflict, single source, unverified cut) minus cost. The gains mirror the terms of the Risk but are not computed as an exact expected Risk difference | `Weights`, `Episode.price` |
| Reasoning Pathology Monitor | retrieval loop, over-retrieval, confirmation bias, premature confidence, ungrounded answer. The ungrounded reasoning leap is checked on the final answer, not on every reasoning edge | `Episode.monitor`, `solve` |
| STOP: core obligations covered, graph connected, no unresolved critical conflict, critical cuts audited, max VNE <= cost | done; model confidence is not used | `Episode.run`, `audit` |
| Answer only with a certificate | the default answers anyway and flags the answer as uncertified, so that EM can be reported both ways; `--abstain-uncertified` refuses to answer, and `report.md` shows the selective EM | `solve`, `compare.py` |
| frozen retrieval (entity + hyperedge dual path, rank fusion) | untouched, used through `/search` | Graph-R1 |
| comparison with the Graph-R1 agent loop | done, same LLM and API | `baseline.py` |
| ablations | no Diversify / Verify / Bridge / Challenge, FillGap only, no fragility term, guards off, naive provenance (every passage independent), scalar risk, LLM comparison | `pipeline.py --ablations` |
| informed query (repair of a missed first hop) | when the plain queries of an obligation are used up: one blind paraphrase, then up to two LLM queries that read titles and openings of the passages retrieved so far | `controller.informed_queries`, `Episode.fillgap_mode` |
| entity type in matching | the matcher sees the question and skips same-named entities of another kind; obligations must keep the type word (film, album, village ...) | `run_mvp.MAP_SYSTEM`, `obligations.validate` |
| statistics | Wilson CI of EM, paired wins / losses, exact McNemar, bootstrap CI of the difference, selective prediction by certified / coverage / grounded | `compare.py` |

Not implemented: a confidence / answer-stability stop rule as a contrast baseline (the plan removes model
confidence from the controller), and the learned classifier for sufficiency that was mentioned as future work.

## Assumptions (what is specific to 2WikiMultiHopQA / English / Wikipedia)

- Provenance unit = article title read from the graph's chunk store (`provenance.py`). On a corpus with another chunk layout every passage is UNKNOWN; the controller prints a warning when more than half of the supporting passages have no title.
- Type cues (`born on`, `died in`, ...), month names, name aliases and the `TYPE_WORDS` list are English and Wikipedia-style (`state.py`, `derive.py`, `obligations.py`). On another language or dataset they weaken guards silently instead of failing; adapt them there.
- The deterministic comparison only fires for `A or B` questions about two dates.

The method section for the paper is `paper/evidence_certificate.tex` (compiles standalone).

## Known limits

- The certificate covers retrieved facts. The Verify step cannot catch a passage that is about a different entity
  with the same name; the final comparison is computed only for dates, other derive steps are an LLM step.
- Independent second sources are rare in 2WikiMultiHopQA (a fact lives in one article), so D stays near 0 there.
- A first-hop miss is only repaired when the informed queries find the right title; if the corpus never returns it, the question stays uncertified.
