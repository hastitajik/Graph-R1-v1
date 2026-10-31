# Evidence-Certificate GraphRAG on top of Graph-R1

A **training-free** layer on top of [Graph-R1](https://arxiv.org/abs/2507.21892) (agentic GraphRAG over a knowledge
hypergraph). Graph-R1 trains an LLM with reinforcement learning to decide what to retrieve. This project keeps the
LLM **frozen** and the retrieval **untouched**, and instead adds a decision-time controller that audits the evidence
and attaches an **Evidence Certificate** to every answer.

> Status: research prototype. Evaluated so far on 100 held-out 2WikiMultiHopQA questions with a small development
> LLM. See [Results](#results-so-far) for what is and is not shown.

## The idea in one paragraph

Multi-hop questions fail silently: the model answers confidently from a wrong or single-source passage. Here, the
question is first turned into **evidence obligations** (the facts that must be found, with `?variables` linking
hops). Each retrieved passage is checked by **guards** (quote really in the passage, value really in the passage,
right type such as date vs place, right entity, no self-reference). The accepted evidence forms an **Evidence Support
Graph**; an **audit** measures coverage, robustness (independent sources, critical cuts, counterfactual fragility),
conflicts and provenance independence. A **controller** picks the next action (`FillGap`, `Bridge`, `Diversify`,
`Challenge`, `Verify`, `Answer`) by `VNE = p * gain - cost`, and stops when the audit passes. The final answer carries
a **certificate**: issued only if every required obligation is covered, the graph is connected, no conflict is
unresolved, critical facts were verified, **and** the answer itself is grounded in that evidence. Model confidence is
never used.

```
question
  -> obligations   evidence/obligations.py   what must be found (retrieve / derive steps)
  -> state         evidence/state.py         accepted evidence + guards
  -> audit         evidence/audit.py         coverage, robustness, conflicts, provenance, cuts, certificate
  -> controller    evidence/controller.py    FillGap / Bridge / Diversify / Challenge / Verify / Answer
  -> answer + certificate   (evidence/derive.py computes date comparisons from certified dates)
```

Full design, plan conformance table, assumptions and known limits: **[EVIDENCE.md](EVIDENCE.md)**.
Method section for the paper (LaTeX): **[paper/evidence_certificate.tex](paper/evidence_certificate.tex)**.

## Results so far

100 held-out 2WikiMultiHopQA questions, one frozen development LLM for all systems, same retrieval API.

| system | EM (95% CI) | F1 | searches / question |
|---|---|---|---|
| Graph-R1 agent loop, frozen LLM (baseline) | 0.700 [0.604, 0.781] | 0.768 | 3.72 |
| Evidence controller (ours) | 0.630 [0.532, 0.718] | 0.663 | 5.82 |

Paired difference is not significant (baseline 12 wins, controller 5, McNemar p = 0.143).

| certificate | share | EM |
|---|---|---|
| certified answers | 67 / 100 | 0.851 |
| uncertified answers | 33 / 100 | 0.182 |
| selective EM (answer only if certified, count the rest wrong) | | 0.570 |

How to read this honestly:

- **Raw accuracy did not improve**; it is lower, though not significantly.
- **The certificate separates answers well**: 85% correct when issued, 18% when not. This is the contribution being
  tested: knowing *which* answers to trust.
- The baseline has a simple signal of its own (answer found in retrieved text): EM 0.780 on 82 questions. The
  certificate is slightly more precise (0.851) at lower coverage; this difference is not tested as significant.
- 10 of the 67 certified answers are wrong (mostly place / country / date of death); not yet diagnosed.
- Not done yet: other benchmarks (HotpotQA, MuSiQue), a larger final LLM, comparison with the RL-trained Graph-R1.
- Ablations (13 systems) are in the generated `report.md`; Verify is the only component that clearly matters here.

## Quick start

Requires a running Graph-R1 retrieval API (`script_api.py`, see [README_GRAPH_R1.md](README_GRAPH_R1.md)) and an
OpenAI-compatible server for the frozen LLM.

```bash
docker build -f Dockerfile.evidence -t graphr1:evidence-v1 .

export EVIDENCE_LLM_BASE_URL=http://<llm-host>/v1
export EVIDENCE_LLM_MODEL=<model>
export EVIDENCE_SEARCH_URL=http://<api-host>:8001/search
export EVIDENCE_DATA_DIR=expr/2WikiMultiHopQA

# everything in one command: obligations, controller, baseline, statistics, graphs, ablations
python3 -m evidence.pipeline --dataset datasets/2WikiMultiHopQA/raw/qa_test.json --n 100 \
    --out-dir runs/main --ablations
```

Outputs in `runs/main/`: `report.md` (EM/F1, certificate tables, paired tests, ablations), `trace_<system>.jsonl`,
`graphs/` (support graph of each question).

Tests (no GPU, no network): `python3 -m unittest discover -s tests -t .`

## Repository map

| path | what |
|---|---|
| `evidence/` | **this project**: obligations, state and guards, audit and certificate, controller, baseline, statistics, pipeline |
| `tests/` | offline tests for the evidence layer |
| `EVIDENCE.md` | architecture, flags, plan conformance, assumptions, known limits |
| `paper/` | LaTeX method section |
| `Dockerfile.evidence` | Graph-R1 image plus the few extra dependencies |
| `graphr1/`, `agent/`, `verl/`, `script_*.py`, `run_*.sh`, `evaluation/`, `inference/` | original Graph-R1 code, unchanged |
| `README_GRAPH_R1.md` | original Graph-R1 README (graph building, retrieval server, RL training) |

## Limits to know before reusing

- Provenance, type cues and name aliases assume English Wikipedia-style text (2WikiMultiHopQA). On another corpus
  they weaken silently; adapt `evidence/provenance.py`, `state.py`, `derive.py`.
- Independent second sources are rare in 2WikiMultiHopQA, so Diversify, Bridge and Challenge show no benefit there.
- The certificate covers retrieved facts; it cannot detect a passage about a different entity with the same name.

## Credits

Built on Graph-R1 (Luo et al., ICML 2026): <https://arxiv.org/abs/2507.21892>. Original citation and acknowledgements
are kept in [README_GRAPH_R1.md](README_GRAPH_R1.md). License: see `LICENSE`.
