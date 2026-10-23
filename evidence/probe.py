"""Prompt ablation for ONE obligation: which prompt variant lets the mapper find the evidence?

Runs the same retrieved passages through several prompt variants and prints the raw LLM reply
of each, so you see which change breaks (or fixes) the match.

  python3 -m evidence.probe --obligations obligations_bonsai30.jsonl --index 1 --obligation o2 \
      --search-url http://$IP:8001/search --api-base http://192.168.4.150:8000/v1 --model Qwen3.8-27B

--index = position among the valid records (same as --start in run_mvp).
The obligation must not depend on other obligations (use a first-hop one such as o1/o2).
"""
import argparse
import asyncio
import json

from openai import AsyncOpenAI

from evidence import run_mvp
from evidence.obligations import extract_json
from evidence.provenance import ProvenanceIndex
from evidence.state import EvidenceState

V1_SYSTEM = """You check whether evidence passages support obligations.

For each obligation (a fact to establish) and each numbered passage, decide whether the \
passage EXPLICITLY states that fact. Use only the passage, never outside knowledge.
If it does, return the value of every ?variable in the obligation exactly as written in \
the passage, and copy a short quote (the exact words) that states the fact.
Skip passages that merely mention the entity without stating the fact.

Return ONLY a JSON object:
{"matches":[{"passage":0,"obligation":"o1","bindings":{"?D":"value"},"quote":"exact words"}]}
Use {"matches":[]} if nothing is supported."""

# name, system prompt, article label in the passage line, passage chars
VARIANTS = [
    ("current          (new prompt, label, 1500)", run_mvp.MAP_SYSTEM, True, 1500),
    ("no_label         (new prompt, no label, 1500)", run_mvp.MAP_SYSTEM, False, 1500),
    ("chars700         (new prompt, label, 700)", run_mvp.MAP_SYSTEM, True, 700),
    ("v1_exact         (OLD prompt, no label, 700)", V1_SYSTEM, False, 700),
    ("v1+label+1500    (OLD prompt, label, 1500)", V1_SYSTEM, True, 1500),
]


def build_user(state, oid, items, label, chars):
    pas = "\n".join(
        f"[{i}] " + (f"(article: {', '.join(it.titles) or 'unknown'}) " if label else "") + it.text[:chars]
        for i, it in enumerate(items))
    return f"Obligations:\n{oid}: {state.instantiate(oid)}\n\nPassages:\n{pas}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obligations", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--obligation", required=True)
    ap.add_argument("--search-url", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    recs = [r for r in (json.loads(l) for l in open(a.obligations, encoding="utf-8")) if r["ok"]]
    rec = recs[a.index]
    state = EvidenceState(rec["obligations"], ProvenanceIndex.load())
    oid = a.obligation
    if not state.ready(oid) or state.dep_vars(oid):
        raise SystemExit(f"{oid} depends on other obligations; pick a first-hop one")
    query = run_mvp.make_query(state.instantiate(oid))
    res = run_mvp.parse_results(run_mvp.retrieve_sync(a.search_url, query))
    new = state.add_items(res, 1, query)
    items = [state.items[k] for k in new]
    print("Q:", rec["question"], "\nobligation:", state.instantiate(oid), "| passages:", len(items))

    client = AsyncOpenAI(base_url=a.api_base, api_key="dummy")
    for name, system, label, chars in VARIANTS:
        resp = await client.chat.completions.create(
            model=a.model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": build_user(state, oid, items, label, chars)}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        raw = resp.choices[0].message.content or ""
        try:
            n = len(extract_json(raw).get("matches", []))
        except Exception:
            n = "PARSE-ERROR"
        print(f"\n[{name}] matches={n}\n   {raw[:300]!r}")


if __name__ == "__main__":
    asyncio.run(main())
