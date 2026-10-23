"""
Layer 1 - Question -> Evidence Obligations O(q)

One LLM call (frozen Qwen3.8-27B, temperature 0, thinking off), strict JSON output,
validated, with retries. Run once per question, at the start of the loop.

Usage (from the project root, inside the graphr1 container; the server and model come from
EVIDENCE_LLM_BASE_URL / EVIDENCE_LLM_MODEL or --api-base / --model, see evidence/common.py):
  python3 -m evidence.obligations --input datasets/2WikiMultiHopQA/raw/qa_dev.json \
      --n 30 --seed 0 --out obligations_dev30.jsonl
"""
import argparse
import asyncio
import json
import random
import re
from collections import Counter

from openai import AsyncOpenAI

from evidence.common import LLM_BASE_URL as BASE_URL, LLM_MODEL as MODEL, VAR

MAX_OBLIGATIONS = 6
MAX_ATTEMPTS = 3
VALID_TYPES = {"compositional", "comparison", "inference", "bridge_comparison", "other"}
VALID_KINDS = {"retrieve", "derive"}

SYSTEM_PROMPT = """You decompose a multi-hop question into EVIDENCE OBLIGATIONS.

An obligation is one atomic factual statement that must be supported by a document \
for the final answer to be valid. Do NOT answer the question. Do NOT invent facts.

Rules:
- Each obligation has: "id" (o1, o2, ...), "text", "depends_on" (list of ids), "kind".
- "text" is a self-contained statement. Use ?X-style variables (?D, ?M, ?T1) for unknown \
values. Copy entity names exactly as written in the question, including any parenthesized qualifier such as (Film) or (1989 Film).
- "depends_on" lists the obligations whose resolved value this one needs.
- kind = "retrieve": must be found in the corpus.
- kind = "derive": computed from other obligations' values (compare dates, same country, \
older/younger). It needs no retrieval.
- Every ?variable is introduced by exactly ONE obligation. An obligation that uses a variable \
introduced by another obligation must list that obligation in "depends_on". Two different \
unknowns must have different names (?D1, ?D2), never the same name.
- Use the fewest obligations possible (maximum 6). One fact = one obligation.
- "answer_id" is the obligation that produces the final answer.
- "type" is one of: compositional, comparison, inference, bridge_comparison, other.

Return ONLY a JSON object, no markdown, no commentary.

Example 1
Question: Who is the mother of the director of film Blue River?
{"type":"compositional","obligations":[
{"id":"o1","text":"The director of the film Blue River is ?D","depends_on":[],"kind":"retrieve"},
{"id":"o2","text":"The mother of ?D is ?M","depends_on":["o1"],"kind":"retrieve"}],
"answer_id":"o2"}

Example 2
Question: Which film came out first, Blue River or Red Lake?
{"type":"comparison","obligations":[
{"id":"o1","text":"Blue River was released on ?T1","depends_on":[],"kind":"retrieve"},
{"id":"o2","text":"Red Lake was released on ?T2","depends_on":[],"kind":"retrieve"},
{"id":"o3","text":"Compare ?T1 and ?T2 to find the earlier film","depends_on":["o1","o2"],"kind":"derive"}],
"answer_id":"o3"}

Example 3
Question: Who is the maternal grandfather of Anna Lee?
{"type":"inference","obligations":[
{"id":"o1","text":"The mother of Anna Lee is ?M","depends_on":[],"kind":"retrieve"},
{"id":"o2","text":"The father of ?M is ?F","depends_on":["o1"],"kind":"retrieve"}],
"answer_id":"o2"}

Example 4
Question: Which film has the director who died first, Blue River or Red Lake?
{"type":"bridge_comparison","obligations":[
{"id":"o1","text":"The director of Blue River is ?D1","depends_on":[],"kind":"retrieve"},
{"id":"o2","text":"The director of Red Lake is ?D2","depends_on":[],"kind":"retrieve"},
{"id":"o3","text":"?D1 died on ?T1","depends_on":["o1"],"kind":"retrieve"},
{"id":"o4","text":"?D2 died on ?T2","depends_on":["o2"],"kind":"retrieve"},
{"id":"o5","text":"Compare ?T1 and ?T2 to find which director died first, then name the film","depends_on":["o3","o4"],"kind":"derive"}],
"answer_id":"o5"}
"""


def extract_json(text: str):
    """Pull the first JSON object out of a model reply (tolerates code fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def validate(obj: dict, question: str = None) -> list:
    """Return a list of problems; empty list means valid."""
    errs = []
    if not isinstance(obj, dict):
        return ["output is not an object"]
    obs = obj.get("obligations")
    if not isinstance(obs, list) or not obs:
        return ["obligations missing or empty"]
    if len(obs) > MAX_OBLIGATIONS:
        errs.append(f"too many obligations ({len(obs)} > {MAX_OBLIGATIONS})")
    if obj.get("type") not in VALID_TYPES:
        errs.append(f"bad type: {obj.get('type')}")

    ids = [o.get("id") for o in obs if isinstance(o, dict)]
    if len(ids) != len(obs) or len(set(ids)) != len(ids):
        errs.append("ids missing or duplicated")
        return errs
    for o in obs:
        if not isinstance(o.get("text"), str) or not o["text"].strip():
            errs.append(f"{o['id']}: empty text")
        if o.get("kind") not in VALID_KINDS:
            errs.append(f"{o['id']}: bad kind {o.get('kind')}")
        deps = o.get("depends_on")
        if not isinstance(deps, list):
            errs.append(f"{o['id']}: depends_on not a list")
            continue
        for d in deps:
            if d not in ids:
                errs.append(f"{o['id']}: unknown dependency {d}")
            if d == o["id"]:
                errs.append(f"{o['id']}: depends on itself")

    if question:
        for span in re.findall(r"\([^)]*\)", question):
            if not any(span in str(o.get("text", "")) for o in obs):
                errs.append(f"qualifier {span} from the question is missing in the obligations")

    if obj.get("answer_id") not in ids:
        errs.append("answer_id not in obligations")
    if not any(o.get("kind") == "retrieve" for o in obs):
        errs.append("no retrieve obligation")

    # cycle check (DFS)
    graph = {o["id"]: [d for d in o.get("depends_on", []) if d in ids] for o in obs}
    state = {}

    def has_cycle(n):
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        if any(has_cycle(m) for m in graph[n]):
            return True
        state[n] = 2
        return False

    if any(has_cycle(n) for n in graph):
        errs.append("dependency cycle")

    # Variable discipline. EvidenceState keeps ONE global namespace of variables and only
    # inherits a value from a DIRECT dependency, so a variable must have a single "root"
    # obligation (where it is introduced); every other obligation that mentions it has to
    # depend directly on an obligation that mentions it too. Otherwise two unrelated unknowns
    # that share a name get merged, or a value is silently not inherited.
    mentions = {}
    for o in obs:
        for v in dict.fromkeys(VAR.findall(str(o.get("text", "")))):
            mentions.setdefault(v, []).append(o["id"])
    deps_of = {o["id"]: set(o["depends_on"]) for o in obs if isinstance(o.get("depends_on"), list)}
    for v, ms in mentions.items():
        roots = [i for i in ms if i in deps_of and not (deps_of[i] & set(ms))]
        if len(roots) > 1:
            errs.append(
                f"variable {v} is introduced independently in {', '.join(roots)}: give the "
                f"unknowns different names, or add the missing depends_on")
    return errs


async def decompose(client: AsyncOpenAI, question: str, model: str = MODEL) -> dict:
    """Returns {"ok", "obligations", "attempts", "errors"}."""
    last_errs = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        user = f"Question: {question}"
        if last_errs:
            user += "\n\nYour previous answer was invalid: " + "; ".join(last_errs) + ". Fix it."
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            obj = extract_json(resp.choices[0].message.content)
            last_errs = validate(obj, question)
            if not last_errs:
                return {"ok": True, "obligations": obj, "attempts": attempt, "errors": []}
        except Exception as e:  # network, parse, etc.
            last_errs = [f"{type(e).__name__}: {e}"]
    return {"ok": False, "obligations": None, "attempts": MAX_ATTEMPTS, "errors": last_errs}


def load_questions(path: str, key: str):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(data, dict):
        data = list(data.values())
    if not data or key not in data[0]:
        raise SystemExit(f"key '{key}' not found. Available keys: {list(data[0].keys())}")
    return data


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--key", default="question")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="obligations_sample.jsonl")
    ap.add_argument("--api-base", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    data = load_questions(args.input, args.key)
    random.Random(args.seed).shuffle(data)
    data = data[: args.n]

    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")
    sem = asyncio.Semaphore(args.concurrency)

    async def run(item):
        async with sem:
            res = await decompose(client, item[args.key], args.model)
        return {"question": item[args.key], "gold": item.get("golden_answers", item.get("answer")), **res}

    results = await asyncio.gather(*(run(x) for x in data))

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if r["ok"]]
    print(f"valid: {len(ok)}/{len(results)}  |  needed retry: {sum(r['attempts'] > 1 for r in ok)}")
    print("types:", dict(Counter(r["obligations"]["type"] for r in ok)))
    if ok:
        avg = sum(len(r["obligations"]["obligations"]) for r in ok) / len(ok)
        print(f"avg obligations: {avg:.2f}")
    for r in results:
        if not r["ok"]:
            print("FAILED:", r["question"], "|", r["errors"])
    print(f"saved -> {args.out}  (review these by hand)")


if __name__ == "__main__":
    asyncio.run(main())
