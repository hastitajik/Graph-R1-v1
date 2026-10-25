"""Graph-R1 style agent baseline on the SAME questions, search API and metric as the controller.

The loop is the one of Graph-R1 inference (<think> ... <query> ... </query> -> <knowledge> ... -> <answer>),
run with the same frozen LLM through the OpenAI-compatible chat endpoint. No RL-trained policy.

Run (project root):
  python3 -m evidence.baseline --obligations obligations_bonsai30.jsonl --n 30 \
      --search-url http://<ip>:8001/search --api-base http://<llm>/v1 --model <name> --out trace_base.jsonl
"""
import argparse
import asyncio
import json
import re

from openai import AsyncOpenAI

from evidence.common import LLM_BASE_URL, LLM_MODEL, SEARCH_URL
from evidence.metrics import em_f1
from evidence.run_mvp import parse_results, retrieve_sync

INSTR = """Answer the given question. You can query from knowledge base provided to you to answer the question. You can query knowledge as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<query>
...
</query>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""
SHORT = "\nKeep the text inside <answer> as short as possible (an entity, date, yes/no, or a few words), not a full sentence."


async def gen(client, args, msgs, stop):
    r = await client.chat.completions.create(
        model=args.model, messages=msgs, temperature=0, max_tokens=1024, stop=stop,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content or ""


async def run_one(client, args, rec):
    msgs = [{"role": "user", "content": INSTR + "Question: " + rec["question"] + (SHORT if args.short else "")}]
    searches, turns, answer, err, trail = 0, 0, "", None, []
    try:
        for _ in range(args.max_turns):
            turns += 1
            out = await gen(client, args, msgs, ["</query>"])
            trail.append(out)
            m = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", out, re.DOTALL)
            if m:
                answer = m.group(1).strip()
                break
            if "<query>" not in out:
                break
            q = out.split("<query>")[-1].strip()
            try:
                res = await asyncio.to_thread(retrieve_sync, args.search_url, q)
                info = "\n".join(x["text"] for x in parse_results(res))
            except Exception as e:
                info = f"[search error: {e}]"
            searches += 1
            msgs += [{"role": "assistant", "content": out + "</query>"},
                     {"role": "user", "content": f"<information>\n{info[:args.info_chars]}\n</information>"}]
        if not answer:  # forced final answer after the turn budget
            msgs.append({"role": "user", "content": "You must give your final answer now, inside <answer>...</answer>."})
            out = await gen(client, args, msgs, None)
            trail.append(out)
            m = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", out, re.DOTALL)
            answer = (m.group(1) if m else out).strip()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:200]
    gold = rec.get("gold") or []
    em, f1 = em_f1(answer, gold)
    return {"question": rec["question"], "gold": gold, "answer": answer, "em": em, "f1": f1,
            "searches": searches, "turns": turns, "error": err, "trail": trail}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obligations", required=True, help="same jsonl as the controller: fixes the question set")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--info-chars", type=int, default=6000)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--short", action="store_true", help="ask for a short answer (fairer EM)")
    ap.add_argument("--search-url", default=SEARCH_URL)
    ap.add_argument("--api-base", default=LLM_BASE_URL)
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--out", default="trace_base.jsonl")
    args = ap.parse_args()
    if not args.search_url:
        ap.error("--search-url (or EVIDENCE_SEARCH_URL) is required")
    recs = [json.loads(l) for l in open(args.obligations, encoding="utf-8")]
    recs = [r for r in recs if r["ok"]][args.start: args.start + args.n]
    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i, rec):
        async with sem:
            return i, await run_one(client, args, rec)

    res = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for fut in asyncio.as_completed([asyncio.create_task(one(i, r)) for i, r in enumerate(recs)]):
            i, r = await fut
            res[i] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}] EM={r['em']:.0f} s={r['searches']} turns={r['turns']} ans={r['answer']!r} "
                  f"gold={r['gold'][:2]} | {r['question'][:60]}" + (f" !! {r['error']}" if r["error"] else ""))
    rs = [res[i] for i in sorted(res)]
    n = len(rs)
    print(f"\n==== BASELINE {n} questions | model={args.model}")
    print(f"EM={sum(r['em'] for r in rs)/n:.3f} F1={sum(r['f1'] for r in rs)/n:.3f} | "
          f"avg searches={sum(r['searches'] for r in rs)/n:.1f} avg turns={sum(r['turns'] for r in rs)/n:.1f}")
    print(f"no answer: {sum(1 for r in rs if not r['answer'])} | errors: {sum(1 for r in rs if r['error'])}")


if __name__ == "__main__":
    asyncio.run(main())
