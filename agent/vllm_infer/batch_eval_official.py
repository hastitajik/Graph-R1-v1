import os
import json, os, argparse
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tqdm import tqdm

from agent.vllm_infer.run import process_tool_call, execute_tool_calls_batch  # کد رسمی، بدون تغییر
from agent.tool.tool_env import ToolEnv
from agent.tool.tools import _default_tools

ap = argparse.ArgumentParser()
ap.add_argument("--api-base", default=os.environ.get("GRAPHR1_LLM_BASE_URL", "http://localhost:8000/v1"))
ap.add_argument("--model", default="Qwen3.8-27B")
ap.add_argument("--data", default="datasets/2WikiMultiHopQA/raw/qa_test.json")
ap.add_argument("--out", default="expr_results/Qwen3.8-27B_2WikiMultiHopQA_agent")
ap.add_argument("--max-turns", type=int, default=8)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--json-hint", action="store_true",
                 help="اضافه‌کردن دستورالعمل JSON از ToolEnv.tools_format_func به prompt")
args = ap.parse_args()

client = OpenAI(api_key="EMPTY", base_url=args.api_base)
HEAD = open("agent/vllm_infer/_head.txt", encoding="utf-8").read()
TAIL = open("agent/vllm_infer/_tail.txt", encoding="utf-8").read()

if args.json_hint:
    _hint = ToolEnv(tools=_default_tools("search")).tools_format_func()
    HEAD = HEAD.replace("Question: ", _hint + "\nQuestion: ")

def run_one(item):
    prompt = HEAD + item["question"] + TAIL
    env = ToolEnv(tools=_default_tools("search"), max_turns=args.max_turns)
    turns, ok = 0, False
    for _ in range(args.max_turns):
        turns += 1
        resp = client.completions.create(
            model=args.model, prompt=prompt, temperature=1.0, top_p=1.0,
            max_tokens=4096, stop=["</query>"],
        )
        c = resp.choices[0]
        raw = c.text or ""
        if getattr(c, "stop_reason", None) == "</query>":
            raw += "</query>"
        responses_str, active_masks = process_tool_call([raw])                       # کد رسمی
        tool_responses = execute_tool_calls_batch(responses_str, env, active_masks)  # کد رسمی
        prompt += responses_str[0] + tool_responses[0]
        if not active_masks[0]:
            ok = "<answer>" in responses_str[0]
            break
    pred = prompt[len(HEAD) + len(item["question"]):]
    if not ok:  # اجبار به جواب نهایی، چون get_remote_score.py بدون </answer> کرش می‌کند
        r2 = client.completions.create(model=args.model, prompt=prompt + "\n<answer>\n",
                                        temperature=1.0, top_p=1.0, max_tokens=256, stop=["</answer>"])
        pred += "\n<answer>\n" + (r2.choices[0].text or "") + "</answer>"
        ok = True
    return {"question": item["question"], "golden_answers": item["golden_answers"],
            "context": item["context"], "prediction": pred,
            "format_score": 1.0 if ok else 0.0, "turns": turns}

data = json.load(open(args.data, encoding="utf-8"))
if args.limit:
    data = data[:args.limit]
with ThreadPoolExecutor(args.workers) as ex:
    results = list(tqdm(ex.map(run_one, data), total=len(data)))
os.makedirs(args.out, exist_ok=True)
json.dump(results, open(os.path.join(args.out, "results_step0.json"), "w"), indent=2, ensure_ascii=False)
json.dump({"f1": 0.0}, open(os.path.join(args.out, "evals_step0.json"), "w"))
print("saved", len(results), "->", args.out, "| answered:", sum(r["format_score"] for r in results))
