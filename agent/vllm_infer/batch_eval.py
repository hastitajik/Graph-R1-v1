import os
import json, re, os, argparse, requests
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--api-base", default=os.environ.get("GRAPHR1_LLM_BASE_URL", "http://localhost:8000/v1"))
ap.add_argument("--model", default="Qwen3.8-27B")
ap.add_argument("--data", default="datasets/2WikiMultiHopQA/raw/qa_test.json")
ap.add_argument("--search-url", default="http://localhost:8001/search")
ap.add_argument("--out", default="expr_results/Qwen3.8-27B_2WikiMultiHopQA_agent")
ap.add_argument("--max-turns", type=int, default=5)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--short", action="store_true")
args = ap.parse_args()

client = OpenAI(api_key="EMPTY", base_url=args.api_base)
TOOL = {"name": "search", "description": "Search for information on the internet using Wikipedia as a knowledge source.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}
HEAD = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" + json.dumps(TOOL) +
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\n"
        "Answer the given question. You can query from knowledge base provided to you to answer the question. You can query knowledge as many times as you want.\n"
        "You must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>.\n"
        "When you have the final answer, you can output the answer inside <answer>...</answer>.\n\n"
        "Output format for tool call:\n<think>\n...\n</think>\n<query>\n...\n</query>\n\n"
        "Output format for answer:\n<think>\n...\n</think>\n<answer>\n...\n</answer>\nQuestion: ")
SHORT = "\nKeep the text inside <answer> as short as possible (an entity, date, yes/no, or a few words), not a full sentence."
TAIL = "<|im_end|>\n<|im_start|>assistant\n"

def search(q):
    try:
        out = requests.post(args.search_url, json={"queries": [q]}, timeout=180).json()[0]
        res = json.loads(out)["results"]
        return "\n".join(x["<knowledge>"] for x in res)
    except Exception as e:
        return f"[search error: {e}]"

def gen(prompt, stops):
    r = client.completions.create(model=args.model, prompt=prompt, temperature=0, max_tokens=1024, stop=stops)
    c = r.choices[0]
    txt = c.text or ""
    sr = getattr(c, "stop_reason", None)
    if sr in stops and sr.startswith("</"):
        txt += sr
    return txt.replace("<thinking>", "<think>").replace("</thinking>", "</think>")

def run_one(item):
    base = HEAD + item["question"] + (SHORT if args.short else "") + TAIL
    pred, turns, ok = "", 0, False
    try:
        for _ in range(args.max_turns):
            turns += 1
            txt = gen(base + pred, ["</query>", "</answer>", "</tool_call>", "<knowledge>", "<information>"])
            pred += txt
            if "<answer>" in txt and "</answer>" not in txt:
                pred += "\n</answer>"
                ok = True
                break
            if "</answer>" in txt:
                ok = True
                break
            m = re.findall(r"<query>\s*(.*?)\s*</query>", txt, re.DOTALL) or re.findall(r'"query"\s*:\s*"(.*?)"', txt, re.DOTALL)
            if not m:
                break
            q = m[-1]
            try:
                q = json.loads(q)["query"]
            except Exception:
                pass
            pred += "\n\n<knowledge>\n" + search(q) + "\n</knowledge>\n\n"
        if not ok:  # اجبار به جواب نهایی
            pred += "\n<think>\nI must give my final answer now.\n</think>\n<answer>\n"
            pred += gen(base + pred, ["</answer>"])
            if not pred.rstrip().endswith("</answer>"):
                pred += "</answer>"
    except Exception as e:
        pred += f"\n<answer>\n</answer>\n[ERROR {e}]"
    return {"question": item["question"], "golden_answers": item["golden_answers"], "context": item["context"],
            "prediction": pred, "format_score": 1.0 if ok else 0.0, "turns": turns}

data = json.load(open(args.data, encoding="utf-8"))
if args.limit:
    data = data[:args.limit]
with ThreadPoolExecutor(args.workers) as ex:
    results = list(tqdm(ex.map(run_one, data), total=len(data)))
os.makedirs(args.out, exist_ok=True)
json.dump(results, open(os.path.join(args.out, "results_step0.json"), "w"), indent=2, ensure_ascii=False)
json.dump({"f1": 0.0}, open(os.path.join(args.out, "evals_step0.json"), "w"))
print("saved", len(results), "->", args.out, "| answered:", sum(r["format_score"] for r in results))
