import requests
from openai import OpenAI

client = OpenAI(base_url="http://192.168.4.114:5577/v1", api_key="EMPTY")

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

def search(q):
    r = requests.post("http://localhost:8001/search", json={"queries": [q]}, timeout=120)
    r.raise_for_status()
    return str(r.json())[:3000]

question = "Which magazine came out first, Tit-Bits or Illustreret Nyhedsblad?"
msgs = [{"role": "user", "content": INSTR + "Question: " + question}]

for turn in range(6):
    out = client.chat.completions.create(
        model="Qwen3.8-27B", messages=msgs, temperature=0, max_tokens=1024,
        stop=["</query>"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).choices[0].message.content or ""
    print(f"\n===== TURN {turn} RAW =====\n{out}")
    if "<answer>" in out:
        print("\n>>> FINAL ANSWER FOUND")
        break
    if "<query>" in out:
        q = out.split("<query>")[-1].strip()
        print(f"\n>>> SEARCHING: {q!r}")
        info = search(q)
        msgs += [
            {"role": "assistant", "content": out + "</query>"},
            {"role": "user", "content": f"<information>\n{info}\n</information>"},
        ]
        continue
    print("\n>>> NO <query> OR <answer> TAG IN OUTPUT")
    break
