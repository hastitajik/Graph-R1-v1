"""Save the Evidence Support Graph of each question of a controller trace (it only lives in memory during a run).

Nodes: obligations (o1..), the provenance units (Wikipedia articles) that support them.
Edges: obligation -> obligation (depends_on), unit -> obligation (supports; the bound value is the label).

  python3 -m evidence.export_graph --trace trace_ctrl_v2.jsonl --obligations obligations_bonsai30.jsonl --out graphs/
writes graphs/q<i>.graphml, graphs/q<i>.dot and, if matplotlib is installed, graphs/q<i>.png
"""
import argparse
import json
import os

import networkx as nx

from evidence.common import VAR


def build(trace_rec, ob_rec):
    G = nx.DiGraph(question=trace_rec["question"], answer=str(trace_rec.get("answer", "")),
                   certified=bool(trace_rec.get("certified")))
    cert = trace_rec.get("certificate", {})
    cut = set(cert.get("unaudited", []))
    obs = {o["id"]: o for o in ob_rec["obligations"]["obligations"]}
    tobs = {o["id"]: o for o in trace_rec["obligations"]}
    for oid, o in obs.items():
        t = tobs.get(oid, {})
        label = t.get("text") or o["text"]
        G.add_node(oid, kind="obligation", label=f"{oid}: {label}", covered=bool(t.get("covered")) or o["kind"] != "retrieve",
                   retrieve=o["kind"] == "retrieve", critical_cut=oid in cut)
        for d in o.get("depends_on", []):
            G.add_edge(d, oid, rel="depends_on")
    for oid, t in tobs.items():
        for title in t.get("titles", []):
            u = f"unit:{title}"
            G.add_node(u, kind="unit", label=title)
            G.add_edge(u, oid, rel="supports")
    return G


def to_dot(G):
    lines = ["digraph G {", "  rankdir=LR; node [fontsize=10];"]
    for n, d in G.nodes(data=True):
        shape = "box" if d["kind"] == "obligation" else "ellipse"
        color = "black" if d.get("covered", True) else "red"
        lab = d["label"].replace('"', "'")
        lines.append(f'  "{n}" [label="{lab}", shape={shape}, color={color}];')
    for a, b, d in G.edges(data=True):
        style = "solid" if d["rel"] == "depends_on" else "dashed"
        lines.append(f'  "{a}" -> "{b}" [style={style}];')
    lines.append("}")
    return "\n".join(lines)


def draw(G, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ob = [n for n, d in G.nodes(data=True) if d["kind"] == "obligation"]
    un = [n for n, d in G.nodes(data=True) if d["kind"] == "unit"]
    pos = {}
    for i, n in enumerate(ob):
        pos[n] = (i * 3.0, 1.0)
    for i, n in enumerate(un):
        sup = [pos[o][0] for o in G.successors(n) if o in pos]
        pos[n] = ((sum(sup) / len(sup)) if sup else i * 3.0, -0.6 - 0.5 * (i % 2))
    fig, ax = plt.subplots(figsize=(max(7, 3 * len(ob)), 4.5))
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, node_size=2500,
                           edge_color=["#444" if d["rel"] == "depends_on" else "#888" for *_, d in G.edges(data=True)],
                           style=["solid" if d["rel"] == "depends_on" else "dashed" for *_, d in G.edges(data=True)])
    nx.draw_networkx_nodes(G, pos, nodelist=ob, node_shape="s", node_size=2500, ax=ax,
                           node_color=["#cfe8cf" if G.nodes[n]["covered"] else "#f4c7c3" for n in ob])
    nx.draw_networkx_nodes(G, pos, nodelist=un, node_size=1800, node_color="#dde6f5", ax=ax)
    wrap = lambda s: "\n".join(s[i:i + 26] for i in range(0, len(s), 26))
    nx.draw_networkx_labels(G, pos, labels={n: wrap(G.nodes[n]["label"]) for n in G}, font_size=6, ax=ax)
    ax.set_title(("CERTIFIED  " if G.graph["certified"] else "NOT certified  ") + G.graph["question"][:90] +
                 f"\nanswer: {G.graph['answer'][:60]}", fontsize=8)
    ax.axis("off")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--obligations", required=True)
    ap.add_argument("--out", default="graphs")
    args = ap.parse_args()
    by_q = {}
    for l in open(args.obligations, encoding="utf-8"):
        r = json.loads(l)
        if r.get("ok"):
            by_q[r["question"]] = r
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for i, l in enumerate(open(args.trace, encoding="utf-8")):
        t = json.loads(l)
        ob = by_q.get(t["question"])
        if not ob or not t.get("obligations"):
            continue
        G = build(t, ob)
        base = os.path.join(args.out, f"q{i:02d}")
        nx.write_graphml(G, base + ".graphml")
        open(base + ".dot", "w", encoding="utf-8").write(to_dot(G))
        try:
            draw(G, base + ".png")
        except ImportError:
            pass
        n += 1
    print(f"{n} graphs written to {args.out}/")


if __name__ == "__main__":
    main()
