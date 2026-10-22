"""
Provenance index for Layer 2.

The retrieval API returns hyperedge text only (no source). A hyperedge node in the
graphml has `source_id` = chunk id(s). A chunk holds many Wikipedia articles, each
introduced by a title line like:  "Rozstanie"
So the independence unit is the ARTICLE TITLE. Matching, in order:
  1. exact   : the (normalized) hyperedge text is inside one titled segment of its chunk
  2. fuzzy   : the hyperedge was rewritten by the LLM (pronouns resolved, facts merged),
               so pick the segment with the best word overlap, +bonus if its title
               appears in the hyperedge text; accept only with a clear margin
  3. unknown : no title (kept as [] - treat as ONE shared unknown source, never independent)
Duplicated articles in different docs collapse to the same title.

Build once (a few minutes):
  python evidence/provenance.py build
Inspect:
  python evidence/provenance.py lookup '<hyperedge text>'

Use in code:
  from evidence.provenance import ProvenanceIndex
  idx = ProvenanceIndex.load()
  idx.titles(knowledge_text)   # -> ["Rozstanie"]  ([] if unknown)
  idx.key(knowledge_text)      # dedup key for evidence items
"""
import argparse
import json
import random
import re
from collections import Counter

DATA_DIR = "expr/2WikiMultiHopQA"
GRAPHML = f"{DATA_DIR}/graph_chunk_entity_relation.graphml"
CHUNKS = f"{DATA_DIR}/kv_store_text_chunks.json"
INDEX = f"{DATA_DIR}/provenance_index.json"
SEP = "<SEP>"  # GRAPH_FIELD_SEP (confirmed in graphr1/prompt.py)

TITLE_LINE = re.compile(r'(?m)^"([^"\n]+)"$')
MIN_LEN = 15
FUZZY_MIN = 0.6      # minimum score to accept a fuzzy match
FUZZY_MARGIN = 0.1   # best must beat the best DIFFERENT title by this much
TITLE_BONUS = 0.4
VALIDATE_RATE = 0.05  # share of exact matches re-checked with the fuzzy method


def norm(s: str) -> str:
    """Lowercase and drop everything that is not a letter/digit (unicode-aware)."""
    return re.sub(r"[\W_]+", "", s.lower())


def tokens(s: str) -> set:
    return {w for w in re.findall(r"\w+", s.lower()) if len(w) >= 4}


def clean_hyperedge(text: str) -> str:
    text = text.strip()
    if text.startswith("<hyperedge>"):
        text = text[len("<hyperedge>"):]
    return text.strip().strip('"')


def title_norm(title: str) -> str:
    """Normalized title without a trailing parenthetical, e.g. 'Chandrika (Film)' -> 'chandrika'."""
    return norm(re.sub(r"\s*\([^)]*\)\s*$", "", title))


def split_segments(content: str):
    """[(title or None, normalized_body, token_set)] for one chunk."""
    parts = TITLE_LINE.split(content)  # [pre, title1, body1, title2, body2, ...]
    segs = [(None, norm(parts[0]), tokens(parts[0]))]
    for i in range(1, len(parts), 2):
        body = parts[i + 1] if i + 1 < len(parts) else ""
        segs.append((parts[i].strip(), norm(body), tokens(body)))
    return [s for s in segs if s[1] or s[0]]


def chunk_segments(chunks: dict):
    """chunk_id -> segments. A leading untitled segment inherits the last title of the
    previous chunk of the same doc (chunk_order_index - 1)."""
    by_pos = {(v.get("full_doc_id"), int(v.get("chunk_order_index", -1))): cid for cid, v in chunks.items()}
    segs = {cid: split_segments(v["content"]) for cid, v in chunks.items()}

    def prev_of(cid):
        v = chunks[cid]
        return by_pos.get((v.get("full_doc_id"), int(v.get("chunk_order_index", -1)) - 1))

    def last_title(cid, depth=0):
        for t, _, _ in reversed(segs[cid]):
            if t:
                return t
        p = prev_of(cid)
        return last_title(p, depth + 1) if p and depth < 5 else None

    out = {}
    for cid in chunks:
        s = segs[cid]
        if s and s[0][0] is None:
            p = prev_of(cid)
            s = [(last_title(p) if p else None, s[0][1], s[0][2])] + s[1:]
        out[cid] = s
    return out


def find_exact(h_norm: str, segs):
    probes = [h_norm]
    if len(h_norm) > 60:
        probes += [h_norm[:50], h_norm[-50:]]
    for probe in probes:
        for title, body, _ in segs:
            if title and probe in body:
                return title
    return None


def find_fuzzy(h_text: str, h_norm: str, segs):
    """Returns (title, score) or (None, best_score)."""
    ht = tokens(h_text)
    if len(ht) < 3:
        return None, 0.0
    best_by_title = {}
    for title, _, toks in segs:
        if not title:
            continue
        score = len(ht & toks) / len(ht)
        tn = title_norm(title)
        if len(tn) >= 3 and tn in h_norm:
            score += TITLE_BONUS
        if score > best_by_title.get(title, -1):
            best_by_title[title] = score
    if not best_by_title:
        return None, 0.0
    ranked = sorted(best_by_title.items(), key=lambda x: -x[1])
    top_title, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if top >= FUZZY_MIN and top - second >= FUZZY_MARGIN:
        return top_title, top
    return None, top


def build(graphml=GRAPHML, chunks_path=CHUNKS, out=INDEX, seed=0):
    import networkx as nx

    rnd = random.Random(seed)
    chunks = json.load(open(chunks_path, encoding="utf-8"))
    segs = chunk_segments(chunks)
    print(f"chunks: {len(chunks)}", flush=True)
    G = nx.read_graphml(graphml)
    print(f"graph: {G.number_of_nodes()} nodes", flush=True)

    index, total, method = {}, 0, Counter()
    val = Counter()
    for node, attrs in G.nodes(data=True):
        if attrs.get("role") != "hyperedge":
            continue
        total += 1
        text = clean_hyperedge(node)
        h = norm(text)
        if len(h) < MIN_LEN:
            method["too_short"] += 1
            continue
        cids = [c for c in str(attrs.get("source_id", "")).split(SEP) if c in segs]
        titles, how = set(), "unknown"
        for cid in cids:
            t = find_exact(h, segs[cid])
            if t:
                titles.add(t); how = "exact"
        if not titles:
            for cid in cids:
                t, _ = find_fuzzy(text, h, segs[cid])
                if t:
                    titles.add(t); how = "fuzzy"
        elif len(titles) == 1 and rnd.random() < VALIDATE_RATE:
            # accuracy check of the fuzzy method against exact matches
            (exact_t,) = tuple(titles)
            ft = {find_fuzzy(text, h, segs[c])[0] for c in cids} - {None}
            val["abstained" if not ft else "agree" if ft == {exact_t} else "disagree"] += 1
        method[how] += 1
        e = index.setdefault(h, {"titles": [], "chunks": [], "how": how})
        e["titles"] = sorted(set(e["titles"]) | titles)
        e["chunks"] = sorted(set(e["chunks"]) | set(cids))
        if how == "exact" or (how == "fuzzy" and e["how"] == "unknown"):
            e["how"] = how

    json.dump(index, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    n = max(total, 1)
    print(f"hyperedges: {total} | unique keys: {len(index)}")
    print("by method:", {k: f"{v} ({100*v/n:.1f}%)" for k, v in method.items()})
    resolved = method["exact"] + method["fuzzy"]
    print(f"resolved overall: {resolved} ({100*resolved/n:.1f}%)")
    if val:
        tot = sum(val.values())
        print(f"fuzzy check on {tot} exact matches:",
              {k: f"{v} ({100*v/tot:.1f}%)" for k, v in val.items()})
    multi = Counter(len(e["titles"]) for e in index.values())
    print("titles per key (0/1/2/3+):", multi[0], multi[1], multi[2], sum(v for k, v in multi.items() if k >= 3))
    print(f"saved -> {out}")


class ProvenanceIndex:
    def __init__(self, data):
        self.data = data

    @classmethod
    def load(cls, path=INDEX):
        return cls(json.load(open(path, encoding="utf-8")))

    def entry(self, knowledge_text: str):
        return self.data.get(norm(clean_hyperedge(knowledge_text)))

    def titles(self, knowledge_text: str):
        e = self.entry(knowledge_text)
        return e["titles"] if e else []

    def key(self, knowledge_text: str) -> str:
        return norm(clean_hyperedge(knowledge_text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "lookup"])
    ap.add_argument("text", nargs="?")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        print(ProvenanceIndex.load().entry(a.text or ""))
