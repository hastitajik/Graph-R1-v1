"""
Layer 2 - Evidence State.

Pure data + rules, no network. Holds the obligations of one question, the evidence items
retrieved so far (deduplicated), which item supports which obligation, and the variable
bindings (?D = "Wojciech Jerzy Has").

Provenance unit of an item = its Wikipedia article title(s) (see provenance.py).
Items with unknown title all collapse into ONE "UNKNOWN" unit: they count for Coverage
but are never counted as independent support.
"""
import re
from dataclasses import dataclass

from evidence.provenance import norm

VAR = re.compile(r"\?[A-Za-z]\w*")
UNKNOWN = "UNKNOWN"
MIN_QUOTE = 10  # normalized characters


@dataclass
class Item:
    key: str
    text: str
    coherence: float
    titles: list
    round: int
    query: str


class EvidenceState:
    def __init__(self, obligations: dict, prov):
        self.prov = prov
        self.obs = {o["id"]: o for o in obligations["obligations"]}
        self.order = [o["id"] for o in obligations["obligations"]]
        self.answer_id = obligations["answer_id"]
        self.items = {}                              # item key -> Item
        self.support = {oid: {} for oid in self.obs}  # oid -> {item key: {var: value}}
        self.exhausted = set()                       # oids whose query returned nothing new
        self._vars = {oid: VAR.findall(o["text"]) for oid, o in self.obs.items()}

    # ---- structure -------------------------------------------------------
    def is_retrieve(self, oid):
        return self.obs[oid]["kind"] == "retrieve"

    def dep_vars(self, oid):
        out = set()
        for d in self.obs[oid]["depends_on"]:
            out |= set(self._vars[d])
        return out

    def own_vars(self, oid):
        dv = self.dep_vars(oid)
        return [v for v in dict.fromkeys(self._vars[oid]) if v not in dv]

    # ---- bindings ---------------------------------------------------------
    def _candidates(self, var):
        """normalized value -> {"value": str, "items": set(item keys)}"""
        cands = {}
        for sup in self.support.values():
            for key, b in sup.items():
                if var in b and b[var]:
                    c = cands.setdefault(norm(b[var]), {"value": b[var], "items": set()})
                    c["items"].add(key)
        return cands

    def value(self, var):
        """Best-supported value of a variable (most independent units, then coherence)."""
        cands = self._candidates(var)
        if not cands:
            return None

        def score(c):
            units = {self.unit(self.items[k]) for k in c["items"]}
            return (len(units), max(self.items[k].coherence for k in c["items"]), len(c["items"]))

        return max(cands.values(), key=score)["value"]

    def alt_values(self, var):
        """All distinct values seen for a variable (more than one = potential conflict)."""
        return [c["value"] for c in self._candidates(var).values()]

    def bound(self, var):
        return self.value(var) is not None

    def instantiate(self, oid):
        text = self.obs[oid]["text"]
        for v in dict.fromkeys(self._vars[oid]):
            val = self.value(v)
            if val is not None:
                text = text.replace(v, val)
        return text

    def ready(self, oid):
        return self.is_retrieve(oid) and all(self.bound(v) for v in self._vars[oid] if v in self.dep_vars(oid))

    # ---- evidence ----------------------------------------------------------
    @staticmethod
    def unit(item: Item):
        return "|".join(item.titles) if item.titles else UNKNOWN

    def add_items(self, results, rnd, query):
        """results: [{"text", "coherence"}]. Returns keys of items that are NEW."""
        new = []
        for r in results:
            key = self.prov.key(r["text"])
            if not key or key in self.items:
                continue
            self.items[key] = Item(key, r["text"], r["coherence"], self.prov.titles(r["text"]), rnd, query)
            new.append(key)
        return new

    def validate_match(self, oid, item: Item, bindings: dict, quote: str):
        """Guards against LLM hallucination. Returns (ok, reason)."""
        if oid not in self.obs or not self.is_retrieve(oid):
            return False, "unknown or non-retrieve obligation"
        if not self.ready(oid):
            return False, "obligation not ready"
        body = norm(item.text)
        q = norm(quote or "")
        if len(q) < MIN_QUOTE or q not in body:
            return False, "quote not found in passage"
        for v in self.own_vars(oid):
            val = (bindings or {}).get(v)
            if not val or norm(val) not in body:
                return False, f"binding {v} missing or not in passage"
        return True, ""

    def apply_match(self, oid, item_key, bindings):
        own = self.own_vars(oid)
        self.support[oid][item_key] = {v: bindings[v] for v in own if bindings.get(v)}

    # ---- metrics (Layer 3 will extend) ---------------------------------------
    def covered(self, oid):
        return bool(self.support[oid])

    def coverage(self):
        r = [o for o in self.order if self.is_retrieve(o)]
        return sum(self.covered(o) for o in r) / len(r) if r else 1.0

    def kappa(self, oid):
        """Number of distinct provenance units supporting an obligation (UNKNOWN counts once)."""
        return len({self.unit(self.items[k]) for k in self.support[oid]})

    def all_values(self):
        vs = {v for oid in self.order for v in self._vars[oid]}
        return {v: self.value(v) for v in vs if self.bound(v)}
