"""
Layer 2 - Evidence State.

Pure data + rules, no network. Holds the obligations of one question, the evidence items
retrieved so far (deduplicated), which item supports which obligation, and the variable
bindings (?D = "Wojciech Jerzy Has").

Provenance unit of an item = its Wikipedia article title(s) (see provenance.py).
Items with unknown title all collapse into ONE "UNKNOWN" unit: they count for Coverage
but are never counted as independent support.

"Live" support (what audit.py and the controller reason about): a support entry counts only if
  1. its unit is not dropped (counterfactual removal of one source),
  2. it was matched under the CURRENT value of every variable the obligation inherits
     (otherwise the hop's evidence talks about a different entity than the chain now uses), and
  3. it agrees with the chosen value of the obligation's own variables.
"""
import os
import re
from dataclasses import dataclass

from evidence.common import VAR  # noqa: F401  (re-exported for callers that import it from here)
from evidence.provenance import norm

UNKNOWN = "UNKNOWN"
MIN_QUOTE = 10  # normalized characters
NO_DROP = frozenset()
SELF_CHECK = os.environ.get("EVIDENCE_SELF_CHECK", "1") != "0"
SUBJECT_CHECK = os.environ.get("EVIDENCE_SUBJECT_CHECK", "1") != "0"

# ---- cheap guards against wrong bindings (found in real traces) ---------------------------------
MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE_CUE = re.compile(r"\b(?:born|died|released|published|founded|married|established|premiered|opened)\s+on$"
                       r"|\bdate of (?:birth|death)\b.*\bis$")
_PLACE_CUE = re.compile(r"\b(?:born|died|buried|located|produced|filmed|based|graduated|educated|studied|raised)\s+(?:in|at|from)$"
                        r"|\b(?:from|citizen of|native of)$"
                        r"|\b(?:nationality|country|place of (?:birth|death)|birthplace|city)\b.*\bis$")


def date_only(v: str) -> bool:
    """True if the value is nothing but a date or a year (no place / name words)."""
    s = re.sub(rf"\b{MONTH}\b", " ", v.lower())
    s = re.sub(r"\b(?:c|ca|circa|ad|bc|bce|ce|th|st|nd|rd|of|the)\b", " ", s)
    return re.sub(r"[\d\W_]+", "", s) == "" and bool(re.search(r"\d", v))


def _tokens(s: str) -> set:
    return {w for w in re.findall(r"\w+", s.lower()) if len(w) >= 3}


def _title_core(t: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", t)


def subject_in(value: str, titles, passage: str) -> bool:
    """Is the entity `value` what this passage is about? (its text, its article title, or >= half of its words)"""
    if norm(value) in norm(passage):
        return True
    for t in titles or []:
        tn = norm(_title_core(t))
        if tn and (tn in norm(value) or norm(value) in tn):
            return True
    vt = _tokens(value)
    # two-word names need both words ('William Clemens' is not 'William I, Count of Nassau-Dillenburg')
    need = 0.6 if len(vt) <= 3 else 0.5
    return not vt or len(vt & _tokens(" ".join(titles or []) + " " + passage)) / len(vt) >= need


_HONORIFIC = {"sir", "dr", "mr", "lord", "lady", "saint"}
_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "vi", "vii", "viii", "ix"}


def _sig(s: str) -> list:
    return [w for w in re.findall(r"\w+", s.lower()) if w not in _HONORIFIC and w not in {"the", "of", "and"}]


def same_entity(a: str, b: str) -> bool:
    """Conservative alias test for two surface forms of one name ('Infanta Elena of Spain' ~
    'Infanta Elena, Duchess of Lugo'): same first two significant words, no digits, same suffixes.
    Anything with digits (dates, ordinals) must match exactly."""
    if norm(a) == norm(b):
        return True
    if re.search(r"\d", a) or re.search(r"\d", b):
        return False
    ta, tb = _sig(a), _sig(b)
    if len(ta) < 2 or len(tb) < 2 or ta[:2] != tb[:2]:
        return False
    return {w for w in ta if w in _SUFFIX} == {w for w in tb if w in _SUFFIX}


def is_self_reference(value: str, question: str) -> bool:
    """The value is just the entity the question is about (e.g. 'uncle of X' bound to X itself)."""
    if not question or re.search(r"\d{4}", value):
        return False
    vt = _tokens(value)
    return len(vt) >= 2 and vt <= _tokens(question)


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
        self.question = ""                           # set by the caller; used by the self-reference guard
        self.items = {}                              # item key -> Item
        self.support = {oid: {} for oid in self.obs}  # oid -> {item key: {var: value}}
        self.context = {}                            # (oid, item key) -> inherited values at match time
        self.verified = {}                           # (oid, item key) -> True / False (Verify action)
        self.queries = {oid: [] for oid in self.obs}  # oid -> queries already sent (as written)
        self.exhausted = set()                       # oids whose query returned nothing new
        self.checked = set()                         # (item key, oid, inherited values) already shown to the matcher
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

    def owner_of(self, var):
        """The retrieve obligation that introduces `var` (own variable, not inherited)."""
        for oid in self.order:
            if self.is_retrieve(oid) and var in self.own_vars(oid):
                return oid
        return None

    def expected_type(self, oid, var):
        """'date' / 'place' / None, read from the words just before the variable in the obligation text."""
        tail = " ".join(self.obs[oid]["text"].split(var)[0].lower().split()[-6:])
        if _DATE_CUE.search(tail):
            return "date"
        if _PLACE_CUE.search(tail):
            return "place"
        return None

    def type_ok(self, oid, var, val):
        t = self.expected_type(oid, var)
        if t == "date" and not re.search(r"\d{3,4}", val):
            return False, f"type mismatch: {var} expects a date, got '{val}'"
        if t == "place" and date_only(val):
            return False, f"type mismatch: {var} expects a place, got date '{val}'"
        return True, ""

    def required_ids(self):
        """The answer obligation and everything it depends on (transitively)."""
        seen, stack = set(), [self.answer_id]
        while stack:
            o = stack.pop()
            if o in seen:
                continue
            seen.add(o)
            stack.extend(self.obs[o]["depends_on"])
        return seen

    def descendants(self, oid):
        """Obligations that depend on `oid`, directly or not."""
        out, frontier = set(), {oid}
        while frontier:
            nxt = {o for o in self.order if set(self.obs[o]["depends_on"]) & frontier} - out
            out |= nxt
            frontier = nxt
        return out

    # ---- bindings ---------------------------------------------------------
    def dep_context(self, oid, drop=NO_DROP):
        """Current (normalized) values of the variables `oid` inherits from its dependencies."""
        out = {}
        for v in sorted(self.dep_vars(oid)):
            val = self.value(v, drop)
            if val is not None:
                out[v] = norm(val)
        return out

    def _candidates(self, var, drop=NO_DROP):
        """normalized value -> {"value": str, "items": set(item keys)}; stale / dropped support is ignored."""
        cands = {}
        for oid, sup in self.support.items():
            ctx = None  # computed lazily: only the obligation that owns `var` is ever asked
            for key, b in sup.items():
                if var not in b or not b[var] or self.unit(self.items[key]) in drop:
                    continue
                if ctx is None:
                    ctx = self.dep_context(oid, drop)
                if self.context.get((oid, key), {}) != ctx:
                    continue
                k = norm(b[var])
                if k not in cands:  # a second surface form of the same name is not a conflict
                    k = next((x for x, c0 in cands.items() if same_entity(c0["value"], b[var])), k)
                c = cands.setdefault(k, {"value": b[var], "items": set()})
                c["items"].add(key)
        return cands

    def ranked_values(self, var, drop=NO_DROP):
        """[(value, units, item keys)], best first. Rank = independent units, coherence, #items."""
        rows = []
        for c in self._candidates(var, drop).values():
            units = {self.unit(self.items[k]) for k in c["items"]}
            coh = max(self.items[k].coherence for k in c["items"])
            rows.append(((len(units), coh, len(c["items"])), c["value"], units, c["items"]))
        rows.sort(key=lambda r: r[0], reverse=True)
        return [(v, u, i) for _, v, u, i in rows]

    def value(self, var, drop=NO_DROP):
        """Best-supported value of a variable (most independent units, then coherence)."""
        r = self.ranked_values(var, drop)
        return r[0][0] if r else None

    def alt_values(self, var):
        """All distinct values seen for a variable (more than one = potential conflict)."""
        return [c["value"] for c in self._candidates(var).values()]

    def conflicts(self):
        """{var: [values]} for variables that have more than one distinct supported value."""
        every = {v for oid in self.order for v in self._vars[oid]}
        return {v: self.alt_values(v) for v in sorted(every) if len(self._candidates(v)) > 1}

    def bound(self, var):
        return self.value(var) is not None

    def instantiate(self, oid, keep_own=False):
        """Obligation text with known variables replaced by their value.

        keep_own=True leaves the obligation's OWN variables as ?X even when they are bound, which is
        what the matcher needs to find a second source or a contradicting value for a fact we
        already hold (instantiated, the text would only match the value we already have).
        """
        skip = set(self.own_vars(oid)) if keep_own else set()

        def sub(m):
            v = m.group(0)
            if v in skip:
                return v
            val = self.value(v)
            return v if val is None else val

        return VAR.sub(sub, self.obs[oid]["text"])

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
            ok, why = self.type_ok(oid, v, val)
            if not ok:
                return False, why
            if SELF_CHECK and is_self_reference(val, self.question):
                return False, f"self-reference: {val} is the entity the question asks about"
            if SELF_CHECK:  # 'the father of X is X' (an entity is not its own parent / spouse / ...)
                for dv in self.dep_vars(oid):
                    other = self.value(dv)
                    if other and norm(other) == norm(val):
                        return False, f"self-reference: {v} equals {dv} ({val})"
        if SUBJECT_CHECK:  # the passage must be about the entity the chain arrived at
            for v in sorted(self.dep_vars(oid)):
                ent = self.value(v)
                if ent and v in self.obs[oid]["text"] and not subject_in(ent, item.titles, item.text):
                    return False, f"subject mismatch: passage is not about {ent}"
        return True, ""

    def apply_match(self, oid, item_key, bindings):
        own = self.own_vars(oid)
        self.support[oid][item_key] = {v: bindings[v] for v in own if bindings.get(v)}
        self.context[(oid, item_key)] = self.dep_context(oid)

    def remove_support(self, oid, item_key):
        """Drop a support entry that Verify found unsupported by its own passage."""
        self.support[oid].pop(item_key, None)
        self.context.pop((oid, item_key), None)
        self.verified[(oid, item_key)] = False

    def note_query(self, oid, query):
        self.queries[oid].append(query)

    def tried(self, oid):
        """Normalized queries already sent for this obligation."""
        return {norm(q) for q in self.queries[oid]}

    # ---- bookkeeping of what the matcher has already seen -------------------------
    def unchecked(self, oid):
        """Keys of held items that were never shown to the matcher for this obligation.

        An item retrieved for one obligation may also state the fact another obligation needs,
        but that other obligation was not 'ready' (a variable it depends on was unbound) when
        the item arrived, so the matcher never looked at the pair. Once the obligation is
        ready these items must be checked before spending a new query on it.
        """
        sig = self._sig(oid)
        return [k for k in self.items if (k, oid, sig) not in self.checked]

    def mark_checked(self, keys, oids):
        for o in oids:
            sig = self._sig(o)
            self.checked.update((k, o, sig) for k in keys)

    def _sig(self, oid):
        """Values of the inherited variables. If they change (a hop was re-resolved), the pairs
        checked under the old values are no longer valid and must be shown to the matcher again."""
        return tuple(sorted(self.dep_context(oid).items()))

    # ---- live support and metrics ---------------------------------------------------
    def live_support(self, oid, drop=NO_DROP):
        """Item keys that currently support `oid` (see module docstring), best coherence first."""
        own = self.own_vars(oid)
        chosen = {v: self.value(v, drop) for v in own}
        ctx = self.dep_context(oid, drop)
        out = []
        for key, b in self.support[oid].items():
            if self.unit(self.items[key]) in drop or self.context.get((oid, key), {}) != ctx:
                continue
            if any(chosen[v] is None or norm(b.get(v, "")) != norm(chosen[v]) for v in own):
                continue
            out.append(key)
        return sorted(out, key=lambda k: -self.items[k].coherence)

    def covered(self, oid):
        return bool(self.live_support(oid))

    def coverage(self):
        r = [o for o in self.order if self.is_retrieve(o)]
        return sum(self.covered(o) for o in r) / len(r) if r else 1.0

    def kappa(self, oid):
        """Distinct provenance units behind the chosen value of an obligation (UNKNOWN counts once)."""
        return len({self.unit(self.items[k]) for k in self.live_support(oid)})

    def kappa_known(self, oid):
        """Like kappa, but only units with a known article title (UNKNOWN is never independent)."""
        return len({self.unit(self.items[k]) for k in self.live_support(oid)} - {UNKNOWN})

    def all_values(self):
        vs = {v for oid in self.order for v in self._vars[oid]}
        return {v: self.value(v) for v in vs if self.bound(v)}
