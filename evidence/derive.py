"""
Deterministic answer for "compare two dates" questions.

A certificate covers the RETRIEVED facts. The last step of a comparison question ("which film came
out first, A or B?") is a derive obligation that an LLM used to answer in free text, and that step was
not covered by anything: a complete certificate could still end in a wrong answer. When the facts are
dates this step needs no LLM:

    1. the question names two options ("..., A or B?" / "... out of A and B?")
    2. two date variables are bound; each is traced back, through the chain of obligations, to the
       option whose name appears in the obligation texts on that chain
    3. the direction (earlier / later) is read from the question words
    4. the option with the earlier / later date is the answer

If anything is ambiguous (not exactly two dates, an option cannot be matched to exactly one date, a
tie at the shared precision, no direction, a yes/no question) the function returns None and the LLM
answer is kept - so it never guesses.
"""
import re

from evidence.provenance import norm

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\s+(\d{{3,4}})\b", re.I)
_MDY = re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{3,4}})\b", re.I)
_MY = re.compile(rf"\b{_MON},?\s+(\d{{3,4}})\b", re.I)
_Y = re.compile(r"(?<!\d)(\d{4})(?!\d)")

_MIN = re.compile(r"\b(earlier|earliest|first|older|oldest|before|sooner)\b", re.I)
_MAX = re.compile(r"\b(later|latest|more recently|recently|younger|youngest|after|last)\b", re.I)
_YESNO = re.compile(r"\b(both|same|either)\b", re.I)


def parse_date(s: str):
    """(year, month, day); month / day are 0 when the text does not give them. None if no year."""
    if not s:
        return None
    m = _DMY.search(s)
    if m:
        return int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))
    m = _MDY.search(s)
    if m:
        return int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2))
    m = _MY.search(s)
    if m:
        return int(m.group(2)), _MONTHS[m.group(1).lower()], 0
    m = _Y.search(s)
    return (int(m.group(1)), 0, 0) if m else None


def _precision(d):
    return 3 if d[2] else 2 if d[1] else 1


def earlier(a, b):
    """True if a is before b, False if after, None if they cannot be told apart at the shared precision."""
    p = min(_precision(a), _precision(b))
    x, y = a[:p], b[:p]
    if x == y:
        return None
    return x < y


def question_options(question: str):
    """The two named options of 'Which ..., A or B?' / 'Who ... out of A and B?', or None."""
    q = question.strip().rstrip("?").strip()
    tail = q.rsplit(",", 1)[-1] if "," in q else q
    for sep in (" or ", " and "):
        # choose the separator that splits the tail into exactly two non-empty parts
        parts = re.split(sep, tail, flags=re.I)
        if len(parts) == 2 and all(p.strip() for p in parts):
            head = parts[0]
            head = re.sub(r"^.*\b(?:out of|between|of)\s+", "", head, flags=re.I) if sep == " and " else head
            return head.strip(), parts[1].strip()
    return None


def direction(question: str):
    """'min' (earlier date wins) / 'max' (later date wins) / None."""
    if _YESNO.search(question):
        return None
    lo, hi = bool(_MIN.search(question)), bool(_MAX.search(question))
    if lo == hi:
        return None
    return "min" if lo else "max"


def _chain_text(state, oid):
    """Original texts of `oid` and of everything it depends on."""
    seen, stack, out = set(), [oid], []
    while stack:
        o = stack.pop()
        if o in seen:
            continue
        seen.add(o)
        out.append(state.obs[o]["text"])
        stack.extend(state.obs[o]["depends_on"])
    return " ".join(out)


def derive_answer(state, question: str):
    """{"answer", "dates": {option: date text}} or None when the comparison is not unambiguous."""
    if state.obs[state.answer_id]["kind"] != "derive":
        return None
    way = direction(question)
    opts = question_options(question)
    if not way or not opts:
        return None

    date_vars = []
    for oid in state.order:
        if state.is_retrieve(oid) and oid in state.required_ids():
            for v in state.own_vars(oid):
                if state.expected_type(oid, v) == "date":
                    date_vars.append((oid, v))
    if len(date_vars) != 2:
        return None

    found = {}
    for oid, v in date_vars:
        val = state.value(v)
        if val is None or parse_date(val) is None:
            return None
        text = norm(_chain_text(state, oid))
        hit = [o for o in opts if norm(o) and norm(o) in text]
        if len(hit) != 1 or hit[0] in found:
            return None
        found[hit[0]] = val
    if set(found) != set(opts):
        return None

    a, b = opts
    first = earlier(parse_date(found[a]), parse_date(found[b]))
    if first is None:
        return None
    pick_a = first if way == "min" else not first
    return {"answer": a if pick_a else b, "dates": found, "direction": way}
