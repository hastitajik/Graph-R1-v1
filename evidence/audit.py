"""
Evidence Audit - builds the Evidence Support Graph G_E and turns the evidence state into a
CERTIFICATE (or a list of reasons why there is none yet).

No network, no LLM: pure function of an EvidenceState.

G_E (networkx.DiGraph), restricted to the obligations the answer actually needs:
    ("u", article title) --from-->     ("i", evidence item)      provenance
    ("i", evidence item) --supports--> ("o", obligation)         live support only
    ("o", dependency)    --depends-->  ("o", obligation)         the reasoning chain
"Live" is defined in state.py: support matched under the current value of every inherited
variable and agreeing with the chosen value of the obligation's own variables.

Evidence cut
    An obligation is supported if some item reaches it AND that item still has a provenance unit.
    kappa_i   = number of independent provenance units behind obligation i (0 = missing,
                1 = single point of failure, >= 2 = robust). UNKNOWN counts once and is never
                counted as independent in D (see d_prov).
    A "critical cut" is a unit whose removal leaves some required obligation unsupported.
    audit() computes kappa and the cuts on the GRAPH and cross-checks them against the state
    (tests/test_evidence_offline.py asserts both agree).

Counterfactual fragility
    For every unit u: drop u from the evidence and recompute. It is fragile if some required
    obligation becomes uncovered, or the chosen value of a variable changes.

Certificate (what the controller must reach before it may answer)
    complete  <=>  every required retrieve obligation is covered
                   and the support graph is connected
                   and no unresolved conflict on a variable of a required obligation
                   and every critical cut has been audited (its sole supporting passage was
                   re-checked by the Verify action and held).
    Robustness beyond that (kappa >= 2, D) is NOT demanded - on many corpora a fact has exactly
    one article - it is what the controller's value-of-evidence logic decides to buy or not.

State vector for Pareto comparison: (C, R, D, -X)
    C coverage of required retrieve obligations      R share of covered ones that are robust
    D share with >= 2 independent KNOWN units        X number of unresolved conflicts
"""
from dataclasses import dataclass, field

import networkx as nx

from evidence.state import NO_DROP, UNKNOWN

DEPENDS, SUPPORTS, FROM = "depends", "supports", "from"


def required_retrieve(state):
    req = state.required_ids()
    return [o for o in state.order if o in req and state.is_retrieve(o)]


def build_graph(state, drop=NO_DROP):
    """Evidence Support Graph G_E for the obligations the answer needs."""
    G = nx.DiGraph()
    req = state.required_ids()
    for oid in state.order:
        if oid in req:
            G.add_node(("o", oid), kind="obligation", retrieve=state.is_retrieve(oid))
    for oid in state.order:
        if oid in req:
            for d in state.obs[oid]["depends_on"]:
                G.add_edge(("o", d), ("o", oid), rel=DEPENDS)
    for oid in required_retrieve(state):
        for key in state.live_support(oid, drop):
            u = state.unit(state.items[key])
            G.add_node(("i", key), kind="item")
            G.add_node(("u", u), kind="unit")
            G.add_edge(("i", key), ("o", oid), rel=SUPPORTS)
            G.add_edge(("u", u), ("i", key), rel=FROM)
    return G


def graph_units(G, oid):
    """Provenance units connected to obligation `oid` in G."""
    node = ("o", oid)
    units = set()
    for i in G.predecessors(node):
        if i[0] == "i":
            units |= {u[1] for u in G.predecessors(i) if u[0] == "u"}
    return units


def graph_critical_cuts(G, oids):
    """{oid: unit} for every obligation that a single unit disconnects from its evidence."""
    cuts = {}
    for oid in oids:
        units = graph_units(G, oid)
        if len(units) == 1:
            cuts[oid] = next(iter(units))
    return cuts


def risk(vector, lam_x=0.5, lam_r=0.5):
    """Scalar risk of an evidence state (lower is better):
        Risk = (1 - C) + lam_x * X + lam_r * (1 - R)
    C coverage, X number of unresolved conflicts, R structural robustness. The weights are hand-set
    hyper-parameters. Model confidence is deliberately not part of it."""
    c, r, _d, neg_x = vector
    return (1.0 - c) + lam_x * (-neg_x) + lam_r * (1.0 - r)


def pareto_improves(before, after, eps=1e-9):
    """True if no component got worse and at least one got better (higher is better)."""
    if any(a < b - eps for a, b in zip(after, before)):
        return False
    return any(a > b + eps for a, b in zip(after, before))


def _required_vars(state):
    req = state.required_ids()
    return sorted({v for o in req for v in state._vars[o]})


def analyse_conflicts(state):
    """One row per variable with several supported values. resolved = the best value has strictly
    more independent KNOWN units than the runner-up; otherwise the choice is arbitrary."""
    rows = []
    for v in _required_vars(state):
        ranked = state.ranked_values(v)
        if len(ranked) > 1:
            known = [len(r[1] - {UNKNOWN}) for r in ranked]
            rows.append({"var": v, "values": [r[0] for r in ranked], "units": known[:2],
                         "resolved": known[0] > known[1]})
    return rows


def counterfactual_fragility(state):
    """[{unit, uncovers, flips}] for every unit whose removal breaks something."""
    req = required_retrieve(state)
    all_vars = _required_vars(state)
    base = {v: state.value(v) for v in all_vars}
    units = sorted({state.unit(state.items[k]) for o in req for k in state.live_support(o)})
    out = []
    for u in units:
        drop = frozenset({u})
        uncovers = [o for o in req if not state.live_support(o, drop)]
        flips = [v for v in all_vars if base[v] is not None and state.value(v, drop) != base[v]]
        if uncovers or flips:
            out.append({"unit": u, "uncovers": uncovers, "flips": flips})
    return out


def is_verified(state, oid):
    return any(state.verified.get((oid, k)) is True for k in state.live_support(oid))


@dataclass
class Certificate:
    complete: bool
    reasons: list
    coverage: float
    uncovered: list          # required retrieve obligations without live support
    blocked: list            # subset of `uncovered` that cannot even be searched yet (upstream unbound)
    connected: bool
    kappa: dict              # oid -> number of units (UNKNOWN counts once)
    critical: list           # [{"oid", "unit", "verified"}] single points of failure
    unaudited: list          # oids in `critical` whose sole passage was not verified yet
    conflicts: list
    unresolved: list         # variables with an unresolved conflict
    fragile: list            # counterfactual fragility
    d_prov: float
    r_struct: float
    vector: tuple = field(default=())

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def audit(state):
    req = required_retrieve(state)
    G = build_graph(state)

    # ---- coverage / chain
    uncovered = [o for o in req if not state.live_support(o)]
    blocked = [o for o in uncovered if not state.ready(o)]
    onodes = [n for n in G.nodes if n[0] == "o"]
    connected = (not uncovered) and bool(onodes) \
        and nx.number_weakly_connected_components(G.subgraph(onodes)) == 1

    # ---- kappa (state) and critical cuts (graph); the test-suite checks both views agree
    kappa = {o: state.kappa(o) for o in req}
    cuts = graph_critical_cuts(G, req)
    critical = [{"oid": o, "unit": u, "verified": is_verified(state, o)} for o, u in cuts.items()]
    unaudited = [c["oid"] for c in critical if not c["verified"]]

    # ---- conflicts, fragility, structure metrics
    conflicts = analyse_conflicts(state)
    unresolved = [c["var"] for c in conflicts if not c["resolved"]]
    fragile = counterfactual_fragility(state)

    n = max(len(req), 1)
    covered = [o for o in req if o not in uncovered]
    d_prov = sum(state.kappa_known(o) >= 2 for o in req) / n if req else 1.0
    robust = [o for o in covered if kappa[o] >= 2 or is_verified(state, o)]
    r_struct = len(robust) / len(covered) if covered else 0.0
    coverage = len(covered) / n if req else 1.0
    vector = (coverage, r_struct, d_prov, -float(len(unresolved)))

    reasons = []
    if uncovered:
        reasons.append(f"uncovered obligations: {', '.join(uncovered)}")
    if not connected and not uncovered:
        reasons.append("support graph is not connected")
    if unresolved:
        reasons.append(f"unresolved conflict on {', '.join(unresolved)}")
    if unaudited:
        reasons.append(f"critical cut not audited: {', '.join(unaudited)}")

    return Certificate(complete=not reasons, reasons=reasons, coverage=coverage, uncovered=uncovered,
                       blocked=blocked, connected=connected, kappa=kappa, critical=critical,
                       unaudited=unaudited, conflicts=conflicts, unresolved=unresolved, fragile=fragile,
                       d_prov=d_prov, r_struct=r_struct, vector=vector)
