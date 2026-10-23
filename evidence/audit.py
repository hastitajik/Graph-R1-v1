"""Layer 3 (v0) - Evidence Audit and Risk. Pure functions over EvidenceState, no network.

  C = coverage            share of retrieve-obligations that have evidence
  R = robustness          mean over retrieve-obligations of min(kappa, 2) / 2
                          (kappa = independent provenance units: 0 none, 1 fragile, >=2 robust)
  X = conflict rate       share of bound variables that carry more than one distinct value
  Risk(s) = (1 - C) + lam_x * X + lam_r * (1 - R)          lam_x, lam_r are fixed by hand
"""


def retrieve_ids(state):
    return [o for o in state.order if state.is_retrieve(o)]


def robustness(state):
    ids = retrieve_ids(state)
    return sum(min(state.kappa(o), 2) / 2 for o in ids) / len(ids) if ids else 1.0


def conflicts(state):
    """Variables for which evidence gave more than one distinct value."""
    vs = {v for oid in state.order for v in state._vars[oid]}
    return sorted(v for v in vs if len(state.alt_values(v)) > 1)


def conflict_rate(state):
    vs = {v for oid in state.order for v in state._vars[oid] if state.bound(v)}
    return len(conflicts(state)) / len(vs) if vs else 0.0


def risk(state, lam_x=0.5, lam_r=0.5):
    return (1 - state.coverage()) + lam_x * conflict_rate(state) + lam_r * (1 - robustness(state))


def audit(state, lam_x=0.5, lam_r=0.5):
    return {
        "C": round(state.coverage(), 3),
        "R": round(robustness(state), 3),
        "X": round(conflict_rate(state), 3),
        "risk": round(risk(state, lam_x, lam_r), 3),
        "kappa": {o: state.kappa(o) for o in retrieve_ids(state)},
        "conflicts": conflicts(state),
    }
