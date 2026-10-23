"""Layer 3 (v0) - Decision-time controller. Rule-based, no training, no model confidence.

Actions implemented in v0:
  FillGap   - retrieve for a ready obligation that has no evidence yet
  Diversify - look for a SECOND independent source for an obligation that has kappa = 1
Not yet: Bridge, Challenge, Verify (see the plan).

Value of an action = predicted risk reduction - cost  (VNE).  Predicted reduction:
  FillGap   : C rises by 1/m and R by 0.5/m          -> (1 + 0.5*lam_r) / m
  Diversify : R rises by 0.5/m                        -> 0.5*lam_r / m
(m = number of retrieve obligations).  Only actions with VNE > 0 are candidates; when none is
left the controller stops (the caller then answers if C = 1, otherwise abstains).
"""
from evidence.audit import retrieve_ids


def plan(state, lam_r=0.5, cost=0.1, diversified=frozenset()):
    ids = retrieve_ids(state)
    m = len(ids) or 1
    acts = []
    for oid in ids:
        if not state.covered(oid):
            if state.ready(oid) and oid not in state.exhausted:
                gain = (1 + 0.5 * lam_r) / m
                acts.append({"action": "FillGap", "oid": oid, "gain": gain, "vne": gain - cost})
        elif state.kappa(oid) == 1 and oid not in diversified:
            gain = 0.5 * lam_r / m
            acts.append({"action": "Diversify", "oid": oid, "gain": gain, "vne": gain - cost})
    acts = [a for a in acts if a["vne"] > 0]
    acts.sort(key=lambda a: -a["vne"])
    return acts
