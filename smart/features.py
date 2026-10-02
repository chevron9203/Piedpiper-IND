"""Live feature rows: the research builders run on the price window for explicit dates."""
from __future__ import annotations
import sys
import pandas as pd

from smart.config import BASE, STEP

sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM          # noqa: E402
from scripts import research_smart_events as EV      # noqa: E402
from scripts import research_smart_peers as PR       # noqa: E402


def rows(P, dates, offset):
    """Unranked feature rows (date, sym) for `dates`: price/volume + events + peers.
    Returns (X, C, univ, mkt) like compute_features, X with all model columns."""
    X, C, univ, mkt = SM.compute_features(P, STEP, dates=dates, age_offset=offset, use_cache=False)
    E = EV.build(STEP, P=P, dates=dates, age_offset=offset, save=False)[0]   # (features, events, insider)
    F = PR.build(STEP, feats=(X, C, univ, mkt), save=False)
    return X.join(E).join(F), C, univ, mkt
