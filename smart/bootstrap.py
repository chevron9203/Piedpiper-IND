"""One-time bootstrap ON THE MAC (needs the full research panel, ~6 GB RAM):
feature store + complete labels + listing-age state + empty paper ledger + models.
The box then only ever appends to these.

Run:  python -m smart.bootstrap [--no-train]
"""
from __future__ import annotations
import argparse, json, sys
import pandas as pd

from smart.config import BASE, CAPITAL, DROP, HORIZONS, LABELS, LEDGER, LIVE, STORE

sys.path.insert(0, str(BASE))
from scripts import research_smart_ml as SM          # noqa: E402
from smart import model as M, window as W            # noqa: E402

D = BASE/"data_store"


def build_store():
    X = pd.read_parquet(D/"smart_features_s5.parquet") \
          .join(pd.read_parquet(D/"smart_events_s5.parquet")) \
          .join(pd.read_parquet(D/"smart_peers_s5.parquet"))
    return X.drop(columns=[c for c in DROP if c in X.columns]).astype("float32")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args()
    LIVE.mkdir(parents=True, exist_ok=True)
    P = SM.load_panel()
    X = build_store()
    X.to_parquet(STORE)
    dates = X.index.get_level_values("date").unique()
    Y = pd.concat([M.complete_labels(P["close"], dates, h) for h in HORIZONS], axis=1).reindex(X.index)
    Y.to_parquet(LABELS)
    W.save_state(W.init_state_from_panel(P["close"]))
    if not LEDGER.exists():
        LEDGER.write_text(json.dumps({"cash": CAPITAL, "holdings": {}, "pending": None,
                                      "start": None, "history": []}, indent=1))
    print(f"store {X.shape} through {dates.max().date()}, labels {Y.notna().sum().to_dict()}")
    del P
    if not a.no_train:
        M.train(X, Y)


if __name__ == "__main__":
    main()
