"""
Phase 0 feasibility checklist.

Run this end-to-end before trusting any data or backtest result.
Each check prints PASS / FAIL / WARN with a short explanation.
Exit code 0 if all critical checks pass, 1 if any critical check fails.

Usage:
    python scripts/phase0_checks.py
    python scripts/phase0_checks.py --skip-eod2   # if EOD2 not yet initialised
"""
import sys
import argparse
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from storage.store import init_schema


# ── Known CA events for EOD2 cross-source agreement check ───────────────────
# SmartAPI and EOD2 both return backward-adjusted data, so the validation
# checks that the two independent sources AGREE around known CA dates
# (max diff ≤ 0.5%). Disagreement indicates one source missed an adjustment.
# Must be within the 5-year SmartAPI pull window (after ~July 2021).
KNOWN_CORPORATE_ACTIONS = {
    "IRCTC-EQ": [
        (date(2021, 10, 28), "bonus", "1:1 bonus issue"),
    ],
    "POLYCAB-EQ": [
        (date(2023, 9, 14), "bonus", "1:1 bonus issue"),
    ],
    "DMART-EQ": [
        (date(2022, 9, 14), "bonus", "1:1 bonus issue"),
    ],
    "HDFCBANK-EQ": [
        (date(2023, 7, 3), "merger", "HDFC Ltd merger"),
    ],
}


def check(name: str, critical: bool = True):
    """Decorator / context for a named check."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                status = result if isinstance(result, str) else ("PASS" if result else "FAIL")
                icon = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
                print(f"  {icon} {name}: {status}")
                return result
            except Exception as exc:
                print(f"  ✗ {name}: FAIL — {exc}")
                return "FAIL"
        wrapper.__name__ = fn.__name__
        wrapper.critical = critical
        return wrapper
    return decorator


def run_checks(skip_eod2: bool = False) -> bool:
    print("\n" + "="*60)
    print("PIEDPIPER — PHASE 0 FEASIBILITY CHECKS")
    print("="*60)

    results = {}

    # ── 1. Credentials & auth ─────────────────────────────────────────────────
    print("\n[1] Authentication")

    def check_auth() -> str:
        from data.auth import SmartAPISession
        session = SmartAPISession()
        session.authenticate()
        client = session.get_client()
        profile = client.getProfile(session.feed_token)
        if profile and profile.get("status"):
            return "PASS"
        return "FAIL"

    print("  Testing SmartAPI auth with TOTP...")
    try:
        result = check_auth()
        print(f"  {'✓' if result == 'PASS' else '✗'} SmartAPI TOTP auth: {result}")
        results["auth"] = result
    except Exception as exc:
        print(f"  ✗ SmartAPI TOTP auth: FAIL — {exc}")
        results["auth"] = "FAIL"

    # ── 2. Instrument master ─────────────────────────────────────────────────
    print("\n[2] Instrument Master")

    def check_instrument_master() -> str:
        from data.instrument_master import refresh_master, get_nse_equity_master, verify_india_vix
        refresh_master(force=True)
        equity = get_nse_equity_master()
        if len(equity) < 1000:
            return f"WARN — only {len(equity)} NSE equity instruments (expected >1000)"
        print(f"       NSE equity instruments: {len(equity)}")
        vix = verify_india_vix()
        print(f"       India VIX token confirmed: {vix.get('token')} (symbol: {vix.get('symbol')})")
        return "PASS"

    try:
        r = check_instrument_master()
        results["instrument_master"] = r
        print(f"  {'✓' if r == 'PASS' else '⚠' if 'WARN' in r else '✗'} Instrument master: {r}")
    except Exception as exc:
        print(f"  ✗ Instrument master: FAIL — {exc}")
        results["instrument_master"] = "FAIL"

    # ── 3. Historical data pull ───────────────────────────────────────────────
    print("\n[3] Historical Data Pull (5 years, 5 stocks)")

    test_symbols = ["RELIANCE-EQ", "INFY-EQ", "HDFCBANK-EQ", "TCS-EQ", "IRCTC-EQ"]
    to_date = date.today()
    from_date = date(to_date.year - 5, to_date.month, to_date.day)

    raw_data: dict[str, pd.DataFrame] = {}
    data_pull_ok = True

    for sym in test_symbols:
        try:
            from data.instrument_master import symbol_to_token
            from data.smartapi_client import fetch_candles
            token = symbol_to_token(sym)
            if not token:
                print(f"  ⚠ {sym}: token not found in master")
                data_pull_ok = False
                continue
            df = fetch_candles(token, "NSE", "1d", from_date, to_date)
            raw_data[sym] = df
            years = (df.index[-1] - df.index[0]).days / 365 if len(df) > 1 else 0
            print(f"  ✓ {sym}: {len(df)} rows, {years:.1f} years")
        except Exception as exc:
            print(f"  ✗ {sym}: FAIL — {exc}")
            data_pull_ok = False

    results["data_pull"] = "PASS" if data_pull_ok and len(raw_data) == len(test_symbols) else "FAIL"

    # ── 4. SmartAPI data quality check ───────────────────────────────────────
    # SmartAPI returns backward-adjusted data (empirically confirmed: no price
    # jumps at known CA dates, closes match EOD2 adjusted series within 0.5%).
    # This check verifies data completeness and absence of spurious >50% jumps.
    print("\n[4] SmartAPI Data Quality")
    print("    (Note: SmartAPI returns backward-adjusted data, same as EOD2)")

    from data.validator import check_price_jumps, check_gaps, check_ohlcv_consistency
    quality_ok = True
    for sym, df in list(raw_data.items())[:3]:
        if df.empty:
            continue
        big_jumps = check_price_jumps(df, sym, threshold_pct=0.50)  # >50% = truly anomalous
        ohlc_errs = check_ohlcv_consistency(df, sym)
        issues = []
        if big_jumps:
            issues.append(f"{len(big_jumps)} anomalous jumps >50%")
        if ohlc_errs:
            issues.append(f"{ohlc_errs} OHLC violations")
        if issues:
            print(f"  ⚠ {sym}: {', '.join(issues)}")
            quality_ok = False
        else:
            print(f"  ✓ {sym}: clean ({len(df)} rows, no anomalies)")
    results["smartapi_quality"] = "PASS" if quality_ok else "WARN"

    # ── 5. EOD2 adjusted data ────────────────────────────────────────────────
    if not skip_eod2:
        print("\n[5] EOD2 Adjusted Data Validation")

        eod2_ok = True
        eod2_data: dict[str, pd.DataFrame] = {}

        try:
            from data.eod2_manager import is_eod2_available, load_adjusted_ohlcv, list_available_symbols
            if not is_eod2_available():
                print("  ⚠ EOD2 not initialised. Run: python -c \"from data.eod2_manager import setup_eod2; setup_eod2()\"")
                results["eod2"] = "WARN — not initialised"
            else:
                available = list_available_symbols()
                print(f"  EOD2 symbols available: {len(available)}")

                # Load EOD2 data for all CA-validation symbols too
                eod2_load_syms = list(set(test_symbols[:3]) | set(KNOWN_CORPORATE_ACTIONS.keys()))
                for sym in eod2_load_syms:
                    bare = sym.replace("-EQ", "")
                    try:
                        df = load_adjusted_ohlcv(bare, from_date, to_date)
                        eod2_data[sym] = df
                        eod2_data[bare] = df  # both keys for lookup convenience
                        if sym in test_symbols[:3]:
                            print(f"  ✓ {bare}: {len(df)} adjusted rows")
                    except FileNotFoundError:
                        if sym in test_symbols[:3]:
                            print(f"  ✗ {bare}: not found in EOD2")
                            eod2_ok = False

                # Pull raw data for CA-validation symbols not already in raw_data
                for sym in KNOWN_CORPORATE_ACTIONS:
                    if sym not in raw_data:
                        try:
                            from data.instrument_master import symbol_to_token
                            from data.smartapi_client import fetch_candles
                            token = symbol_to_token(sym)
                            if token:
                                df = fetch_candles(token, "NSE", "1d", from_date, to_date)
                                raw_data[sym] = df
                        except Exception:
                            pass

                # CA cross-source agreement (SmartAPI vs EOD2, both adjusted)
                from data.validator import validate_eod2_adjustment
                print("\n  EOD2 vs SmartAPI Cross-Source Agreement (around CA dates):")
                ca_pass_count = 0
                ca_total = 0
                for sym, ca_list in KNOWN_CORPORATE_ACTIONS.items():
                    bare = sym.replace("-EQ", "")
                    smartapi_sym_df = raw_data.get(sym, pd.DataFrame())
                    eod2_sym_df = eod2_data.get(sym, eod2_data.get(bare, pd.DataFrame()))
                    if not smartapi_sym_df.empty and not eod2_sym_df.empty:
                        ca_total += 1
                        result_v = validate_eod2_adjustment(
                            sym, smartapi_sym_df, eod2_sym_df,
                            [ca[0] for ca in ca_list]
                        )
                        status = result_v.get("overall", "unknown")
                        max_diff = result_v.get("overall_max_diff_pct", 0)
                        icon = "✓" if status == "ok" else "✗"
                        print(f"    {icon} {sym}: {status} (max diff {max_diff:.3f}%)")
                        if result_v.get("unadjusted_jumps_in_eod2"):
                            print(f"      ⚠ EOD2 price jumps >40%: {result_v['unadjusted_jumps_in_eod2']}")
                        if status == "ok":
                            ca_pass_count += 1
                    else:
                        missing = "SmartAPI" if smartapi_sym_df.empty else "EOD2"
                        print(f"    - {sym}: skipped (no {missing} data)")

                if ca_total == 0:
                    print("  ⚠ No CA symbols had data from both sources — EOD2 cross-check skipped")
                    results["eod2"] = "WARN"
                elif ca_pass_count >= 2:
                    print(f"  ✓ EOD2 cross-source agreement confirmed ({ca_pass_count}/{ca_total} symbols)")
                    results["eod2"] = "PASS"
                elif ca_pass_count >= 1:
                    print(f"  ⚠ EOD2 partial agreement: {ca_pass_count}/{ca_total} symbols — verify manually")
                    results["eod2"] = "WARN"
                else:
                    print(f"  ✗ EOD2 cross-source agreement failed for all symbols — investigate")
                    results["eod2"] = "FAIL"
        except Exception as exc:
            print(f"  ✗ EOD2: FAIL — {exc}")
            results["eod2"] = "FAIL"
    else:
        results["eod2"] = "SKIPPED"
        print("\n[5] EOD2: SKIPPED (--skip-eod2 flag)")

    # ── 6. NSE holiday calendar ──────────────────────────────────────────────
    print("\n[6] NSE Holiday Calendar")
    try:
        from data.holiday_calendar import get_holidays, is_trading_day, next_trading_day
        holidays = get_holidays()
        today = date.today()
        print(f"  Loaded {len(holidays)} holidays")
        print(f"  Today ({today}) is trading day: {is_trading_day(today)}")
        print(f"  Next trading day: {next_trading_day(today)}")
        results["holidays"] = "PASS" if len(holidays) >= 5 else "WARN"
        print(f"  {'✓' if results['holidays'] == 'PASS' else '⚠'} Holiday calendar: {results['holidays']}")
    except Exception as exc:
        print(f"  ✗ Holiday calendar: FAIL — {exc}")
        results["holidays"] = "FAIL"

    # ── 7. Universe screener ─────────────────────────────────────────────────
    print("\n[7] Universe Screener")
    try:
        from data.universe import fetch_nifty200, build_sector_map, screen_universe
        nifty200 = fetch_nifty200()
        print(f"  Nifty 200 raw: {len(nifty200)} stocks")
        sector_map = build_sector_map()
        print(f"  Sector map: {len(sector_map)} stocks, {sector_map['sector'].nunique()} sectors")
        universe = screen_universe(nifty200)
        print(f"  Filtered universe: {len(universe)} stocks")
        if 80 <= len(universe) <= 200:
            results["universe"] = "PASS"
            print(f"  ✓ Universe size in expected range (80-200): PASS")
        else:
            results["universe"] = "WARN"
            print(f"  ⚠ Universe size {len(universe)} outside expected range — review filters")
    except Exception as exc:
        print(f"  ✗ Universe screener: FAIL — {exc}")
        results["universe"] = "FAIL"

    # ── 8. Database schema ───────────────────────────────────────────────────
    print("\n[8] Database Schema")
    try:
        init_schema()
        print(f"  ✓ DuckDB schema initialised: {settings.DB_PATH}")
        results["db"] = "PASS"
    except Exception as exc:
        print(f"  ✗ DB schema: FAIL — {exc}")
        results["db"] = "FAIL"

    # ── 9. Cost model sanity check ───────────────────────────────────────────
    print("\n[9] Cost Model Sanity Check")
    try:
        from config.cost_model import round_trip_cost, net_return
        # ₹50,000 trade in a Nifty 100 stock (e.g. 100 shares at ₹500)
        costs = round_trip_cost(100, 500.0, 520.0, "nifty100")
        net = net_return(100, 500.0, 520.0, "nifty100")
        gross_pct = (520 - 500) / 500 * 100
        net_pct = net * 100
        cost_pct = costs["total"] / (100 * 500) * 100
        print(f"  Sample: 100 shares @ ₹500→₹520 (Nifty100)")
        print(f"    Gross return: {gross_pct:.2f}%")
        print(f"    Total costs: ₹{costs['total']:.2f} ({cost_pct:.3f}%)")
        print(f"    Net return: {net_pct:.3f}%")
        print(f"    Cost breakdown: brok={costs['brokerage']:.2f}, stt={costs['stt']:.2f}, "
              f"slip={costs['slippage']:.2f}, dp={costs['dp_charge']:.2f}")
        results["cost_model"] = "PASS" if 0 < costs["total"] < 500 else "WARN"
        print(f"  {'✓' if results['cost_model'] == 'PASS' else '⚠'} Cost model: {results['cost_model']}")
    except Exception as exc:
        print(f"  ✗ Cost model: FAIL — {exc}")
        results["cost_model"] = "FAIL"

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    critical = ["auth", "instrument_master", "data_pull", "holidays", "db", "cost_model"]
    non_critical = ["smartapi_quality", "universe"]
    if not skip_eod2:
        critical.append("eod2")

    all_critical_pass = True
    for key in critical:
        status = results.get(key, "NOT RUN")
        icon = "✓" if status == "PASS" else "⚠" if status in ("WARN", "SKIPPED") else "✗"
        if status not in ("PASS", "WARN", "SKIPPED"):
            all_critical_pass = False
        print(f"  {icon} [{key}]: {status}")

    for key in non_critical:
        status = results.get(key, "NOT RUN")
        icon = "✓" if status == "PASS" else "⚠"
        print(f"  {icon} [{key}] (non-critical): {status}")

    print()
    if all_critical_pass:
        print("  ✓ All critical checks passed. Ready to proceed to Phase 1.")
    else:
        failed = [k for k in critical if results.get(k) not in ("PASS", "WARN", "SKIPPED")]
        print(f"  ✗ {len(failed)} critical check(s) failed: {failed}")
        print("    Resolve these before proceeding to Phase 1.")

    return all_critical_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piedpiper Phase 0 feasibility checks")
    parser.add_argument("--skip-eod2", action="store_true",
                        help="Skip EOD2 checks (use if EOD2 not yet initialised)")
    args = parser.parse_args()

    ok = run_checks(skip_eod2=args.skip_eod2)
    sys.exit(0 if ok else 1)
