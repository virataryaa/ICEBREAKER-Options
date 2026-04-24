"""
LCC_Ingest.py -- LCC London Cocoa #7 Options (ICE Futures Europe)
Fetches last 10 days, upserts into parquet.
Symbol format : C {mc}{yy}C{strike}-ICE  (e.g. C K26C115000-ICE)
Strike units  : symbol integer = GBP/tonne × 10  (so 115000 = £11,500/t)
Parquet stores: strike in GBP/tonne (strike_sym / 10)
Active months : H K N U Z only (Mar May Jul Sep Dec)
Sequential batches -- no threading (ICE COM is not thread-safe).
"""

import datetime
import json
import time
from pathlib import Path

import icepython as ice
import numpy as np
import pandas as pd

# -- Config -------------------------------------------------------------------

OUT_PATH    = Path(__file__).resolve().parent.parent / "Database" / "LCC_options_ice.parquet"
ATM_PATH    = Path(__file__).resolve().parent.parent / "Dashboard" / "atm.json"
TODAY       = datetime.date.today().isoformat()
FETCH_FROM  = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
RETRIES     = 3
BATCH_SIZE  = 50
N_MONTHS    = 8          # LCC has 5 expiry months/year; 8 covers ~1.5 years
ATM_WING    = 20
STRIKE_STEP = 2500       # symbol-unit increments (GBP/tonne × 10)
SYM_FACTOR  = 10         # %C 1!-ICE returns GBP/tonne; multiply by 10 for symbol

FIELDS = ["Settle", "Volume", "Open Interest"]

# LCC only has H K N U Z contract months
LCC_MONTHS    = [3, 5, 7, 9, 12]   # Mar May Jul Sep Dec
CODE_TO_MONTH = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
MONTH_TO_CODE = {v: k for k, v in CODE_TO_MONTH.items()}

# -- Helpers ------------------------------------------------------------------

def active_contracts() -> list[tuple[str, int]]:
    today = datetime.date.today()
    pairs = []
    year  = today.year
    while len(pairs) < N_MONTHS:
        for m in LCC_MONTHS:
            if year > today.year or (year == today.year and m > today.month):
                pairs.append((MONTH_TO_CODE[m], year))
                if len(pairs) == N_MONTHS:
                    break
        year += 1
    return pairs


def get_atm_strike() -> int:
    """Fetch %C 1!-ICE last settle (GBP/tonne), convert to symbol units, round to STRIKE_STEP."""
    raw = ice.get_timeseries(
        ["%C 1!-ICE"], ["Settle"], granularity="D",
        start_date=(datetime.date.today() - datetime.timedelta(days=5)).isoformat(),
        end_date=TODAY
    )
    if not raw or len(raw) < 2:
        raise RuntimeError("Could not fetch %C 1!-ICE price for ATM calculation")
    last_settle = next(
        (float(row[1]) for row in reversed(raw[1:]) if row[1] is not None),
        None
    )
    if last_settle is None:
        raise RuntimeError("No valid settle found for %C 1!-ICE")
    # Convert GBP/tonne to symbol units then round
    last_settle_sym = last_settle * SYM_FACTOR
    atm = round(last_settle_sym / STRIKE_STEP) * STRIKE_STEP
    print(f"LCC 1! last settle: {last_settle:.0f} GBP/t  ->  ATM strike: {atm} (sym) = {atm/SYM_FACTOR:.0f} GBP/t")

    existing = {}
    if ATM_PATH.exists():
        try:
            existing = json.loads(ATM_PATH.read_text())
        except Exception:
            pass
    existing["LCC"]     = float(atm / SYM_FACTOR)   # store in GBP/tonne; matches parquet strikes
    existing["updated"] = TODAY
    ATM_PATH.write_text(json.dumps(existing))
    return atm   # integer symbol units


def make_strikes(atm: int) -> list[int]:
    return [atm + i * STRIKE_STEP for i in range(-ATM_WING, ATM_WING + 1)]


def make_symbols(month_code: str, year: int, strikes: list[int]) -> list[str]:
    yy   = f"{year % 100:02d}"
    syms = []
    for s in strikes:
        syms.append(f"C {month_code}{yy}C{s}-ICE")
        syms.append(f"C {month_code}{yy}P{s}-ICE")
    return syms


def parse_symbol(sym: str) -> dict | None:
    try:
        base  = sym.replace("-ICE", "")
        parts = base.strip().split()
        if len(parts) != 2 or parts[0] != "C":
            return None
        c          = parts[1]
        mc, yy, cp = c[0], c[1:3], c[3]
        strike_sym = int(c[4:])
        if cp not in ("C", "P") or mc not in CODE_TO_MONTH:
            return None
        return {
            "ric":          sym,
            "option_type":  "Call" if cp == "C" else "Put",
            "strike":       strike_sym / SYM_FACTOR,   # store in GBP/tonne
            "expiry_month": CODE_TO_MONTH[mc],
            "expiry_year":  2000 + int(yy),
        }
    except Exception:
        return None


def fetch_batch(symbols: list[str]) -> pd.DataFrame:
    for attempt in range(1, RETRIES + 1):
        try:
            raw = ice.get_timeseries(symbols, FIELDS, granularity="D",
                                     start_date=FETCH_FROM, end_date=TODAY)
            if not raw or len(raw) < 2:
                return pd.DataFrame()
            df = pd.DataFrame(list(raw[1:]), columns=raw[0])
            df["Time"] = pd.to_datetime(df["Time"])
            rows = []
            for sym in symbols:
                if f"{sym}.Settle" not in df.columns:
                    continue
                tmp = pd.DataFrame({"Time": df["Time"]})
                tmp["settle"] = pd.to_numeric(df[f"{sym}.Settle"], errors="coerce")
                tmp["volume"] = pd.to_numeric(df.get(f"{sym}.Volume", pd.NA), errors="coerce")
                tmp["oi"]     = pd.to_numeric(df.get(f"{sym}.Open Interest", pd.NA), errors="coerce")
                tmp["ric"]    = sym
                rows.append(tmp)
            return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        except Exception as e:
            if attempt == RETRIES:
                print(f"  batch failed after {RETRIES} attempts: {e}")
                return pd.DataFrame()
            time.sleep(attempt)
    return pd.DataFrame()


# -- Main ---------------------------------------------------------------------

def build() -> pd.DataFrame:
    atm       = get_atm_strike()
    strikes   = make_strikes(atm)
    contracts = active_contracts()

    all_batches = []
    for mc, yr in contracts:
        syms = make_symbols(mc, yr, strikes)
        for i in range(0, len(syms), BATCH_SIZE):
            all_batches.append(syms[i: i + BATCH_SIZE])

    print(f"Fetching last 10 days ({FETCH_FROM} to {TODAY})")
    print(f"Contracts: {contracts}")
    print(f"ATM {atm} +/-{ATM_WING} (step {STRIKE_STEP}) | Batches: {len(all_batches)}")

    parts = []
    for i, batch in enumerate(all_batches, 1):
        df = fetch_batch(batch)
        if not df.empty:
            parts.append(df)
        print(f"  {i}/{len(all_batches)} batches done ({len(df)} rows)", flush=True)

    if not parts:
        print("No data returned.")
        return pd.DataFrame()

    new_df = pd.concat(parts, ignore_index=True)
    new_df = new_df.rename(columns={"Time": "date"})

    new_df["oi"] = pd.to_numeric(new_df["oi"], errors="coerce")
    new_df = new_df.dropna(subset=["settle"])

    meta_cache: dict = {}
    def get_meta(ric):
        if ric not in meta_cache:
            meta_cache[ric] = parse_symbol(ric)
        return meta_cache[ric] or {}

    new_df["option_type"]  = new_df["ric"].map(lambda r: get_meta(r).get("option_type"))
    new_df["strike"]       = new_df["ric"].map(lambda r: get_meta(r).get("strike"))
    new_df["expiry_month"] = new_df["ric"].map(lambda r: get_meta(r).get("expiry_month"))
    new_df["expiry_year"]  = new_df["ric"].map(lambda r: get_meta(r).get("expiry_year"))
    new_df = new_df.dropna(subset=["option_type", "strike"])

    new_df["date"]         = pd.to_datetime(new_df["date"])
    new_df["settle"]       = new_df["settle"].astype("Float64")
    new_df["oi"]           = new_df["oi"].astype("Int64")
    new_df["volume"]       = new_df["volume"].astype("Int64")
    new_df["strike"]       = new_df["strike"].astype(float)
    new_df["expiry_month"] = new_df["expiry_month"].astype("int64")
    new_df["expiry_year"]  = new_df["expiry_year"].astype("int64")
    new_df = new_df[["date", "settle", "oi", "volume", "ric",
                     "option_type", "strike", "expiry_month", "expiry_year"]]

    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        cutoff   = pd.Timestamp(FETCH_FROM)
        existing = existing[existing["date"] < cutoff]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ric", "date"], keep="last")
    else:
        combined = new_df

    return combined.sort_values(["ric", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    print("LCC Options -- ICE Ingest\n" + "=" * 40)
    df = build()

    if df.empty:
        print("Nothing to save.")
    else:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)
        print(f"\nSaved : {OUT_PATH}")
        print(f"Rows  : {len(df):,}")
        print(f"RICs  : {df['ric'].nunique():,}")
        print(f"Dates : {df['date'].min().date()} to {df['date'].max().date()}")
        months = sorted(df[["expiry_month", "expiry_year"]].drop_duplicates().apply(tuple, axis=1))
        print(f"Months: {months}")
        print(f"Strikes (GBP/t): {sorted(df['strike'].unique())[:5]} ... {sorted(df['strike'].unique())[-5:]}")
