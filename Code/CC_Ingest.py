"""
CC_Ingest.py -- CC Cocoa Options (ICE Connect)
Fetches last 10 days, live options only (OI > 0), upserts into parquet.
Strike range: ATM +/- 20 strikes (5pt increments) centred on %CC 1! last settle.
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

OUT_PATH   = Path(__file__).resolve().parent.parent / "Database" / "CC_options_ice.parquet"
ATM_PATH   = Path(__file__).resolve().parent.parent / "Dashboard" / "atm.json"
TODAY      = datetime.date.today().isoformat()
FETCH_FROM = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
RETRIES    = 3
BATCH_SIZE = 50
N_MONTHS   = 12   # next N calendar months starting from next month
ATM_WING   = 20   # strikes each side of ATM
STRIKE_STEP  = 5        # CC ICE symbols use 5pt $/cwt increments
MT_TO_CWT    = 22.046  # 1 metric ton = 22.046 cwt (100lb units)

FIELDS = ["Settle", "Volume", "Open Interest"]

CODE_TO_MONTH = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
MONTH_TO_CODE = {v: k for k, v in CODE_TO_MONTH.items()}

# -- Helpers ------------------------------------------------------------------

def active_contracts() -> list[tuple[str, int]]:
    """Next N_MONTHS calendar months starting from the month after today."""
    today = datetime.date.today()
    month = today.month + 1
    year  = today.year
    if month > 12:
        month, year = 1, year + 1
    pairs = []
    for _ in range(N_MONTHS):
        pairs.append((MONTH_TO_CODE[month], year))
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return pairs


def get_atm_strike() -> float:
    """Fetch %CC 1! last settle and round to nearest STRIKE_STEP increment."""
    raw = ice.get_timeseries(
        ["%CC 1!"], ["Settle"], granularity="D",
        start_date=(datetime.date.today() - datetime.timedelta(days=5)).isoformat(),
        end_date=TODAY
    )
    if not raw or len(raw) < 2:
        raise RuntimeError("Could not fetch %CC 1! price for ATM calculation")
    last_settle_mt = next(
        (float(row[1]) for row in reversed(raw[1:]) if row[1] is not None),
        None
    )
    if last_settle_mt is None:
        raise RuntimeError("No valid settle found for %CC 1!")
    # ICE CC options are quoted in $/cwt; %CC 1! returns $/metric ton
    last_settle_cwt = last_settle_mt / MT_TO_CWT
    atm = round(last_settle_cwt / STRIKE_STEP) * STRIKE_STEP
    print(f"CC 1! last settle: {last_settle_mt:.2f} $/mt  =  {last_settle_cwt:.2f} $/cwt  ->  ATM strike: {atm:.0f} $/cwt")

    # Update atm.json — merge CC into existing keys
    existing = {}
    if ATM_PATH.exists():
        try:
            existing = json.loads(ATM_PATH.read_text())
        except Exception:
            pass
    existing["CC"]      = round(atm * MT_TO_CWT)  # store rounded ATM in $/mt (matches parquet strikes)
    existing["updated"] = TODAY
    ATM_PATH.write_text(json.dumps(existing))
    return atm


def make_strikes(atm: float) -> list[float]:
    """ATM +/- ATM_WING strikes in STRIKE_STEP increments."""
    return [atm + i * STRIKE_STEP for i in range(-ATM_WING, ATM_WING + 1)]


def make_symbols(month_code: str, year: int, strikes: list[float]) -> list[str]:
    yy   = f"{year % 100:02d}"
    syms = []
    for s in strikes:
        r = int(round(s))
        syms.append(f"CC {month_code}{yy}C{r}")
        syms.append(f"CC {month_code}{yy}P{r}")
    return syms


def parse_symbol(sym: str) -> dict | None:
    try:
        parts = sym.strip().split()
        if len(parts) != 2 or parts[0] != "CC":
            return None
        c          = parts[1]
        mc, yy, cp = c[0], c[1:3], c[3]
        strike_raw = int(c[4:])
        if cp not in ("C", "P") or mc not in CODE_TO_MONTH:
            return None
        return {
            "ric":          sym,
            "option_type":  "Call" if cp == "C" else "Put",
            "strike":       round(strike_raw * MT_TO_CWT),   # convert $/cwt -> $/mt
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
    print(f"Contracts: {len(contracts)} | Strikes: ATM {atm:.0f} +/-{ATM_WING} (step {STRIKE_STEP}) | Batches: {len(all_batches)}")

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
    new_df = new_df.dropna(subset=["settle"])  # keep all priced strikes; Min OI filter is in dashboard

    # Parse symbol metadata
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

    # Cast schema
    new_df["date"]         = pd.to_datetime(new_df["date"])
    new_df["settle"]       = new_df["settle"].astype("Float64")
    new_df["oi"]           = new_df["oi"].astype("Int64")
    new_df["volume"]       = new_df["volume"].astype("Int64")
    new_df["strike"]       = new_df["strike"].astype(float)
    new_df["expiry_month"] = new_df["expiry_month"].astype("int64")
    new_df["expiry_year"]  = new_df["expiry_year"].astype("int64")
    new_df = new_df[["date", "settle", "oi", "volume", "ric",
                     "option_type", "strike", "expiry_month", "expiry_year"]]

    # Upsert
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
    print("CC Options -- ICE Ingest\n" + "=" * 40)
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
        print(f"Strikes: {sorted(df['strike'].unique())[:5]} ... {sorted(df['strike'].unique())[-5:]}")
