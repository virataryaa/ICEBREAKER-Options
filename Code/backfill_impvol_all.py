"""
backfill_impvol_all.py -- One-time ImpVol backfill for CC, SB, CT options parquets.
Fetches last 60 days of Implied Volatility from ICE get_timeseries
and stamps it onto every matching (ric, date) row in each parquet.
"""

import datetime
import time
from pathlib import Path

import icepython as ice
import pandas as pd

DB = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Options\Database")

TARGETS = [
    ("CC", DB / "CC_options_ice.parquet"),
    ("SB", DB / "SB_options_ice.parquet"),
    ("CT", DB / "CT_options_ice.parquet"),
]

BATCH_SIZE = 100
TODAY      = datetime.date.today().isoformat()
START      = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()


def backfill(label: str, path: Path):
    print(f"\n{'='*55}")
    print(f"  {label} Options ImpVol Backfill")
    print(f"{'='*55}")

    if not path.exists():
        print(f"  [SKIP] {path} does not exist.")
        return

    opt = pd.read_parquet(path)
    print(f"  Parquet rows : {len(opt):,}")
    print(f"  Date range   : {opt['date'].min().date()} -> {opt['date'].max().date()}")
    print(f"  Fetch window : {START} -> {TODAY}")

    symbols = opt["ric"].unique().tolist()
    print(f"  Unique RICs  : {len(symbols)}")

    records = []
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        try:
            raw = ice.get_timeseries(
                batch, ["Implied Volatility"], granularity="D",
                start_date=START, end_date=TODAY
            )
            if not raw or len(raw) < 2:
                done = min(i + BATCH_SIZE, len(symbols))
                print(f"  {done}/{len(symbols)} — no data in response", flush=True)
                continue

            df = pd.DataFrame(list(raw[1:]), columns=raw[0])
            df["Time"] = pd.to_datetime(df["Time"])

            for sym in batch:
                col = f"{sym}.Implied Volatility"
                if col not in df.columns:
                    continue
                tmp = df[["Time", col]].copy()
                tmp.columns = ["date", "impvol"]
                tmp["impvol"] = pd.to_numeric(tmp["impvol"], errors="coerce")
                tmp = tmp.dropna(subset=["impvol"])
                tmp["ric"] = sym
                records.append(tmp)

        except Exception as e:
            print(f"  [ERROR] batch {i // BATCH_SIZE + 1}: {e}")

        done = min(i + BATCH_SIZE, len(symbols))
        hits = sum(len(r) for r in records)
        print(f"  {done}/{len(symbols)} symbols  |  {hits:,} IV rows so far", flush=True)
        time.sleep(0.2)

    if not records:
        print(f"  No ImpVol data returned for {label}. Skipping save.")
        return

    iv_df = pd.concat(records, ignore_index=True)
    iv_df["date"] = pd.to_datetime(iv_df["date"])

    print(f"\n  IV rows fetched  : {len(iv_df):,}")
    print(f"  RICs with data   : {iv_df['ric'].nunique()}")
    print(f"  IV date range    : {iv_df['date'].min().date()} -> {iv_df['date'].max().date()}")
    print(f"  IV range         : {iv_df['impvol'].min():.2f}% -> {iv_df['impvol'].max():.2f}%")

    if "impvol" in opt.columns:
        opt = opt.drop(columns=["impvol"])

    opt["date"] = pd.to_datetime(opt["date"])
    merged = opt.merge(iv_df[["ric", "date", "impvol"]], on=["ric", "date"], how="left")

    filled  = merged["impvol"].notna().sum()
    missing = merged["impvol"].isna().sum()
    print(f"\n  Rows matched     : {filled:,} / {len(merged):,}")
    print(f"  Rows without IV  : {missing:,}  (expired/delisted options)")

    merged.to_parquet(path, index=False)
    print(f"  Saved -> {path}")


if __name__ == "__main__":
    print(f"ImpVol Backfill — CC / SB / CT")
    print(f"Fetch window: {START} -> {TODAY}")

    for label, path in TARGETS:
        backfill(label, path)

    print("\n\nAll done.")
