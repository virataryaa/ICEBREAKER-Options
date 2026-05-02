"""
app.py — Soft Options Dashboard (ICE Connect data)
===================================================
Commodities : KC (Coffee C) | CC (Cocoa) | SB (Sugar #11)
Sidebar     : Old Date + New Date (shared)
Each Tab    : Min OI + ATM info + butterfly tables
Inner Tab 1 : OI Change (left) | Volume (right)
Inner Tab 2 : Px Change (left) | % Change (right)
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Options Dashboard", layout="wide")

DB_PATH  = Path(__file__).parent.parent / "Database"
ATM_JSON = Path(__file__).parent / "atm.json"

MONTH_NAMES   = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
CALL_CODES    = {1:"A",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G",8:"H",9:"I",10:"J",11:"K",12:"L"}
PUT_CODES     = {1:"M",2:"N",3:"O",4:"P",5:"Q",6:"R",7:"S",8:"T",9:"U",10:"V",11:"W",12:"X"}
MONTH_TO_CODE = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_kc():
    df = pd.read_parquet(DB_PATH / "KC_options_ice.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=1800)
def load_cc():
    df = pd.read_parquet(DB_PATH / "CC_options_ice.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=1800)
def load_sb():
    df = pd.read_parquet(DB_PATH / "SB_options_ice.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=1800)
def load_ct():
    df = pd.read_parquet(DB_PATH / "CT_options_ice.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_atm():
    try:
        with open(ATM_JSON) as f:
            return json.load(f)
    except Exception:
        return {}

def _try_load(fn, name):
    try:
        return fn()
    except Exception as e:
        st.warning(f"Could not load {name} data: {e}")
        return pd.DataFrame()


df_kc    = _try_load(load_kc,  "KC")
df_cc    = _try_load(load_cc,  "CC")
df_sb    = _try_load(load_sb,  "SB")
df_ct    = _try_load(load_ct,  "CT")
atm_data = load_atm()

all_dates = set()
for _df in [df_kc, df_cc, df_sb, df_ct]:
    if not _df.empty:
        all_dates.update(_df["date"].dt.date.unique())
available_dates = sorted(all_dates)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Options Dashboard")
    st.divider()
    old_date = st.selectbox("Old Date", available_dates,
                             index=max(0, len(available_dates) - 10),
                             format_func=lambda d: d.strftime("%d %b %Y"))
    new_date = st.selectbox("New Date", available_dates,
                             index=len(available_dates) - 1,
                             format_func=lambda d: d.strftime("%d %b %Y"))
    if old_date == new_date:
        st.warning("Old Date and New Date are the same.")


# ── Pivot helpers (all parameterised) ─────────────────────────────────────────
def _month_keys(df):
    return (df[["expiry_month", "expiry_year"]]
            .drop_duplicates()
            .sort_values(["expiry_year", "expiry_month"])
            .apply(lambda r: (int(r.expiry_month), int(r.expiry_year)), axis=1)
            .tolist())

def _meta(df, opt):
    return (df[df["option_type"] == opt]
            [["ric", "strike", "expiry_month", "expiry_year"]]
            .drop_duplicates()
            .assign(mk=lambda x: list(zip(x.expiry_month.astype(int), x.expiry_year.astype(int))))
            .set_index("ric"))

def _clean(pivot, month_keys):
    if pivot.empty:
        return pivot
    pivot = pivot.reindex(columns=month_keys)
    return pivot.apply(lambda c: pd.to_numeric(c, errors="coerce")).astype(float)

def _valid(df, opt, new_date, min_oi):
    if min_oi <= 0:
        return None
    d2 = df[(df["date"].dt.date == new_date) & (df["option_type"] == opt)][["ric", "oi"]]
    return d2[pd.to_numeric(d2["oi"], errors="coerce") >= min_oi]["ric"]

def _change_pivot(df, month_keys, opt, src, old_date, new_date, min_oi):
    d1 = (df[(df["date"].dt.date == old_date) & (df["option_type"] == opt)]
          [["ric", src]].set_index("ric"))
    d2 = (df[(df["date"].dt.date == new_date) & (df["option_type"] == opt)]
          [["ric", src]].set_index("ric"))
    merged = d1.join(d2, how="outer", lsuffix="_1", rsuffix="_2")
    merged["val"] = (pd.to_numeric(merged[src + "_2"], errors="coerce")
                     - pd.to_numeric(merged[src + "_1"], errors="coerce"))
    v = _valid(df, opt, new_date, min_oi)
    if v is not None:
        merged = merged[merged.index.isin(v)]
    meta = _meta(df, opt)
    result = merged.join(meta[["strike", "mk"]]).dropna(subset=["strike"])
    result = result[result["mk"].notna()]
    piv = result.pivot_table(index="strike", columns="mk", values="val", aggfunc="first")
    return _clean(piv, month_keys).sort_index(ascending=False)

def get_oi_pivot(df, month_keys, opt, old_date, new_date, min_oi):
    return _change_pivot(df, month_keys, opt, "oi", old_date, new_date, min_oi)

def get_px_pivot(df, month_keys, opt, old_date, new_date, min_oi):
    return _change_pivot(df, month_keys, opt, "settle", old_date, new_date, min_oi)

def get_vol_pivot(df, month_keys, opt, old_date, new_date, min_oi):
    lo, hi = min(old_date, new_date), max(old_date, new_date)
    sub = df[(df["option_type"] == opt)
             & (df["date"].dt.date >= lo)
             & (df["date"].dt.date <= hi)].copy()
    v = _valid(df, opt, new_date, min_oi)
    if v is not None:
        sub = sub[sub["ric"].isin(v)]
    sub["mk"] = list(zip(sub["expiry_month"].astype(int), sub["expiry_year"].astype(int)))
    sub["volume"] = pd.to_numeric(sub["volume"], errors="coerce")
    piv = sub.groupby(["strike", "mk"])["volume"].sum().unstack("mk")
    return _clean(piv, month_keys).sort_index(ascending=False)

def get_pct_pivot(df, month_keys, opt, old_date, new_date, min_oi):
    d1 = (df[(df["date"].dt.date == old_date) & (df["option_type"] == opt)]
          [["ric", "settle"]].set_index("ric"))
    d2 = (df[(df["date"].dt.date == new_date) & (df["option_type"] == opt)]
          [["ric", "settle"]].set_index("ric"))
    merged = d1.join(d2, how="outer", lsuffix="_1", rsuffix="_2")
    s1 = pd.to_numeric(merged["settle_1"], errors="coerce")
    s2 = pd.to_numeric(merged["settle_2"], errors="coerce")
    merged["val"] = ((s2 - s1) / s1.abs()) * 100
    v = _valid(df, opt, new_date, min_oi)
    if v is not None:
        merged = merged[merged.index.isin(v)]
    meta = _meta(df, opt)
    result = merged.join(meta[["strike", "mk"]]).dropna(subset=["strike"])
    result = result[result["mk"].notna()]
    piv = result.pivot_table(index="strike", columns="mk", values="val", aggfunc="first")
    return _clean(piv, month_keys).sort_index(ascending=False)

def get_oi_snapshot_pivot(df, month_keys, opt, snap_date, new_date, min_oi):
    d = (df[(df["date"].dt.date == snap_date) & (df["option_type"] == opt)]
         [["ric", "oi"]].set_index("ric"))
    d = d.copy()
    d["oi"] = pd.to_numeric(d["oi"], errors="coerce")
    v = _valid(df, opt, new_date, min_oi)
    if v is not None:
        d = d[d.index.isin(v)]
    meta = _meta(df, opt)
    result = d.join(meta[["strike", "mk"]]).dropna(subset=["strike"])
    result = result[result["mk"].notna()]
    piv = result.pivot_table(index="strike", columns="mk", values="oi", aggfunc="first")
    return _clean(piv, month_keys).sort_index(ascending=False)


# ── Colors ─────────────────────────────────────────────────────────────────────
def _alpha(v, mx): return round(0.15 + min(abs(float(v)) / max(mx, 0.01), 1.0) * 0.50, 2)

def oi_color(val, mx):
    if pd.isna(val) or val == 0: return ""
    a = _alpha(val, mx)
    return (f"background:rgba(66,133,244,{a});color:#1a1a2e" if val > 0
            else f"background:rgba(220,75,75,{a});color:#1a1a2e")

def vol_color(val, mx):
    if pd.isna(val) or val == 0: return ""
    a = _alpha(val, mx)
    return f"background:rgba(66,133,244,{a});color:#1a1a2e"

def px_color(val, mx):
    if pd.isna(val) or val == 0: return ""
    a = _alpha(val, mx)
    return (f"background:rgba(52,168,83,{a});color:#1a1a2e" if val > 0
            else f"background:rgba(220,75,75,{a});color:#1a1a2e")


# ── Butterfly HTML ─────────────────────────────────────────────────────────────
_CSS = """<style>
.bft{border-collapse:collapse;font-size:11px;font-family:-apple-system,sans-serif}
.bft th,.bft td{white-space:nowrap;padding:2px 5px}
.bft th{font-weight:600;letter-spacing:.03em;font-size:10px;text-align:center}
.bft td{text-align:right;border:1px solid #f0f0f0;color:#1a1a2e}
.bft .sc{text-align:center;font-weight:700;font-size:11px;color:#1a1a2e;
         background:#f5f5f5;border-left:2px solid #ccc;border-right:2px solid #ccc}
.bft .sc-atm{background:#f59e0b!important;color:#1a1a2e!important;font-weight:900!important}
.bft tr.atm-row td{border-top:2px solid #f59e0b!important;border-bottom:2px solid #f59e0b!important}
.bft tfoot td{font-weight:700;border-top:2px solid #bbb}
.bft tfoot .sc{font-size:9px;color:#888;background:#efefef}
.ch{background:#dce8fb;color:#1a56cc}
.ph{background:#fde8e8;color:#c0392b}
.kch{background:#ebebeb;color:#555}
</style>"""


def butterfly_html(cpiv, ppiv, atm, cfn, month_keys, fmt="{:.0f}",
                   footer=True, sfx="", title="", atm_tol=None, fixed_strikes=None):
    ccols = list(reversed(month_keys))
    pcols = list(month_keys)

    if fixed_strikes is not None:
        strikes = list(fixed_strikes)  # caller controls order (asc = low at top, ATM centered)
    else:
        strikes_set = set()
        if not cpiv.empty: strikes_set.update(cpiv.index.tolist())
        if not ppiv.empty: strikes_set.update(ppiv.index.tolist())
        strikes = sorted(strikes_set)  # low to high

    if atm_tol is None:
        if len(strikes) >= 2:
            gaps = [abs(strikes[i] - strikes[i+1]) for i in range(len(strikes)-1)]
            atm_tol = min(gaps) * 0.6
        else:
            atm_tol = 1.0

    def _flat(p):
        if p.empty: return np.array([], dtype=float)
        return p.values.astype(float).flatten()

    av = np.concatenate([_flat(cpiv), _flat(ppiv)])
    av = av[~np.isnan(av)]
    mx = float(np.max(np.abs(av))) if len(av) > 0 else 1.0

    nc, np_ = len(ccols), len(pcols)

    h1 = (f'<tr><th colspan="{nc}" class="ch">Call</th>'
          f'<th class="kch">{title}</th>'
          f'<th colspan="{np_}" class="ph">Put</th></tr>')

    h2 = ('<tr>'
          + "".join(f'<th class="ch" style="color:#999;font-weight:400">'
                    f'{CALL_CODES[m]}{str(y)[-2:]}</th>' for m, y in ccols)
          + '<th class="kch"></th>'
          + "".join(f'<th class="ph" style="color:#ccc;font-weight:400">'
                    f'{PUT_CODES[m]}{str(y)[-2:]}</th>' for m, y in pcols)
          + '</tr>')

    h3 = ('<tr>'
          + "".join(f'<th class="ch">{MONTH_NAMES[m]}</th>' for m, y in ccols)
          + '<th class="kch"></th>'
          + "".join(f'<th class="ph">{MONTH_NAMES[m]}</th>' for m, y in pcols)
          + '</tr>')

    def cv(piv, s, mk):
        if piv.empty or s not in piv.index or mk not in piv.columns: return np.nan
        v = piv.at[s, mk]
        return float(v) if not pd.isna(v) else np.nan

    def td(v):
        style = cfn(v, mx)
        txt = (fmt.format(v) + sfx) if not np.isnan(v) and v != 0 else ""
        return f'<td style="{style}">{txt}</td>'

    body = []
    for s in strikes:
        is_atm = atm is not None and abs(s - atm) < atm_tol
        sc     = "sc sc-atm" if is_atm else "sc"
        tr_cls = ' class="atm-row"' if is_atm else ""
        lbl    = int(s) if s == int(s) else s
        row = ("".join(td(cv(cpiv, s, mk)) for mk in ccols)
               + f'<td class="{sc}">{lbl}</td>'
               + "".join(td(cv(ppiv, s, mk)) for mk in pcols))
        body.append(f"<tr{tr_cls}>{row}</tr>")

    ft = ""
    if footer:
        def cs(piv, mk):
            if piv.empty or mk not in piv.columns: return 0.0
            return float(piv[mk].sum(skipna=True))
        cft = "".join(td(cs(cpiv, mk)) for mk in ccols)
        pft = "".join(td(cs(ppiv, mk)) for mk in pcols)
        ft = (f'<tfoot><tr>{cft}'
              f'<td class="sc" style="font-size:9px;color:#888">TOT</td>'
              f'{pft}</tr></tfoot>')

    est_h = max(400, (len(strikes) + 4) * 22 + 90)
    return (f'{_CSS}<div style="overflow-x:auto;overflow-y:auto;max-height:{est_h}px">'
            f'<table class="bft"><thead>{h1}{h2}{h3}</thead>'
            f'{ft}<tbody>{"".join(body)}</tbody></table></div>')


# ── Misc helpers ───────────────────────────────────────────────────────────────
def _tot(piv): return float(piv.sum(skipna=True).sum()) if not piv.empty else 0.0
def _fn(v, f="{:,.0f}"):
    try: return f.format(float(v))
    except: return "—"

def _ric_kc(strike, month, year, opt):
    mc  = MONTH_TO_CODE[month]
    yy  = f"{year % 100:02d}"
    cp  = "C" if opt == "Call" else "P"
    return f"KC {mc}{yy}{cp}{int(round(strike * 10))}"

def _ric_cc(strike, month, year, opt):
    """Strike stored as $/mt; convert back to $/cwt integer for the symbol."""
    mc  = MONTH_TO_CODE[month]
    yy  = f"{year % 100:02d}"
    cp  = "C" if opt == "Call" else "P"
    return f"CC {mc}{yy}{cp}{int(round(strike / 22.046))}"

def _ric_sb(strike, month, year, opt):
    """Strike stored as cts/lb; multiply by 100 for symbol integer."""
    mc  = MONTH_TO_CODE[month]
    yy  = f"{year % 100:02d}"
    cp  = "C" if opt == "Call" else "P"
    return f"SB {mc}{yy}{cp}{int(round(strike * 100))}"

def _ric_ct(strike, month, year, opt):
    """Strike stored as cts/lb integer directly."""
    mc  = MONTH_TO_CODE[month]
    yy  = f"{year % 100:02d}"
    cp  = "C" if opt == "Call" else "P"
    return f"CT {mc}{yy}{cp}{int(round(strike))}"


# ── Commodity tab renderer ─────────────────────────────────────────────────────
def render_commodity_tab(df, atm_val, atm_label, old_date, new_date,
                         key_prefix, title, ric_fn):
    if df.empty:
        st.info(f"No data available for {title}.")
        return

    month_keys       = _month_keys(df)
    all_strikes_data = sorted(df["strike"].unique())  # ascending, for step inference
    atm_updated      = atm_data.get("updated", "—")

    # Build ±35-strike centered display window around ATM (descending = high at top)
    if atm_val is not None and len(all_strikes_data) > 1:
        diffs = [all_strikes_data[i+1] - all_strikes_data[i]
                 for i in range(len(all_strikes_data)-1)]
        step  = sorted(diffs)[len(diffs)//2]  # median step (robust to outliers)
        snap  = {}
        for s in all_strikes_data:
            bucket = round((s - atm_val) / step)
            if bucket not in snap or abs(s - atm_val) < abs(snap[bucket] - atm_val):
                snap[bucket] = s
        N = 35
        all_strikes = sorted([snap[b] for b in range(-N, N+1) if b in snap])
        if not all_strikes:
            all_strikes = sorted(all_strikes_data)
    else:
        all_strikes = sorted(all_strikes_data)

    col_oi, col_atm = st.columns([1, 3])
    with col_oi:
        min_oi = st.number_input("Min OI filter (New Date)", value=0, min_value=0,
                                  step=10, key=f"{key_prefix}_min_oi")
    with col_atm:
        st.caption(
            f"ATM ({title}): **{atm_label}** as of {atm_updated} | "
            f"Data: {df['date'].min().date()} to {df['date'].max().date()}"
        )

    call_oi  = get_oi_pivot(df, month_keys, "Call", old_date, new_date, min_oi)
    put_oi   = get_oi_pivot(df, month_keys, "Put",  old_date, new_date, min_oi)
    call_vol = get_vol_pivot(df, month_keys, "Call", old_date, new_date, min_oi)
    put_vol  = get_vol_pivot(df, month_keys, "Put",  old_date, new_date, min_oi)

    c_oi  = _tot(call_oi);  p_oi  = _tot(put_oi)
    c_vol = _tot(call_vol); p_vol = _tot(put_vol)
    cp_oi  = f"{abs(c_oi/p_oi):.2f}"  if p_oi  != 0 else "—"
    cp_vol = f"{c_vol/p_vol:.2f}"     if p_vol > 0  else "—"

    items = [
        ("Call OI Delta", _fn(c_oi)),
        ("Put OI Delta",  _fn(p_oi)),
        ("Call Volume",   _fn(c_vol)),
        ("Put Volume",    _fn(p_vol)),
        ("C/P OI Ratio",  cp_oi),
        ("C/P Vol Ratio", cp_vol),
    ]
    st.markdown(
        '<div style="display:flex;gap:28px;padding:6px 0 12px;border-bottom:1px solid #eee;flex-wrap:wrap">'
        + "".join(
            f'<div><div style="font-size:9px;color:#888;letter-spacing:.07em;'
            f'text-transform:uppercase;margin-bottom:2px">{lbl}</div>'
            f'<div style="font-size:14px;font-weight:600;color:#1a1a2e">{val}</div></div>'
            for lbl, val in items
        )
        + '</div>',
        unsafe_allow_html=True
    )

    inner1, inner2 = st.tabs([f"OI Change + Volume", f"Px Change"])

    with inner1:
        cl, cr = st.columns(2)
        with cl:
            st.markdown("**OI Change**")
            st.markdown(
                butterfly_html(call_oi, put_oi, atm_val, oi_color, month_keys,
                               fmt="{:.0f}", footer=True, title=title,
                               fixed_strikes=all_strikes),
                unsafe_allow_html=True)
        with cr:
            st.markdown("**Volume**")
            st.markdown(
                butterfly_html(call_vol, put_vol, atm_val, vol_color, month_keys,
                               fmt="{:.0f}", footer=True, title=title,
                               fixed_strikes=all_strikes),
                unsafe_allow_html=True)

        with st.expander("OI Snapshot — Old Date vs New Date"):
            call_oi_old = get_oi_snapshot_pivot(df, month_keys, "Call", old_date, new_date, min_oi)
            put_oi_old  = get_oi_snapshot_pivot(df, month_keys, "Put",  old_date, new_date, min_oi)
            call_oi_new = get_oi_snapshot_pivot(df, month_keys, "Call", new_date, new_date, min_oi)
            put_oi_new  = get_oi_snapshot_pivot(df, month_keys, "Put",  new_date, new_date, min_oi)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**Old Date — {old_date.strftime('%d %b %Y')}**")
                st.markdown(
                    butterfly_html(call_oi_old, put_oi_old, atm_val, vol_color, month_keys,
                                   fmt="{:.0f}", footer=False, title=title,
                                   fixed_strikes=all_strikes),
                    unsafe_allow_html=True)
            with sc2:
                st.markdown(f"**New Date — {new_date.strftime('%d %b %Y')}**")
                st.markdown(
                    butterfly_html(call_oi_new, put_oi_new, atm_val, vol_color, month_keys,
                                   fmt="{:.0f}", footer=False, title=title,
                                   fixed_strikes=all_strikes),
                    unsafe_allow_html=True)

        with st.expander("Drill Down — Single Option Time Series"):
            call_dd_piv = get_oi_snapshot_pivot(df, month_keys, "Call", new_date, new_date, min_oi)
            put_dd_piv  = get_oi_snapshot_pivot(df, month_keys, "Put",  new_date, new_date, min_oi)

            col_labels = {mk: f"{MONTH_NAMES[mk[0]]} '{str(mk[1])[-2:]}" for mk in month_keys}
            mk_lookup  = {v: k for k, v in col_labels.items()}

            def _flat_list(piv):
                rows = []
                for strike in sorted(piv.index):
                    for mk in month_keys:
                        if mk not in piv.columns:
                            continue
                        try:
                            v = float(piv.at[strike, mk])
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(v) or v <= 0:
                            continue
                        rows.append({"Strike": strike, "Expiry": col_labels[mk], "OI": int(v)})
                return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Strike", "Expiry", "OI"])

            def _style_oi(s, rgb):
                mx = s.max() if len(s) > 0 else 1.0
                if pd.isna(mx) or mx == 0: mx = 1.0
                return [f"background-color:rgba({rgb},{round(0.15+min(v/mx,1.0)*0.5,2)});color:#1a1a2e"
                        if pd.notna(v) and v > 0 else "" for v in s]

            call_flat = _flat_list(call_dd_piv)
            put_flat  = _flat_list(put_dd_piv)

            all_expiries = [col_labels[mk] for mk in month_keys]
            fc1, fc2 = st.columns([1, 3])
            with fc1:
                exp_filter = st.selectbox("Filter by Expiry", ["All"] + all_expiries,
                                          key=f"{key_prefix}_dd_exp_filter")

            if exp_filter != "All":
                call_show = call_flat[call_flat["Expiry"] == exp_filter].reset_index(drop=True)
                put_show  = put_flat[put_flat["Expiry"]  == exp_filter].reset_index(drop=True)
            else:
                call_show, put_show = call_flat, put_flat

            st.caption(f"OI as of **{new_date.strftime('%d %b %Y')}** — click a row to view its time series")
            ddc1, ddc2 = st.columns(2)

            def _fmt_strike(x):
                return f"{x:.1f}" if x % 1 != 0 else f"{int(x)}"

            with ddc1:
                st.markdown("**Calls**")
                call_evt = st.dataframe(
                    call_show.style.apply(_style_oi, rgb="66,133,244", subset=["OI"])
                             .format({"Strike": _fmt_strike, "OI": "{:,}"}),
                    on_select="rerun", selection_mode="single-row",
                    key=f"{key_prefix}_dd_call", use_container_width=True, hide_index=True,
                )
            with ddc2:
                st.markdown("**Puts**")
                put_evt = st.dataframe(
                    put_show.style.apply(_style_oi, rgb="220,75,75", subset=["OI"])
                            .format({"Strike": _fmt_strike, "OI": "{:,}"}),
                    on_select="rerun", selection_mode="single-row",
                    key=f"{key_prefix}_dd_put", use_container_width=True, hide_index=True,
                )

            sel_type = sel_strike = sel_mk = None
            c_rows = call_evt.selection.get("rows", [])
            p_rows = put_evt.selection.get("rows", [])

            if c_rows and not call_show.empty:
                row = call_show.iloc[c_rows[0]]
                sel_type, sel_strike, sel_mk = "Call", row["Strike"], mk_lookup.get(row["Expiry"])
            elif p_rows and not put_show.empty:
                row = put_show.iloc[p_rows[0]]
                sel_type, sel_strike, sel_mk = "Put", row["Strike"], mk_lookup.get(row["Expiry"])

            if sel_type and sel_strike is not None and sel_mk:
                ric = ric_fn(sel_strike, sel_mk[0], sel_mk[1], sel_type)
                rdf = df[df["ric"] == ric].sort_values("date")
                strike_lbl = _fmt_strike(sel_strike)
                exp_lbl    = f"{MONTH_NAMES[sel_mk[0]]} '{str(sel_mk[1])[-2:]}"
                friendly   = f"{title} {exp_lbl} {strike_lbl} {sel_type} ({ric})"
                st.caption(f"**{friendly}** — {len(rdf)} trading days")
                if rdf.empty:
                    st.info(f"No data for {ric}")
                else:
                    cc1, cc2, cc3 = st.columns(3)
                    for col, field, label in [
                        (cc1, "oi", "Open Interest"), (cc2, "volume", "Volume"),
                        (cc3, "settle", "Settle Price"),
                    ]:
                        s = pd.to_numeric(rdf.set_index("date")[field], errors="coerce").dropna()
                        if not s.empty:
                            col.markdown(f"**{label}**")
                            if field == "volume":
                                col.bar_chart(s)
                            else:
                                col.line_chart(s)
            else:
                st.caption("Click any row above to view its time series.")

        with st.expander("OI & Volume Time Series — All Strikes"):
            all_d = sorted(df["date"].dt.date.unique())
            if len(all_d) >= 2:
                dr = st.slider("Date Range", min_value=all_d[0], max_value=all_d[-1],
                               value=(all_d[0], all_d[-1]), key=f"{key_prefix}_ts_dr")
                sub = df[(df["date"].dt.date >= dr[0]) & (df["date"].dt.date <= dr[1])].copy()
                daily = (sub.groupby(["date", "option_type"])
                         .agg(oi=("oi", "sum"), volume=("volume", "sum"))
                         .reset_index())
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown("**Call / Put OI**")
                    oi_w = daily.pivot(index="date", columns="option_type", values="oi")
                    oi_w.columns.name = None
                    st.line_chart(oi_w.rename(columns={"Call": "Call OI", "Put": "Put OI"}))
                with tc2:
                    st.markdown("**Call / Put Volume**")
                    vol_w = daily.pivot(index="date", columns="option_type", values="volume")
                    vol_w.columns.name = None
                    st.line_chart(vol_w.rename(columns={"Call": "Call Vol", "Put": "Put Vol"}))

    with inner2:
        call_px  = get_px_pivot(df, month_keys, "Call", old_date, new_date, min_oi)
        put_px   = get_px_pivot(df, month_keys, "Put",  old_date, new_date, min_oi)
        call_pct = get_pct_pivot(df, month_keys, "Call", old_date, new_date, min_oi)
        put_pct  = get_pct_pivot(df, month_keys, "Put",  old_date, new_date, min_oi)

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Px Change**")
            st.markdown(
                butterfly_html(call_px, put_px, atm_val, px_color, month_keys,
                               fmt="{:.2f}", footer=False, title=title,
                               fixed_strikes=all_strikes),
                unsafe_allow_html=True)
        with pc2:
            st.markdown("**% Change**")
            st.markdown(
                butterfly_html(call_pct, put_pct, atm_val, px_color, month_keys,
                               fmt="{:.1f}", footer=False, sfx="%", title=title,
                               fixed_strikes=all_strikes),
                unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
st.title("Options Dashboard")
st.caption(
    f"Old Date: **{old_date.strftime('%d %b %Y')}**  |  "
    f"New Date: **{new_date.strftime('%d %b %Y')}**"
)

tab_kc, tab_cc, tab_sb, tab_ct = st.tabs(["KC — Coffee C", "CC — Cocoa", "SB — Sugar #11", "CT — Cotton"])

atm_kc  = atm_data.get("KC")
atm_cc  = atm_data.get("CC")
atm_sb  = atm_data.get("SB")
atm_ct  = atm_data.get("CT")

with tab_kc:
    atm_kc_lbl = (f"{int(atm_kc) if atm_kc == int(atm_kc) else atm_kc}"
                  if atm_kc is not None else "—")
    render_commodity_tab(
        df=df_kc,
        atm_val=atm_kc,
        atm_label=atm_kc_lbl,
        old_date=old_date,
        new_date=new_date,
        key_prefix="kc",
        title="KC",
        ric_fn=_ric_kc,
    )

with tab_cc:
    atm_cc_lbl = f"{int(atm_cc):,}" if atm_cc is not None else "—"
    render_commodity_tab(
        df=df_cc,
        atm_val=atm_cc,
        atm_label=atm_cc_lbl,
        old_date=old_date,
        new_date=new_date,
        key_prefix="cc",
        title="CC",
        ric_fn=_ric_cc,
    )

with tab_sb:
    atm_sb_lbl = f"{atm_sb:.2f}" if atm_sb is not None else "—"
    render_commodity_tab(
        df=df_sb,
        atm_val=atm_sb,
        atm_label=atm_sb_lbl,
        old_date=old_date,
        new_date=new_date,
        key_prefix="sb",
        title="SB",
        ric_fn=_ric_sb,
    )

with tab_ct:
    atm_ct_lbl = f"{int(atm_ct)}" if atm_ct is not None else "—"
    render_commodity_tab(
        df=df_ct,
        atm_val=atm_ct,
        atm_label=atm_ct_lbl,
        old_date=old_date,
        new_date=new_date,
        key_prefix="ct",
        title="CT",
        ric_fn=_ric_ct,
    )

