"""
app.py — KC Options Dashboard (ICE Connect data)
=================================================
Tab 1 : OI Change (left) | Volume (right)  — fully separate side by side
Tab 2 : Px Change (left) | % Change (right)
Top   : compact summary row
Below : collapsible drill-down + aggregate time-series
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="KC Options Dashboard", layout="wide")

PARQUET_PATH = Path(__file__).parent.parent / "Database" / "KC_options_ice.parquet"
ATM_JSON     = Path(__file__).parent / "atm.json"

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
CALL_CODES = {1:"A",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G",8:"H",9:"I",10:"J",11:"K",12:"L"}
PUT_CODES  = {1:"M",2:"N",3:"O",4:"P",5:"Q",6:"R",7:"S",8:"T",9:"U",10:"V",11:"W",12:"X"}
MONTH_TO_CODE = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_atm():
    try:
        with open(ATM_JSON) as f:
            return json.load(f)
    except Exception:
        return {}


try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

atm_data = load_atm()
atm_kc   = atm_data.get("KC")
atm_date = atm_data.get("updated")

available_dates = sorted(df["date"].dt.date.unique())
all_strikes     = sorted(df["strike"].unique(), reverse=True)

month_keys = (
    df[["expiry_month", "expiry_year"]]
    .drop_duplicates()
    .sort_values(["expiry_year", "expiry_month"])
    .apply(lambda r: (int(r.expiry_month), int(r.expiry_year)), axis=1)
    .tolist()
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("KC Options")
    st.caption(f"Data: {df['date'].min().date()} to {df['date'].max().date()}")
    if atm_kc:
        st.caption(f"ATM (KCc2): **{atm_kc}** as of {atm_date}")
    st.divider()
    old_date = st.selectbox("Old Date", available_dates,
                             index=max(0, len(available_dates) - 10),
                             format_func=lambda d: d.strftime("%d %b %Y"))
    new_date = st.selectbox("New Date", available_dates,
                             index=len(available_dates) - 1,
                             format_func=lambda d: d.strftime("%d %b %Y"))
    if old_date == new_date:
        st.warning("Old Date and New Date are the same.")
    st.divider()
    min_oi = st.number_input("Min OI filter (on New Date)", value=0, min_value=0, step=10)


# ── Pivot helpers ──────────────────────────────────────────────────────────────
def _meta(opt):
    return (df[df["option_type"] == opt]
            [["ric", "strike", "expiry_month", "expiry_year"]]
            .drop_duplicates()
            .assign(mk=lambda x: list(zip(x.expiry_month.astype(int), x.expiry_year.astype(int))))
            .set_index("ric"))


def _clean(pivot):
    if pivot.empty:
        return pivot
    pivot = pivot.reindex(columns=month_keys)
    return pivot.apply(lambda c: pd.to_numeric(c, errors="coerce")).astype(float)


def _valid(opt):
    if min_oi <= 0:
        return None
    d2 = df[(df["date"].dt.date == new_date) & (df["option_type"] == opt)][["ric", "oi"]]
    return d2[pd.to_numeric(d2["oi"], errors="coerce") >= min_oi]["ric"]


def _change_pivot(opt, src):
    d1 = (df[(df["date"].dt.date == old_date) & (df["option_type"] == opt)]
          [["ric", src]].set_index("ric"))
    d2 = (df[(df["date"].dt.date == new_date) & (df["option_type"] == opt)]
          [["ric", src]].set_index("ric"))
    merged = d1.join(d2, how="outer", lsuffix="_1", rsuffix="_2")
    merged["val"] = (pd.to_numeric(merged[src + "_2"], errors="coerce")
                     - pd.to_numeric(merged[src + "_1"], errors="coerce"))
    v = _valid(opt)
    if v is not None:
        merged = merged[merged.index.isin(v)]
    meta = _meta(opt)
    result = merged.join(meta[["strike", "mk"]]).dropna(subset=["strike"])
    result = result[result["mk"].notna()]
    piv = result.pivot_table(index="strike", columns="mk", values="val", aggfunc="first")
    return _clean(piv).sort_index(ascending=False)


def get_oi_pivot(opt):  return _change_pivot(opt, "oi")
def get_px_pivot(opt):  return _change_pivot(opt, "settle")


def get_vol_pivot(opt):
    lo, hi = min(old_date, new_date), max(old_date, new_date)
    sub = df[(df["option_type"] == opt)
             & (df["date"].dt.date >= lo)
             & (df["date"].dt.date <= hi)].copy()
    v = _valid(opt)
    if v is not None:
        sub = sub[sub["ric"].isin(v)]
    sub["mk"] = list(zip(sub["expiry_month"].astype(int), sub["expiry_year"].astype(int)))
    sub["volume"] = pd.to_numeric(sub["volume"], errors="coerce")
    piv = sub.groupby(["strike", "mk"])["volume"].sum().unstack("mk")
    return _clean(piv).sort_index(ascending=False)


def get_pct_pivot(opt):
    d1 = (df[(df["date"].dt.date == old_date) & (df["option_type"] == opt)]
          [["ric", "settle"]].set_index("ric"))
    d2 = (df[(df["date"].dt.date == new_date) & (df["option_type"] == opt)]
          [["ric", "settle"]].set_index("ric"))
    merged = d1.join(d2, how="outer", lsuffix="_1", rsuffix="_2")
    s1 = pd.to_numeric(merged["settle_1"], errors="coerce")
    s2 = pd.to_numeric(merged["settle_2"], errors="coerce")
    merged["val"] = ((s2 - s1) / s1.abs()) * 100
    v = _valid(opt)
    if v is not None:
        merged = merged[merged.index.isin(v)]
    meta = _meta(opt)
    result = merged.join(meta[["strike", "mk"]]).dropna(subset=["strike"])
    result = result[result["mk"].notna()]
    piv = result.pivot_table(index="strike", columns="mk", values="val", aggfunc="first")
    return _clean(piv).sort_index(ascending=False)


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
.bft .sc-atm{background:#1a1a2e!important;color:#fff!important}
.bft tfoot td{font-weight:700;border-top:2px solid #bbb}
.bft tfoot .sc{font-size:9px;color:#888;background:#efefef}
.ch{background:#dce8fb;color:#1a56cc}
.ph{background:#fde8e8;color:#c0392b}
.kch{background:#ebebeb;color:#555}
</style>"""


def butterfly_html(cpiv, ppiv, atm, cfn, fmt="{:.0f}", footer=True, sfx="", title="KC"):
    ccols = list(reversed(month_keys))
    pcols = list(month_keys)

    strikes_set = set()
    if not cpiv.empty: strikes_set.update(cpiv.index.tolist())
    if not ppiv.empty: strikes_set.update(ppiv.index.tolist())
    strikes = sorted(strikes_set, reverse=True)

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
        is_atm = atm is not None and abs(s - atm) < 1.25
        sc  = "sc sc-atm" if is_atm else "sc"
        lbl = int(s) if s == int(s) else s
        row = ("".join(td(cv(cpiv, s, mk)) for mk in ccols)
               + f'<td class="{sc}">{lbl}</td>'
               + "".join(td(cv(ppiv, s, mk)) for mk in pcols))
        body.append(f"<tr>{row}</tr>")

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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _tot(piv): return float(piv.sum(skipna=True).sum()) if not piv.empty else 0.0
def _fn(v, f="{:,.0f}"):
    try: return f.format(float(v))
    except: return "—"

def _ric(strike, month, year, opt):
    """Build ICE option symbol: KC M26C2850"""
    mc  = MONTH_TO_CODE[month]
    yy  = f"{year % 100:02d}"
    cp  = "C" if opt == "Call" else "P"
    raw = int(round(strike * 10))
    return f"KC {mc}{yy}{cp}{raw}"


# ── Compute all pivots up-front ────────────────────────────────────────────────
call_oi  = get_oi_pivot("Call")
put_oi   = get_oi_pivot("Put")
call_vol = get_vol_pivot("Call")
put_vol  = get_vol_pivot("Put")

c_oi  = _tot(call_oi);  p_oi  = _tot(put_oi)
c_vol = _tot(call_vol); p_vol = _tot(put_vol)
cp_oi  = f"{abs(c_oi/p_oi):.2f}"   if p_oi  != 0 else "—"
cp_vol = f"{c_vol/p_vol:.2f}"       if p_vol > 0  else "—"


# ── Title + compact summary ────────────────────────────────────────────────────
atm_lbl = (f"{int(atm_kc) if atm_kc == int(atm_kc) else atm_kc}"
           if atm_kc is not None else "—")

st.title("KC Options Dashboard")
st.caption(
    f"Old Date: **{old_date.strftime('%d %b %Y')}** | "
    f"New Date: **{new_date.strftime('%d %b %Y')}** | "
    f"ATM (KCc2): **{atm_lbl}** as of {atm_date or '—'}"
)

items = [
    ("Call OI Delta",   _fn(c_oi)),
    ("Put OI Delta",    _fn(p_oi)),
    ("Call Volume",     _fn(c_vol)),
    ("Put Volume",      _fn(p_vol)),
    ("C/P OI Ratio",    cp_oi),
    ("C/P Vol Ratio",   cp_vol),
]
st.html(
    '<div style="display:flex;gap:28px;padding:8px 0 14px;border-bottom:1px solid #eee;flex-wrap:wrap">'
    + "".join(
        f'<div><div style="font-size:9px;color:#888;letter-spacing:.07em;text-transform:uppercase;margin-bottom:2px">{lbl}</div>'
        f'<div style="font-size:14px;font-weight:600;color:#1a1a2e">{val}</div></div>'
        for lbl, val in items
    )
    + '</div>'
)

tab1, tab2 = st.tabs(["OI Change + Volume", "Px Change"])

# ── Tab 1: OI | Volume side by side ───────────────────────────────────────────
with tab1:
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**OI Change**")
        st.html(butterfly_html(call_oi, put_oi, atm_kc, oi_color, fmt="{:.0f}", footer=True))
    with cr:
        st.markdown("**Volume**")
        st.html(butterfly_html(call_vol, put_vol, atm_kc, vol_color, fmt="{:.0f}", footer=True))

    with st.expander("Drill Down — Single Option Time Series"):
        dc1, dc2, dc3 = st.columns(3)
        sel_s    = dc1.selectbox("Strike", all_strikes, key="dd_s")
        sel_mk   = dc2.selectbox("Expiry", month_keys, key="dd_mk",
                                  format_func=lambda mk: f"{MONTH_NAMES[mk[0]]} {mk[1]}")
        sel_type = dc3.selectbox("Type", ["Call", "Put"], key="dd_t")

        ric = _ric(sel_s, sel_mk[0], sel_mk[1], sel_type)
        rdf = df[df["ric"] == ric].sort_values("date")

        if rdf.empty:
            st.info(f"No data for **{ric}**")
        else:
            st.caption(f"RIC: **{ric}** — {len(rdf)} trading days")
            cc1, cc2, cc3 = st.columns(3)
            for col, field, label in [(cc1,"oi","Open Interest"),(cc2,"volume","Volume"),(cc3,"settle","Settle Price")]:
                s = pd.to_numeric(rdf.set_index("date")[field], errors="coerce").dropna()
                if not s.empty:
                    col.markdown(f"**{label}**")
                    col.line_chart(s)

    with st.expander("OI & Volume Time Series — All Strikes"):
        all_d = sorted(df["date"].dt.date.unique())
        if len(all_d) >= 2:
            dr = st.slider("Date Range", min_value=all_d[0], max_value=all_d[-1],
                           value=(all_d[0], all_d[-1]), key="ts_dr")
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

# ── Tab 2: Px Change | % Change ───────────────────────────────────────────────
with tab2:
    call_px  = get_px_pivot("Call")
    put_px   = get_px_pivot("Put")
    call_pct = get_pct_pivot("Call")
    put_pct  = get_pct_pivot("Put")

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Px Change**")
        st.html(butterfly_html(call_px, put_px, atm_kc, px_color,
                               fmt="{:.2f}", footer=False))
    with pc2:
        st.markdown("**% Change**")
        st.html(butterfly_html(call_pct, put_pct, atm_kc, px_color,
                               fmt="{:.1f}", footer=False, sfx="%"))
