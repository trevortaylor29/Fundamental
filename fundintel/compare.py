# fundintel/compare.py
import pandas as pd
from typing import Dict
from pandas.tseries.offsets import DateOffset
from . import data as d

# -------- helpers (calendar-based, percent returns) --------

def _tz_naive_index(c: pd.Series) -> pd.Series:
    """Ensure tz-naive index (prevents tz-aware vs tz-naive slicing)."""
    idx = c.index
    try:
        if getattr(idx, "tz", None) is not None:
            c.index = idx.tz_convert("UTC").tz_localize(None)
    except Exception:
        try:
            c.index = c.index.tz_localize(None)
        except Exception:
            pass
    return c

def _close_series(px: pd.DataFrame) -> pd.Series:
    if px is None or px.empty or "Close" not in px:
        return pd.Series(dtype="float64")
    c = px["Close"].dropna()
    if c.empty:
        return pd.Series(dtype="float64")
    return _tz_naive_index(c)

def _period_return_from_date(px: pd.DataFrame, start_date: pd.Timestamp) -> float:
    """
    Percent return from the first trading day >= start_date to the last available close.
    Returns a decimal (e.g., 0.237 for +23.7%).
    """
    c = _close_series(px)
    if c.empty:
        return float("nan")
    start_date = pd.to_datetime(start_date)
    sub = c[c.index >= start_date]
    if sub.empty or len(sub) < 2:
        return float("nan")
    return float(sub.iloc[-1] / sub.iloc[0] - 1.0)

def _n_month_return(px: pd.DataFrame, months: int) -> float:
    c = _close_series(px)
    if c.empty:
        return float("nan")
    last_date = c.index[-1]
    start_date = last_date - DateOffset(months=months)
    return _period_return_from_date(px, start_date)

def _n_year_return(px: pd.DataFrame, years: int) -> float:
    c = _close_series(px)
    if c.empty:
        return float("nan")
    last_date = c.index[-1]
    start_date = last_date - DateOffset(years=years)
    return _period_return_from_date(px, start_date)

def _ytd_return(px: pd.DataFrame) -> float:
    """
    YTD = percent change from the last trading day's close of the previous year
    to the latest available close (Yahoo-style).
    Fallback: if no prior-year data, use first trading day of current year.
    """
    c = _close_series(px)
    if c.empty:
        return float("nan")

    last_date = c.index[-1]
    start_of_year = pd.Timestamp(year=last_date.year, month=1, day=1)

    # Anchor = last close strictly BEFORE Jan 1 (previous year's final trading day)
    prev_year_slice = c[c.index < start_of_year]
    if not prev_year_slice.empty:
        anchor = float(prev_year_slice.iloc[-1])
        return float(c.iloc[-1] / anchor - 1.0)

    # Fallback: no data before Jan 1 → use first trading day on/after Jan 1
    curr_year_slice = c[c.index >= start_of_year]
    if curr_year_slice.empty or len(curr_year_slice) < 2:
        return float("nan")
    return float(curr_year_slice.iloc[-1] / curr_year_slice.iloc[0] - 1.0)


# -------- main table -------------------------------------------------------

def build_compare_table(tickers, profiles: Dict[str, dict], prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        pr = profiles.get(t, {}) or {}
        px = prices.get(t)

        # Latest price & as-of
        c = _close_series(px)
        last_price = float(c.iloc[-1]) if not c.empty else None
        as_of = c.index[-1] if not c.empty else None

        # Calendar lookbacks (Yahoo-style). These return decimals (e.g., 0.123 = +12.3%)
        r_1m  = _n_month_return(px, 1)
        r_3m  = _n_month_return(px, 3)
        r_1y  = _n_year_return(px, 1)
        r_5y  = _n_year_return(px, 5)
        r_ytd = _ytd_return(px)

        # Risk stats (price-only; unaffected)
        vol_3m = d.realized_vol(px, 63)
        mdd    = d.max_drawdown(px)

        # Fundamentals (best-effort from profile)
        er  = pr.get("expense_ratio")
        aum = pr.get("aum")
        raw = pr.get("raw") or {}
        div_yield = raw.get("trailingAnnualDividendYield") or raw.get("yield")

        rows.append({
            "ticker": t,
            "name": pr.get("longName") or pr.get("shortName"),
            "type": pr.get("instrument_type"),
            "price": last_price,
            "as_of": as_of,
            "return_ytd": r_ytd,
            "return_1m": r_1m,
            "return_3m": r_3m,
            "return_1y": r_1y,
            "return_5y": r_5y,
            "vol_3m_ann": vol_3m,
            "max_drawdown": mdd,
            "expense_ratio": er,
            "dividend_yield": div_yield,
            "aum": aum,
        })
    return pd.DataFrame(rows)
