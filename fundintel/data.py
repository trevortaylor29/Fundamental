import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import yfinance as yf

def fetch_profile(ticker: str) -> Dict[str, Any]:
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    longName = info.get("longName") or info.get("shortName")
    quoteType = info.get("quoteType")
    expense_ratio = info.get("annualReportExpenseRatio") or info.get("expenseRatio")
    aum = info.get("totalAssets") or info.get("totalAssetsUSD")
    summary = info.get("longBusinessSummary") or ""
    instrument_type = _infer_type(quoteType, info)
    return {
        "ticker": ticker,
        "longName": longName,
        "shortName": info.get("shortName"),
        "instrument_type": instrument_type,
        "expense_ratio": expense_ratio,
        "aum": aum,
        "summary": summary,
        "raw": info,
    }

def _infer_type(quoteType: Optional[str], info: Dict[str, Any]) -> str:
    qt = (quoteType or "").upper()
    if "etn" in (info.get("category","") or "").lower():
        return "ETN"
    if qt == "ETF":
        return "ETF"
    if qt == "MUTUALFUND":
        return "Mutual Fund"
    if qt == "EQUITY":
        return "Stock"
    if qt:
        return qt.title()
    return "Unknown"

def fetch_prices(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    import pandas as pd
    import yfinance as yf

    tk = yf.Ticker(ticker)
    try:
        px = tk.history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        px = pd.DataFrame()

    if not px.empty:
        # ensure datetime index
        px.index = pd.to_datetime(px.index, errors="coerce")
        # drop timezone to avoid tz-aware vs tz-naive comparisons later
        try:
            if getattr(px.index, "tz", None) is not None:
                # convert to UTC then strip tz info (now tz-naive)
                px.index = px.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            # some indexes may only need tz_localize(None)
            try:
                px.index = px.index.tz_localize(None)
            except Exception:
                pass

    return px


def align_prices(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    if not series_map:
        return pd.DataFrame()
    df = pd.concat(series_map, axis=1)
    df.index = pd.to_datetime(df.index)
    return df

def compute_return(px: pd.DataFrame, days: int = 1) -> float:
    if px is None or px.empty or "Close" not in px:
        return float("nan")
    c = px["Close"].dropna()
    if len(c) <= days:
        return float("nan")
    return (c.iloc[-1] / c.iloc[-1 - days]) - 1.0

def realized_vol(px: pd.DataFrame, lookback: int = 63) -> float:
    if px is None or px.empty or "Close" not in px:
        return float("nan")
    r = px["Close"].pct_change().dropna()
    if len(r) < 2:
        return float("nan")
    lb = r.tail(lookback)
    return float((252 ** 0.5) * lb.std())

def max_drawdown(px: pd.DataFrame) -> float:
    if px is None or px.empty or "Close" not in px:
        return float("nan")
    c = px["Close"].dropna()
    if c.empty:
        return float("nan")
    roll_max = c.cummax()
    dd = c / roll_max - 1.0
    return float(dd.min())
