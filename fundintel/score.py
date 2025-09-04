# fundintel/score.py
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from . import data as d


# ----------------------------- small helpers -----------------------------

def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))

def _safe(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _parse_risk(risk: str) -> str:
    s = (risk or "").strip().lower()
    if s.startswith("low"): return "low"
    if s.startswith("high"): return "high"
    return "medium"

def _parse_horizon(h: str) -> str:
    s = (h or "").strip().lower()
    if "short" in s or "0-6" in s: return "short"
    if "long" in s or "3y" in s:   return "long"
    return "medium"

@lru_cache(maxsize=1)
def _bench_prices(period: str = "5y") -> pd.DataFrame:
    try:
        px = d.fetch_prices("^GSPC", period=period)
        if isinstance(px, pd.DataFrame) and not px.empty:
            return px
    except Exception:
        pass
    try:
        px = d.fetch_prices("SPY", period=period)
        if isinstance(px, pd.DataFrame) and not px.empty:
            return px
    except Exception:
        pass
    return pd.DataFrame()

def _ret(prices: pd.DataFrame, days: int) -> float:
    try:
        r = d.compute_return(prices, days=days)
        return float(r) if r is not None else np.nan
    except Exception:
        return np.nan

def _rel(r: float, b: float) -> float:
    if _nan(r) or _nan(b): return np.nan
    return r - b

def _last_close(prices: pd.DataFrame) -> float:
    if isinstance(prices, pd.DataFrame) and "Close" in prices and not prices["Close"].dropna().empty:
        return float(prices["Close"].dropna().iloc[-1])
    return np.nan

def _logistic01(x: float, mid: float, width: float, invert: bool = False) -> float:
    """
    Smooth map to [0,1], centered at 'mid', spread by 'width'.
    invert=True flips so smaller is better.
    """
    if _nan(x) or width <= 0:
        return 0.5
    z = (x - mid) / (width / 2.0)
    y = 1.0 / (1.0 + np.exp(-z))  # (0,1)
    return 1.0 - y if invert else y

def _downside_vol(prices: pd.DataFrame, window: int = 63) -> float:
    if not isinstance(prices, pd.DataFrame) or "Close" not in prices:
        return np.nan
    s = prices["Close"].dropna()
    if len(s) < 10:
        return np.nan
    r = s.pct_change().dropna()
    dn = r[r < 0]
    if dn.empty:
        return 0.0
    return float(dn.std() * np.sqrt(252))

def _slope_r2(prices: pd.DataFrame, window: int = 63) -> Tuple[float, float]:
    if not isinstance(prices, pd.DataFrame) or "Close" not in prices:
        return np.nan, np.nan
    s = prices["Close"].dropna()
    if len(s) < window + 5:
        return np.nan, np.nan
    y = np.log(s.iloc[-window:].values)
    x = np.arange(len(y), dtype=float)
    try:
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        y_hat = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return float(slope), float(max(0.0, min(1.0, r2)))
    except Exception:
        return np.nan, np.nan


# ------------------------------ main API --------------------------------

def compute_score(
    ticker: str,
    profile: Dict[str, Any],
    prices: pd.DataFrame,
    news_items: List[Dict[str, Any]],
    risk_tolerance: str = "Medium",
    horizon: str = "Medium (6-36m)",
) -> Dict[str, Any]:
    """
    Risk-aware scoring with:
      • Absolute & relative performance across horizons + trend
      • Risk & resilience (vol / downside vol / max DD)
      • Quality (cost / liquidity / AUM) + structure penalty (dials way down in High risk)
      • Consistency vs S&P + upside kicker in High risk
    """
    rt = _parse_risk(risk_tolerance)
    hz = _parse_horizon(horizon)

    # ---------- Pillar weights by risk preference ----------
    if rt == "low":
        W = {"perf": 0.20, "risk": 0.45, "qlt": 0.30, "cons": 0.05}
    elif rt == "high":
        W = {"perf": 0.70, "risk": 0.10, "qlt": 0.15, "cons": 0.05}  # perf dominates in High risk
    else:
        W = {"perf": 0.45, "risk": 0.30, "qlt": 0.20, "cons": 0.05}

    # ---------- Returns & benchmark ----------
    bench = _bench_prices("5y")
    R = {
        "1m":  _ret(prices, 21),
        "3m":  _ret(prices, 63),
        "6m":  _ret(prices, 126),
        "1y":  _ret(prices, 252),
        "5y":  _ret(prices, 252 * 5),
    }
    B = {
        "1m":  _ret(bench, 21),
        "3m":  _ret(bench, 63),
        "6m":  _ret(bench, 126),
        "1y":  _ret(bench, 252),
        "5y":  _ret(bench, 252 * 5),
    }
    EX = {k: _rel(R[k], B[k]) for k in R.keys()}

    # ---------- Performance & momentum (horizon-aware) ----------
    if hz == "short":
        w_abs = {"1m": 0.45, "3m": 0.30, "6m": 0.18, "1y": 0.05, "5y": 0.02}
        w_rel = {"1m": 0.50, "3m": 0.30, "6m": 0.15, "1y": 0.04, "5y": 0.01}
        trend_w = 0.20 if rt == "high" else 0.12
    elif hz == "long":
        w_abs = {"1m": 0.08, "3m": 0.12, "6m": 0.20, "1y": 0.25, "5y": 0.35}
        w_rel = {"1m": 0.10, "3m": 0.18, "6m": 0.25, "1y": 0.27, "5y": 0.20}
        trend_w = 0.05
    else:
        w_abs = {"1m": 0.30, "3m": 0.25, "6m": 0.20, "1y": 0.15, "5y": 0.10}
        w_rel = {"1m": 0.35, "3m": 0.25, "6m": 0.20, "1y": 0.15, "5y": 0.05}
        trend_w = 0.10

    A_abs = (
        w_abs["1m"] * _logistic01(_safe(R["1m"]), 0.00, 0.12) +
        w_abs["3m"] * _logistic01(_safe(R["3m"]), 0.00, 0.22) +
        w_abs["6m"] * _logistic01(_safe(R["6m"]), 0.00, 0.32) +
        w_abs["1y"] * _logistic01(_safe(R["1y"]), 0.00, 0.45) +
        w_abs["5y"] * _logistic01(_safe(R["5y"]), 0.00, 1.20)
    )
    A_rel = (
        w_rel["1m"] * _logistic01(_safe(EX["1m"]), 0.00, 0.05) +
        w_rel["3m"] * _logistic01(_safe(EX["3m"]), 0.00, 0.10) +
        w_rel["6m"] * _logistic01(_safe(EX["6m"]), 0.00, 0.15) +
        w_rel["1y"] * _logistic01(_safe(EX["1y"]), 0.00, 0.22) +
        w_rel["5y"] * _logistic01(_safe(EX["5y"]), 0.00, 0.40)
    )
    slope, r2 = _slope_r2(prices, 63)
    slope_ann = None if _nan(slope) else float(slope) * 252.0
    T = 0.0 if _nan(slope_ann) or _nan(r2) else max(0.0, slope_ann) * float(r2)
    T_n = _logistic01(T, 0.00, 0.25)

    S_perf = _clip01(0.45 * A_abs + 0.45 * A_rel + trend_w * T_n)

    # ---------- Risk & resilience ----------
    vol = d.realized_vol(prices, 63)      # annualized
    dvol = _downside_vol(prices, 63)
    dd = d.max_drawdown(prices)

    # In High risk, we soften the strictness a bit (centers a touch higher)
    if rt == "high":
        S_vol = _logistic01(_safe(vol), 0.22, 0.11, invert=True)
        S_dvol = _logistic01(_safe(dvol), 0.18, 0.10, invert=True)
        S_dd  = _logistic01(abs(_safe(dd)), 0.35, 0.18, invert=True)
    else:
        S_vol = _logistic01(_safe(vol), 0.18, 0.09, invert=True)
        S_dvol = _logistic01(_safe(dvol), 0.14, 0.08, invert=True)
        S_dd  = _logistic01(abs(_safe(dd)), 0.25, 0.15, invert=True)

    S_risk = _clip01(0.40 * S_vol + 0.35 * S_dvol + 0.25 * S_dd)

    # ---------- Quality (cost/liquidity/AUM) + structure ----------
    er = profile.get("expense_ratio")
    avg_vol_sh = (profile.get("raw") or {}).get("averageVolume")
    last_px = _last_close(prices)
    dollar_vol = None if _nan(avg_vol_sh) or _nan(last_px) else float(avg_vol_sh) * float(last_px)
    aum = profile.get("aum")

    # cost/liquidity/aum
    S_cost = _logistic01(_safe(er), 0.0020, 0.0015, invert=True)   # 0.20% mid
    S_liq  = _logistic01(_safe(dollar_vol), 30e6, 12e6)            # $30M/day mid
    S_aum  = _logistic01(_safe(aum), 1.0e10, 6.0e9)                # $10B mid

    # structure penalty (drastically reduced in High risk)
    itype = (profile.get("instrument_type") or "").lower()
    name  = (profile.get("longName") or profile.get("shortName") or "").lower()
    base_pen = 0.0
    if "etn" in itype:
        base_pen += 0.30
    if "leveraged" in name or " 2x" in name or " 3x" in name or "2x" in name or "3x" in name:
        base_pen += 0.35
    struct_pen = base_pen * (0.2 if rt == "high" else 1.0)   # 80% reduction in High risk

    S_qlt = _clip01(0.45 * S_cost + 0.35 * S_liq + 0.20 * S_aum - struct_pen)

    # ---------- Consistency vs S&P + upside kicker ----------
    horizons = ["1m", "3m", "6m", "1y", "5y"]
    beats = [(_safe(EX[h]) > 0.0) for h in horizons if not _nan(EX[h])]
    win_rate = (sum(beats) / len(beats)) if beats else 0.0
    avg_excess = np.mean([_safe(EX[h]) for h in horizons if not _nan(EX[h])]) if beats else 0.0
    S_cons = _clip01(0.65 * win_rate + 0.35 * _logistic01(avg_excess, 0.0, 0.08))

    # Upside kicker (helps explosive names when they’re actually beating)
    if beats:
        all_win = all(beats)
        if rt == "high":
            if all_win:
                S_cons = _clip01(S_cons + 0.12)  # big boost if beating across all windows
            elif win_rate >= 0.8:
                S_cons = _clip01(S_cons + 0.08)  # 4 of 5 windows
        elif rt == "medium":
            if all_win:
                S_cons = _clip01(S_cons + 0.06)
            elif win_rate >= 0.8:
                S_cons = _clip01(S_cons + 0.03)
        else:  # low
            if all_win:
                S_cons = _clip01(S_cons + 0.03)

    # ---------- Combine pillars ----------
    total01 = (
        W["perf"] * S_perf +
        W["risk"] * S_risk +
        W["qlt"]  * S_qlt  +
        W["cons"] * S_cons
    )

    # Broad-market anchor bonus: SPY/VOO/IVV/SPLG/VTI/^GSPC & “s&p 500”/“total market”
    broad_tickers = {"SPY","VOO","IVV","SPLG","VTI","^GSPC"}
    broad_terms = ("s&p 500", "sp 500", "total market")
    is_broad = (ticker.upper() in broad_tickers) or any(t in name for t in broad_terms)
    if is_broad:
        if rt == "low":
            total01 += 0.10   # strong anchor for set-and-forget
        elif rt == "medium":
            total01 += 0.05
        else:
            total01 += 0.02
    total01 = _clip01(total01)

    # Expand spread so leaders don’t cluster in the 60s/70s
    if rt == "low":
        total01 = _clip01(0.12 + 1.30 * (total01 - 0.12))
    elif rt == "medium":
        total01 = _clip01(0.10 + 1.18 * (total01 - 0.10))
    else:
        total01 = _clip01(0.08 + 1.15 * (total01 - 0.08))

    total = round(total01 * 100.0, 1)

    # Soft cap for exotic *except* High risk (we removed it here on purpose)
    if rt != "high" and base_pen >= 0.30:
        total = min(total, 90.0)

    return {
        "ticker": ticker,
        "score_0_100": total,

        # Expose pillar subscores (0–100) for explainability if you want to display later
        "perf_score": round(S_perf * 100, 1),
        "risk_score": round(S_risk * 100, 1),
        "quality_score": round(S_qlt * 100, 1),
        "consistency_score": round(S_cons * 100, 1),

        # Fields your tables already expect
        "expense_ratio": profile.get("expense_ratio"),
        "avg_volume": (profile.get("raw") or {}).get("averageVolume"),
        "vol_annualized": d.realized_vol(prices, 63),
        "max_drawdown": d.max_drawdown(prices),

        # little extra if you want badges
        "beats_sp500_all_periods": bool(beats and all(beats)),
    }
