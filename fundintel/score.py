from typing import Dict, Any, List
import numpy as np
import pandas as pd
from . import data as d

# Base weights (same idea as before; "Medium" baseline)
BASE_WEIGHTS = {
    "cost": 0.20,
    "liquidity": 0.15,
    "volatility": 0.15,
    "drawdown": 0.10,
    "structure": 0.15,
    "momentum": 0.10,
    "tracking": 0.10,
    "news": 0.05,
}

# How Risk changes the *importance* (weights) of each pillar
# (bigger multipliers here = more impact from the slider)
RISK_WEIGHT_MULTIPLIERS = {
    "low": {
        "cost": 1.30,
        "liquidity": 0.80,
        "volatility": 2.00,
        "drawdown": 2.00,
        "structure": 1.20,
        "momentum": 0.40,
        "tracking": 1.00,
        "news": 1.00,
    },
    "medium": {k: 1.00 for k in BASE_WEIGHTS.keys()},
    "high": {
        "cost": 0.60,
        "liquidity": 1.60,
        "volatility": 0.40,
        "drawdown": 0.40,
        "structure": 0.80,
        "momentum": 2.50,
        "tracking": 1.00,
        "news": 0.80,
    },
}


def _normalize(value, low, high, invert=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.5
    x = (value - low) / (high - low) if high > low else 0.5
    x = min(max(x, 0.0), 1.0)
    return 1.0 - x if invert else x

def _risk_weighted(base: Dict[str, float], risk_tolerance: str) -> Dict[str, float]:
    """Apply risk multipliers to weights and renormalize to sum=1."""
    mult = RISK_WEIGHT_MULTIPLIERS.get((risk_tolerance or "medium").lower(), RISK_WEIGHT_MULTIPLIERS["medium"])
    w = {k: base.get(k, 0.0) * mult.get(k, 1.0) for k in base.keys()}
    s = sum(w.values())
    if s > 0:
        w = {k: v / s for k, v in w.items()}
    return w

def compute_score(
    ticker: str,
    profile: Dict[str, Any],
    prices: pd.DataFrame,
    news_items: List[Dict[str, Any]],
    risk_tolerance: str = "Medium",
    horizon: str = "Medium (6-36m)",  # accepted but unused here (kept for compatibility)
) -> Dict[str, Any]:
    # Cost
    er = profile.get("expense_ratio")
    cost_score = _normalize(er if isinstance(er, (int, float)) else float("nan"), 0.0, 0.02, invert=True)

    # Liquidity (avg daily volume proxy)
    avg_vol = (profile.get("raw") or {}).get("averageVolume")
    liq_score = _normalize(avg_vol if isinstance(avg_vol, (int, float)) else float("nan"), 1e4, 5e6, invert=False)

    # Volatility & Drawdown
    vol = d.realized_vol(prices, 63)
    vol_score = _normalize(vol, 0.05, 0.60, invert=True)

    dd = d.max_drawdown(prices)
    dd_score = _normalize(abs(dd) if isinstance(dd, (int, float)) else float("nan"), 0.05, 0.95, invert=True)

    # Structure (penalize ETNs/leveraged)
    itype = (profile.get("instrument_type") or "").lower()
    name = (profile.get("longName") or profile.get("shortName") or "").lower()
    structure_penalty = 0.0
    if "etn" in itype:
        structure_penalty += 0.25
    if "leveraged" in name or "3x" in name or "2x" in name:
        structure_penalty += 0.35
    structure_score = max(0.0, 1.0 - structure_penalty)

    # Momentum (fixed windows: ~3m vs ~12m)
    m3 = d.compute_return(prices, days=63)
    m12 = d.compute_return(prices, days=252)
    m3 = 0.0 if (m3 is None or (isinstance(m3, float) and np.isnan(m3))) else m3
    m12 = 0.0 if (m12 is None or (isinstance(m12, float) and np.isnan(m12))) else m12
    mom_raw = m3 - 0.3 * m12
    momentum_score = _normalize(mom_raw, -0.5, 0.5, invert=False)

    # Tracking (placeholder)
    tracking_score = 0.5

    # News (keyword heuristic)
    neg_words = ("downgrade","lawsuit","default","class action","liquidate","delist","halt","suspend","warning","cut","loss","risk")
    pos_words = ("upgrade","increase","record","beat","approval","launch","gains","profit","flows")
    titles = " ".join([(n.get("title") or "").lower() for n in (news_items or [])])
    neg_hits = sum(w in titles for w in neg_words)
    pos_hits = sum(w in titles for w in pos_words)
    news_score = 0.5 if (neg_hits + pos_hits) == 0 else max(0.0, min(1.0, 0.5 + 0.1 * (pos_hits - neg_hits)))

    # --- MAKE RISK MORE IMPACTFUL ------------------------------------------
    rt = (risk_tolerance or "medium").lower()

    # 1) Scale pillar scores themselves (stronger nudges than before)
    if rt == "low":
        vol_score *= 1.6
        dd_score  *= 1.6
        momentum_score *= 0.6
        structure_score = max(0.0, min(1.0, structure_score * 0.95))  # slightly tougher on exotic structures
    elif rt == "high":
        vol_score *= 0.6
        dd_score  *= 0.6
        momentum_score *= 1.6
        structure_score = max(0.0, min(1.0, structure_score + 0.10))  # soften structure penalty by +10 pts

    # Clip pillar scores to [0,1] after scaling
    scores = {
        "cost": cost_score,
        "liquidity": liq_score,
        "volatility": vol_score,
        "drawdown": dd_score,
        "structure": structure_score,
        "momentum": momentum_score,
        "tracking": tracking_score,
        "news": news_score,
    }
    for k in scores:
        scores[k] = max(0.0, min(1.0, scores[k]))

    # 2) Reweight pillars based on risk preference (biggest impact)
    weights = _risk_weighted(BASE_WEIGHTS, risk_tolerance)

    # Weighted sum → total score
    total = 0.0
    for k, w in weights.items():
        total += scores[k] * w
    total = max(0.0, min(1.0, total))

    return {
        "ticker": ticker,
        "score_0_100": round(total * 100, 1),
        **{f"{k}_score": round(v * 100, 1) for k, v in scores.items()},
        "expense_ratio": er,
        "avg_volume": avg_vol,
        "vol_annualized": vol,
        "max_drawdown": dd,
    }
