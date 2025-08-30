# fundintel/holdings.py
from __future__ import annotations

import io
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
import yfinance as yf

# ------- shared schema -------
STD_COLS = [
    "symbol", "name", "weight", "shares", "market_value",
    "sector", "country", "source", "as_of"
]

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
    "Connection": "keep-alive",
}

# ===========================
# Low-level helpers
# ===========================

def _http_get(url: str, **kw) -> requests.Response:
    kw.setdefault("headers", UA)
    kw.setdefault("timeout", 20)
    r = requests.get(url, **kw)
    r.raise_for_status()
    return r

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in STD_COLS:
        if c not in out.columns:
            out[c] = None
    return out[STD_COLS]

def _pick_weight_col(df: pd.DataFrame) -> Optional[str]:
    # Find a numeric column that looks like percent (0–100 or 0–1)
    cand_cols = [c for c in df.columns if re.search(r"(weight|%|allocation|net assets|percent)", str(c), re.I)]
    for col in cand_cols + list(df.select_dtypes(include="number").columns):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < max(3, len(s) // 5):
            continue
        m = s.dropna().abs().median()
        if 0 < m <= 100:
            return col
    return None

def _standardize_rows(rows: List[Dict], source: str, as_of: Optional[pd.Timestamp]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=STD_COLS)

    df = pd.DataFrame(rows)
    # Normalize frequent names
    ren = {
        "Ticker": "symbol", "ticker": "symbol", "Symbol": "symbol",
        "Name": "name", "Holding": "name", "Holding Name": "name",
        "Company Name": "name", "Security": "name",
        "Weight": "weight", "% of Net Assets": "weight", "Percent": "weight",
        "Weight (%)": "weight", "Allocation": "weight",
        "Shares": "shares", "Share": "shares", "Quantity": "shares", "Qty": "shares",
        "Market Value": "market_value", "MarketValue": "market_value", "Position": "market_value",
        "ISIN": "isin", "CUSIP": "cusip",
        "Sector": "sector", "Country": "country"
    }
    for k, v in list(ren.items()):
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # If weight not yet mapped, try auto-pick
    if "weight" not in df.columns:
        wcol = _pick_weight_col(df)
        if wcol:
            df.rename(columns={wcol: "weight"}, inplace=True)

    # Clean numerics / percent
    if "weight" in df.columns:
        def _clean_w(x):
            if x is None or (isinstance(x, float) and pd.isna(x)): return None
            if isinstance(x, str):
                x = x.replace("%", "").strip()
            try:
                val = float(x)
                return val/100.0 if val > 1.0 else val
            except Exception:
                return None
        df["weight"] = df["weight"].map(_clean_w)

    for col in ("shares", "market_value"):
        if col in df.columns:
            def _num(x):
                if x is None or (isinstance(x, float) and pd.isna(x)): return None
                if isinstance(x, str):
                    y = x.replace("$", "").replace(",", "").strip()
                else:
                    y = x
                try:
                    return float(y)
                except Exception:
                    return None
            df[col] = df[col].map(_num)

    # Keep only informative rows
    keep_cols = ["name", "symbol", "weight", "shares", "market_value"]
    df = df.dropna(how="all", subset=[c for c in keep_cols if c in df.columns])

    # Order & cap
    if "weight" in df.columns and not df["weight"].isna().all():
        df = df.sort_values("weight", ascending=False)
    df = df.head(25)

    # Final shape
    df["source"] = source
    df["as_of"] = as_of
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().replace({"nan": None, "None": None})
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip().replace({"nan": None, "None": None})

    return _ensure_cols(df.reset_index(drop=True))

# ===========================
# Issuer adapters
# ===========================

# ---- Global X (e.g., ARGT) ----
def _globalx_fetch(ticker: str, website: Optional[str]) -> pd.DataFrame:
    # Prefer website if present, else guess standard fund path
    url = website or f"https://www.globalxetfs.com/funds/{ticker.lower()}/"
    try:
        html = _http_get(url).text
    except Exception:
        return pd.DataFrame(columns=STD_COLS)

    # Find a "Full Holdings (.csv)" link
    m = re.search(r'href="([^"]+\.csv[^"]*)"', html, re.I)
    if not m:
        return pd.DataFrame(columns=STD_COLS)

    csv_url = urljoin(url, m.group(1))
    try:
        r = _http_get(csv_url)
        content = r.content
        # Try CSV parse with best-effort encoding/sep
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content), engine="python", sep=None)
    except Exception:
        return pd.DataFrame(columns=STD_COLS)

    # Try to extract "As of ..." from page
    as_of = None
    m2 = re.search(r"As of\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})", html)
    if m2:
        try:
            as_of = pd.to_datetime(m2.group(1))
        except Exception:
            as_of = None

    return _standardize_rows(df.to_dict("records"), source="globalx-csv", as_of=as_of)

# ---- iShares / BlackRock (e.g., IVV) ----
def _ishares_fetch(website: Optional[str]) -> pd.DataFrame:
    if not website:
        return pd.DataFrame(columns=STD_COLS)
    try:
        html = _http_get(website).text
    except Exception:
        return pd.DataFrame(columns=STD_COLS)

    # Look for any link that requests a CSV holdings download
    # common patterns: fileType=csv & (holdings|constituents)
    matches = re.findall(r'href="([^"]+fileType=csv[^"]+)"', html, re.I)
    csv_url = None
    for href in matches:
        if re.search(r"holding|constituent", href, re.I):
            csv_url = urljoin(website, href)
            break
    if not csv_url and matches:
        csv_url = urljoin(website, matches[0])

    if not csv_url:
        return pd.DataFrame(columns=STD_COLS)

    try:
        rr = _http_get(csv_url)
        content = rr.content
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content), engine="python", sep=None)
    except Exception:
        return pd.DataFrame(columns=STD_COLS)

    return _standardize_rows(df.to_dict("records"), source="ishares-csv", as_of=None)

# ---- SPDR / State Street (e.g., SPY) ----
def _spdr_fetch(ticker: str, website: Optional[str]) -> pd.DataFrame:
    # If we have the product page, scan for an xlsx "holdings" link; else try a common pattern
    try_urls = []
    if website:
        try_urls.append(website)
    if ticker:
        # Common daily holdings pattern (works for many SPDR US ETFs)
        try_urls.append("https://www.ssga.com/us/en/intermediary/etfs")
    # Scan pages for an xlsx link containing 'holdings' and the ticker slug
    xlsx_link = None
    for base in try_urls:
        try:
            html = _http_get(base).text
        except Exception:
            continue
        m = re.search(r'href="([^"]+holdings[^"]+\.xlsx)"', html, re.I)
        if m:
            xlsx_link = urljoin(base, m.group(1))
            break
    if not xlsx_link:
        # Last-ditch guess using known library path (often: .../holdings-daily-us-en-<ticker>.xlsx)
        guess = f"https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
        try:
            _http_get(guess)
            xlsx_link = guess
        except Exception:
            return pd.DataFrame(columns=STD_COLS)

    try:
        r = _http_get(xlsx_link)
        df = pd.read_excel(io.BytesIO(r.content))
    except Exception:
        return pd.DataFrame(columns=STD_COLS)

    return _standardize_rows(df.to_dict("records"), source="spdr-xlsx", as_of=None)

# ===========================
# Enrichment (sector/country)
# ===========================

@lru_cache(maxsize=4096)
def _lookup_meta(symbol: str) -> Dict[str, Optional[str]]:
    if not symbol or str(symbol).lower() in ("nan", "none"):
        return {"sector": None, "country": None}
    try:
        info = yf.Ticker(str(symbol)).info or {}
        sector  = info.get("sector")
        country = info.get("country") or info.get("countryOfIncorporation") or info.get("countryOfOrigin")
        time.sleep(0.05)
        return {"sector": sector, "country": country}
    except Exception:
        return {"sector": None, "country": None}

def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    syms = df["symbol"].fillna("").astype(str).tolist()
    meta = pd.DataFrame([_lookup_meta(s) for s in syms])
    out = df.copy()
    out["sector"]  = out.get("sector").fillna(meta.get("sector"))
    out["country"] = out.get("country").fillna(meta.get("country"))
    return out

# ===========================
# Public API
# ===========================

def fetch_holdings(ticker: str, profile: dict) -> pd.DataFrame:
    """
    Fetch Top 10–25 holdings for a fund using issuer endpoints when possible.
    - Global X → CSV (page 'Full Holdings (.csv)')
    - iShares/BlackRock → CSV link on product page
    - SPDR/SSGA → daily XLSX holdings
    Fallbacks: none (Yahoo blocked by 401 for many users).
    """
    itype = (profile.get("instrument_type") or "").lower()
    name  = (profile.get("longName") or profile.get("shortName") or "").lower()
    website = (profile.get("website") or "").strip()
    fam    = (profile.get("fund_family") or profile.get("fundFamily") or "").lower()

    # ETNs don't publish equity constituents like ETFs; return empty by design
    if "etn" in itype or "note" in name:
        return pd.DataFrame(columns=STD_COLS)

    # Route by website domain / family / name
    domain = ""
    try:
        domain = re.search(r"https?://([^/]+)/", website).group(1).lower() if website else ""
    except Exception:
        domain = ""

    df = pd.DataFrame(columns=STD_COLS)

    # Global X
    if "globalx" in domain or "globalxetfs.com" in website or "global x" in name or "global x" in fam:
        df = _globalx_fetch(ticker, website)

    # iShares / BlackRock
    elif any(s in domain for s in ("ishares.com", "blackrock.com")) or "ishares" in name or "ishares" in fam:
        df = _ishares_fetch(website)

    # SPDR / State Street
    elif any(s in domain for s in ("ssga.com", "spdrs.com", "sectorspdrs.com")) or "spdr" in name or "state street" in fam:
        df = _spdr_fetch(ticker, website)

    # Enrich + finalize
    if not df.empty:
        df = _enrich(df)
        df = _ensure_cols(df.reset_index(drop=True))
    return df

def sector_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "weight" not in df:
        return pd.DataFrame({"sector": [], "weight": []})
    tmp = df.copy()
    tmp["sector"] = tmp["sector"].fillna("Other")
    out = tmp.groupby("sector", dropna=False)["weight"].sum().sort_values(ascending=False).reset_index()
    out["weight"] = out["weight"].fillna(0.0)
    return out

def country_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "weight" not in df:
        return pd.DataFrame({"country": [], "weight": []})
    tmp = df.copy()
    tmp["country"] = tmp["country"].fillna("Other")
    out = tmp.groupby("country", dropna=False)["weight"].sum().sort_values(ascending=False).reset_index()
    out["weight"] = out["weight"].fillna(0.0)
    return out
