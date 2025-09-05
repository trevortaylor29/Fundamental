import streamlit as st
import re
import pandas as pd
import numpy as np
from fundintel import data, score, news, compare

st.set_page_config(page_title="Fundamental", layout="wide")

st.title(" Fundamental - The fundamentals of every fund.")
st.caption("Not Investment Advice. Do your own research.")

with st.sidebar:
    st.header("Tickers")
    tickers_input = st.text_input("Enter 1–5 symbols (comma-separated)", value="USD,PLTR")
    st.header("⚙️ Settings")
    risk = st.select_slider("Risk tolerance", options=["Low","Medium","High"], value="Medium")
    # horizon = st.select_slider("Time horizon", options=["Short (0-6m)","Medium (6-36m)","Long (3y+)"], value="Medium (6-36m)")
    
    horizon = "Medium (6-36m)"
    show_news = st.checkbox("Fetch recent news", value=True)
    st.markdown("---")
    st.markdown("**Tip:** Try ETFs (e.g., ARGT) or ETNs (e.g., FNGU).")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:5]

if not tickers:
    st.stop()

tab_overview, tab_perf, tab_news, tab_holdings, tab_risk, tab_compare, tab_discover = st.tabs(
    ["Overview", "Performance", "News & Filings", "Holdings", "Risk & Score", "Compare", "Discover Top Funds"]
)


# Cache fetches
profiles = {}
prices = {}
news_items = {}

with st.spinner("Fetching data..."):
    for t in tickers:
        profiles[t] = data.fetch_profile(t)
        prices[t] = data.fetch_prices(t, period="5y")
        news_items[t] = news.fetch_news(t) if show_news else []

# -------- Overview --------
with tab_overview:
    cols = st.columns(min(3, len(tickers)))
    for i, t in enumerate(tickers):
        pr = profiles[t]
        px = prices[t]
        last = px["Close"].dropna().iloc[-1] if not px.empty else np.nan
        chg_1d = data.compute_return(px, days=1)
        aum = pr.get("aum")
        er = pr.get("expense_ratio")
        with cols[i % len(cols)]:
            st.subheader(f"{t} — {pr.get('longName') or pr.get('shortName') or 'Unknown'}")
            st.metric("Last price", f"${last:,.2f}" if pd.notna(last) else "—",
                      delta=f"{chg_1d*100:.2f}%" if pd.notna(chg_1d) else None)
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Type:** {pr.get('instrument_type','—')}")
            c2.write(f"**Expense ratio:** {er:.2%}" if isinstance(er, (int,float)) else "**Expense ratio:** —")
            c3.write(f"**AUM:** ${aum:,.0f}" if isinstance(aum, (int,float)) else "**AUM:** —")
            st.caption(pr.get("summary",""))
    st.markdown("---")
    st.caption("""
    ### Disclosures & Data Notes
    - **Not investment advice.** This tool is for **educational and research** use only. Nothing here is a recommendation to buy, sell, or hold any security.
    - **Data sources.** Quotes, profiles, and history are fetched from **public endpoints** (e.g., Yahoo Finance via yfinance), plus public news feeds. Data can be **delayed, revised, or incomplete**. This project is not affiliated with or endorsed by Yahoo Finance. References to Yahoo are for descriptive purposes only.
    - **Accuracy & uptime.** We **do not warrant** accuracy, completeness, or uninterrupted availability. Validate with official sources before making decisions.
    - **Trademarks.** Tickers and brand marks belong to their respective owners and are used for identification only.
    - **Performance math.** Returns use **adjusted prices** (splits/dividends) to align with common retail portals; provider methodologies can differ.
    - **Liability.** The authors/operators are **not liable** for any losses or decisions made based on this site.
    """)

# -------- Performance --------
with tab_perf:
    import plotly.express as px
    st.subheader("Total Return since inception (or last 5 years)")

    st.info(
        "**How to read this chart**\n"
        "- The y-axis shows **how many times up** the fund is **since the first day in this window** (default: last 5 years).\n"
        "- At the start it’s **0.00×** (no gain). **1.00×** = **+100%**, **2.00×** = **+200%**, **8.33×** = **+833%**.\n"
        "- Each fund uses its **first available trading day within this window** as its start (unless you pick *Common start*).\n"
        "- Common start uses the first date where all selected tickers have data, so every line covers the same period to remove inception-date bias.\n"
        "- Hover any point to see **Up X.XX×** and the **actual $ price** on that date.\n"
        "- For today’s dollar price, check the **Overview** tab. For worst dips, see **Rolling Drawdown**."
    )

    # Pull Close prices for selected tickers and forward-fill gaps
    df = data.align_prices({t: prices[t]["Close"] for t in tickers if not prices[t].empty})
    if df.empty:
        st.info("No price data available.")
    else:
        df = df.ffill()

        # Alignment choice (kept here so you can copy-paste this block as-is)
        align_mode = st.radio(
            "Date alignment",
            ["Common start (strict)", "Each fund's own start (max coverage)"],
            index=1,
            help="Common start trims to the overlapping period so all tickers share the same start date.",
            horizontal=True,
        )
        st.caption(
            "Note: You can click and drag on the graph to zoom into that time frame "
        )

        if align_mode.startswith("Common"):
            # Use only the overlapping window where ALL tickers have data
            df_used = df.dropna()
            if df_used.empty:
                st.warning("No overlapping history across all selected tickers.")
                st.stop()
            start = df_used.iloc[0]  # one common start row (Series of start prices)
        else:
            df_used = df
            # Per-ticker start = first non-NaN value in the 5y window
            start = df_used.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)

        # Up multiple (gain multiple): (Price / Start) - 1  →  0.00× at start
        up_mult = df_used.divide(start, axis=1) - 1

        # Plot Up X.XX× line chart
        fig = px.line(
            up_mult,
            labels={"value": "Up multiple since start (0.00× at start)", "index": "Date", "variable": "Ticker"}
        )
        fig.update_yaxes(tickformat=".2f", ticksuffix="×")

        # Add actual $ price to hover and last price to legend
        for tr in fig.data:
            t = tr.name  # ticker
            series = df_used[t]
            tr.customdata = series.values  # actual $ price series for hover
            last = series.dropna()
            last_price = last.iloc[-1] if not last.empty else None
            tr.name = f"{t} (${last_price:,.2f})" if last_price is not None else t
            tr.hovertemplate = (
                "%{x|%Y-%m-%d}"
                "<br>Up %{y:.2f}×  ($%{customdata:.2f})"
                "<extra>%{fullData.name}</extra>"
            )

        st.plotly_chart(fig, width='stretch')

        # Rolling drawdown (as % from running peak)
        st.subheader("Rolling Drawdown")
        dd = df_used / df_used.cummax() - 1.0
        fig2 = px.line(dd, labels={"value": "Drawdown from peak", "index": "Date", "variable": "Ticker"})
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, width='stretch')

        st.caption(
            "Note: “Up X.XX×” compares price to the start of the selected window. "
            "Use **Common start** for strict apples-to-apples comparisons."
        )


    st.markdown("---")
    st.caption(
        "© Fundamental — Educational tool only. Not investment advice. Data from public web sources (incl. Yahoo Finance via yfinance) "
        "may be delayed or incomplete. No warranty of accuracy or fitness. Past performance is not indicative of future results."
    )

# -------- Holdings --------
with tab_holdings:
    st.subheader("Top Holdings (Under Construction)")
    st.warning(
        "Holdings data can't be displayed until I find a workaround. "
        "Unlike price, volume, and profile information (which Yahoo Finance exposes via their quote APIs), "
        "constituent-level holdings are **not returned in the standard Yahoo endpoints**. "
        "Yahoo migrated these holdings feeds behind gated, cookie-authenticated services. "
        "At the same time, most ETF issuers (e.g. Global X, iShares, Vanguard) now inject their daily holdings tables "
        "into the page dynamically using JavaScript. This means a simple HTTP request "
        "returns only the page shell — without the rendered table — so parsing yields an empty result. "
        "This project is not affiliated with or endorsed by Yahoo Finance. References to Yahoo are for descriptive purposes only."

    )

    st.markdown("---")
    st.caption("© Fundamental. For education only. Do your own research.")
# -------- Exposure --------
# with tabs[3]:
#     st.subheader("Sector & Country Exposure (Under Construction)")
#     st.info(
#         "Exposure charts are derived directly from fund holdings. "
#         "Because upstream holdings feeds are currently inaccessible (see Holdings tab), "
#         "exposure breakdowns by sector and country are also unavailable."
#     )

#     st.markdown("---")
#     st.caption("© Fundamental. For education only. Do your own research.")

# -------- News & Filings --------
with tab_news:
    for t in tickers:
        st.subheader(f"📰 {t}")
        items = news_items.get(t, [])
        if not items:
            st.write("No recent items found.")
            continue
        for it in items[:6]:
            date_str = it['published'].strftime("%Y-%m-%d") if it.get('published') else ""
            st.markdown(f"- [{it['title']}]({it['link']}) — {date_str}")

    st.markdown("---")
    st.caption(
        "© Fundamental — Educational tool only. Not investment advice. Data from public web sources (incl. Yahoo Finance via yfinance) "
        "may be delayed or incomplete. No warranty of accuracy or fitness. Past performance is not indicative of future results."
    )

# -------- Risk & Score --------
with tab_risk:
    import pandas as pd

    st.subheader("Transparent Score (0–100)")
    st.caption(
        "Our score blends fundamentals and market behavior: liquidity (volume), realized volatility & drawdown, "
        "structure (penalties for ETNs/leveraged), multi-window momentum, and a light news signal. "
        "Weights adapt to your Risk setting."
    )

    # --- compute scores for the active tickers ---
    rows = []
    for t in tickers:
        pr = profiles[t]
        px = prices[t]
        nw = news_items.get(t, [])
        try:
            s = score.compute_score(
                ticker=t,
                profile=pr,
                prices=px,
                news_items=nw,
                risk_tolerance=risk,
                horizon=horizon,
            )
            rows.append(s)
        except Exception:
            rows.append({"ticker": t, "score_0_100": float("nan")})

    raw_df = pd.DataFrame(rows).set_index("ticker")

    # Friendly headers (no Expense Ratio here)
    rename_map = {
        "score_0_100": "Score (0–100)",
        "cost_score": "Cost",
        "liquidity_score": "Liquidity",
        "volatility_score": "Volatility",
        "drawdown_score": "Drawdown",
        "structure_score": "Structure",
        "momentum_score": "Momentum",
        "tracking_score": "Tracking",
        "news_score": "News",
        "avg_volume": "Avg Volume",
        "vol_annualized": "Volatility (Ann.)",
        "max_drawdown": "Max Drawdown",
    }
    df = raw_df.rename(columns=rename_map)

    # Convenience profile fields
    def _name(t):
        p = profiles.get(t, {})
        return p.get("longName") or p.get("shortName") or ""

    def _type(t):
        p = profiles.get(t, {})
        return p.get("instrument_type") or ""

    def _aum(t):
        p = profiles.get(t, {})
        return p.get("aum")

    def _last_price(t):
        px = prices.get(t)
        try:
            return float(px["Close"].dropna().iloc[-1]) if px is not None and not px.empty else None
        except Exception:
            return None

    df.insert(0, "Name", [ _name(t) for t in df.index ])
    df.insert(1, "Type", [ _type(t) for t in df.index ])
    # Insert Last Price and AUM columns
    df.insert(2, "Last Price ($)", [ _last_price(t) for t in df.index ])
    df.insert(3, "AUM ($)", [ _aum(t) for t in df.index ])

    # Arrange columns we want to show (skip gracefully if any missing)
    factor_cols = ["Momentum", "Cost", "Liquidity", "Volatility", "Drawdown", "Structure", "Tracking", "News"]
    raw_cols    = ["Avg Volume", "Volatility (Ann.)", "Max Drawdown"]  # NO "Expense Ratio"

    keep = ["Name", "Type", "Last Price ($)", "AUM ($)", "Score (0–100)"] \
           + [c for c in factor_cols if c in df.columns] \
           + [c for c in raw_cols if c in df.columns]
    df = df[[c for c in keep if c in df.columns]]

    # Sort by score and add Rank; expose Ticker as first column
    df_sorted = df.sort_values(by="Score (0–100)", ascending=False).copy()
    df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))
    out = df_sorted.reset_index(names="Ticker")

    # Ensure numeric for display formatting (prevents Streamlit warnings)
    for num_col in ["Last Price ($)", "AUM ($)", "Avg Volume", "Volatility (Ann.)", "Max Drawdown"]:
        if num_col in out.columns:
            out[num_col] = pd.to_numeric(out[num_col], errors="coerce")

    # Percent scale for these two raw risk inputs (NOT Avg Volume/AUM/Last Price)
    for c in ["Volatility (Ann.)", "Max Drawdown"]:
        if c in out.columns:
            out[c] = out[c] * 100.0

    # Final visible order: Ticker first
    ordered = ["Ticker", "Rank", "Name", "Type", "Score (0–100)", "Last Price ($)", "AUM ($)"] \
              + factor_cols + raw_cols
    show = out[[c for c in ordered if c in out.columns]]

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d"),
            "Last Price ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Score (0–100)": st.column_config.NumberColumn(format="%.0f"),

            # factors are already 0–100
            "Momentum":  st.column_config.NumberColumn(format="%.0f"),
            "Cost":      st.column_config.NumberColumn(format="%.0f"),
            "Liquidity": st.column_config.NumberColumn(format="%.0f"),
            "Volatility":st.column_config.NumberColumn(format="%.0f"),
            "Drawdown":  st.column_config.NumberColumn(format="%.0f"),
            "Structure": st.column_config.NumberColumn(format="%.0f"),
            "Tracking":  st.column_config.NumberColumn(format="%.0f"),
            "News":      st.column_config.NumberColumn(format="%.0f"),
            # raw inputs
            "Avg Volume":         st.column_config.NumberColumn(format="%.0f"),
            "Volatility (Ann.)":  st.column_config.NumberColumn(format="%.2f%%"),
            "Max Drawdown":       st.column_config.NumberColumn(format="%.0f%%"),
            "AUM ($)": st.column_config.NumberColumn(format="dollar"),

        },
    )

    # --- Methodology (replaces the old glossary) ---
    st.markdown("### About the methodology")
    st.markdown(
        """
**Pipeline.** For each symbol we gather profile fields (expense ratio, average volume, instrument type, name) and recent
prices. From prices we compute **3-month annualized volatility** and **max drawdown** over the lookback. We also derive a
two-window **momentum composite** using ≈63-day and ≈252-day horizons. A lightweight **headline signal** counts positive/
negative keywords in recent news.

**Normalization.** Each raw input is mapped to a [0,1] score using bounded min–max transforms, then clipped:
- Lower-is-better metrics (expense ratio, volatility, drawdown) are **inverted** after scaling.  
- Higher-is-better metrics (volume) are scaled directly.  
- Missing values default to a neutral 0.5 before weighting.

**Structure filter.** We apply explicit penalties when the instrument is an **ETN** or contains **2x/3x/“leveraged”**
in the name, yielding a Structure sub-score. This penalty is risk-aware (see below).

**Momentum.** The momentum pillar blends a short window (~3 months) against a 12-month baseline, rewarding persistent,
recent strength while dampening one-off spikes. The composite is normalized to [0,1] and contributes more at higher risk.

**Aggregation.** Pillar scores are combined with weights that depend on your **Risk** slider. We start from a balanced
baseline and multiply per-pillar weights, then re-normalize so weights sum to 1. The final total is clipped to [0,1]
and shown as **0–100**.

**Risk setting (how weights & penalties shift):**
- **Low** — Emphasizes *Cost*, *Structure*, and *Risk control* (Volatility/Drawdown). Momentum’s influence is reduced and
structure penalties are harsher, favoring conservative, set-and-forget funds.
- **Medium** — Balanced blend; no extra tilts beyond the baseline.
- **High** — Boosts *Momentum* and *Liquidity*, and **softens** Volatility/Drawdown/Structure penalties so faster,
higher-beta names can surface.

This score is an educational summary of multiple signals—not advice. Always sanity-check constituents, taxes, and fit to
your own objectives.
"""
    )

    st.markdown("---")
    st.caption("© Fundamental — educational only. Do your own research.")



# -------- Compare --------
with tab_compare:
    import pandas as pd
    st.subheader("Compare Funds")

    # Small CSS tweaks (scoped to this page render)
    st.markdown("""
        <style>
        /* compact pill-style download button */
        div[data-testid="stDownloadButton"] button {
            padding: 0.30rem 0.70rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            line-height: 1.1;
        }
        </style>
    """, unsafe_allow_html=True)

    # Build raw (returns are decimals, e.g., 0.237 = +23.7%)
    cmp_df = compare.build_compare_table(tickers, profiles, prices)

    # Pretty headers / order
    rename_map = {
        "ticker": "Ticker",
        "name": "Name",
        "type": "Type",
        "price": "Last Price ($)",
        "as_of": "As of",
        "return_ytd": "YTD Return",
        "return_1m": "1M Return",
        "return_3m": "3M Return",
        "return_1y": "1Y Return",
        "return_5y": "5Y Return",
        "vol_3m_ann": "Volatility (Ann., 3M)",
        "max_drawdown": "Max Drawdown",
        "dividend_yield": "Dividend Yield",
        "aum": "AUM ($)",
    }
    df = cmp_df.rename(columns=rename_map)

    ordered = [
        "Ticker", "Name", "Type",
        "Last Price ($)", "As of",
        "YTD Return", "1M Return", "3M Return", "1Y Return", "5Y Return",
        "Volatility (Ann., 3M)", "Max Drawdown",
        "Expense Ratio", "Dividend Yield", "AUM ($)",
    ]
    df = df[[c for c in ordered if c in df.columns]].copy()

    if "Ticker" in df.columns:
        df = df.set_index("Ticker")

    # DISPLAY: scale decimals to percent for UI only (keeps numeric sorting)
    percent_cols = [
        "YTD Return", "1M Return", "3M Return", "1Y Return", "5Y Return",
        "Volatility (Ann., 3M)", "Max Drawdown", "Expense Ratio", "Dividend Yield",
    ]
    df_display = df.copy()
    for col in percent_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col] * 100.0

    # Left-aligned, non-stretch Sort control (doesn't affect the table width)
    sort_options = list(df_display.columns)
    default_sort = "1Y Return" if "1Y Return" in sort_options else sort_options[0]
    sort_col = st.selectbox(
        "Sort by",
        sort_options,
        index=sort_options.index(default_sort),
        key="compare_sort",
    )

    # Make only THIS select box narrower, keep it left-aligned
    st.markdown("""
    <style>
    /* Limit just the 'Sort by' select's inner box; don't touch the table */
    div[data-testid="stSelectbox"][aria-label="Sort by"] > div {
    max-width: 320px;   /* tweak to taste: 280–400px */
    }
    </style>
    """, unsafe_allow_html=True)


    # Sorting direction (lower is better for these)
    lower_is_better = {"Expense Ratio", "Volatility (Ann., 3M)"}
    ascending = sort_col in lower_is_better
    df_display = df_display.sort_values(by=sort_col, ascending=ascending)

    # Table
    st.dataframe(
        df_display,
        width="stretch",
        column_config={
            "Last Price ($)": st.column_config.NumberColumn(format="$%.2f"),
            "As of": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
            "YTD Return": st.column_config.NumberColumn(format="%.2f%%"),
            "1M Return": st.column_config.NumberColumn(format="%.2f%%"),
            "3M Return": st.column_config.NumberColumn(format="%.2f%%"),
            "1Y Return": st.column_config.NumberColumn(format="%.2f%%"),
            "5Y Return": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatility (Ann., 3M)": st.column_config.NumberColumn(format="%.2f%%"),
            "Max Drawdown": st.column_config.NumberColumn(format="%.0f%%"),
            #"Expense Ratio": st.column_config.NumberColumn(format="%.2f%%"),
            "Dividend Yield": st.column_config.NumberColumn(format="%.2f%%"),
            "AUM ($)": st.column_config.NumberColumn(format="dollar"),
        },
    )

    # Compact download row (aligned left; small pill)
    dleft, dright = st.columns([1, 9])
    with dleft:
        st.download_button(
            "⬇️ CSV",
            df_display.reset_index().to_csv(index=False).encode(),
            file_name="fund_compare.csv",
            mime="text/csv",
            help="Download this table as CSV"
        )


    # Quick glossary
    st.markdown("### Glossary")
    st.markdown(
        """
- **YTD/1M/3M/1Y/5Y Return:** Price change from the start of each period to the latest date in view.\n
  *Note: This uses adjusted prices (dividends & splits applied like Yahoo Finance). 5Y uses the last 5 years of available history.*
- **Volatility (Ann., 3M):** Annualized standard deviation of daily returns over ~63 trading days.
- **Max Drawdown:** Largest %-drop from any prior peak within the window (less negative is better).
- **Expense Ratio:** Annual operating fee charged by the fund.
- **Dividend Yield:** Trailing 12-month dividend yield when available (from profile data).
- **AUM ($):** Assets under management (USD).
- **Type:** Instrument structure (ETF, Mutual Fund, ETN, etc.).
"""
    )

    st.markdown("---")
    st.caption(
        "© Fundamental — Educational tool only. Not investment advice. Data from public web sources (incl. Yahoo Finance via yfinance) "
        "may be delayed or incomplete. No warranty of accuracy or fitness. Past performance is not indicative of future results."
    )


# -------- Discover Top Funds --------
with tab_discover:
    import re
    from fundintel.universe import DEFAULT_UNIVERSE

    st.subheader("Top Funds by Transparent Score")
    st.caption(
        "Scores blend fundamentals and market behavior, adapted to your risk setting: "
        "• Cost (expense ratio) • Liquidity (volume) • Volatility (realized) • Drawdown "
        "• Structure (penalize ETNs/leveraged) • Momentum (multi-window) • Recent multi-period "
        "performance vs. S&P 500 • Light news signal. We show the same return stats as Compare, "
        "and always include the ^GSPC benchmark for context."
    )

    # ── Controls ──────────────────────────────────────────────────────────────────
    colf, coln = st.columns([3, 1])
    with colf:
        st.markdown("**Universe:** A base list of ~350 ETFs/stocks is preloaded.")
        extras_text = st.text_input(
            "Add tickers (optional)",
            value="",
            key="discover_extra_tickers",
            placeholder="e.g., ARKG, TECL, BTI",
            help="Comma/space/newline separated. These will be appended to the default universe (no removals).",
        )
        with st.expander("Show base universe (read-only)", expanded=False):
            st.code(", ".join(DEFAULT_UNIVERSE), language="text")

    with coln:
        fetch_news = st.checkbox(
            "Include News Boost",
            value=False,
            help="Slightly slower; small weight in score.",
            key="discover_include_news",
        )

    # Parse EXTRA tickers, append to DEFAULT_UNIVERSE, dedupe
    extras = [t.strip().upper() for t in re.split(r"[,\s]+", extras_text or "") if t.strip()]
    base_uni = list(DEFAULT_UNIVERSE)  # do not mutate the imported list
    merged_uni = list(dict.fromkeys(base_uni + extras))  # preserve order, drop dups

    # Controls for safety/perf
    max_universe = st.number_input(
        "Max universe size to fetch (safety cap)",
        min_value=50, max_value=2000, value=400, step=50,
        help="Fetch/score at most this many symbols from the merged universe.",
        key="discover_max_universe",
    )
    uni = merged_uni[: int(max_universe)]

    if not uni:
        st.info("No universe to rank. Add at least one ticker.")
        st.stop()

    # Always fetch ^GSPC for the benchmark mini-table + relative checks
    fetch_set = uni.copy()
    if "^GSPC" not in fetch_set:
        fetch_set.append("^GSPC")

    # ── Data fetch + status panel ────────────────────────────────────────────────
    @st.cache_data(show_spinner=False, ttl=60 * 30)
    def discover_batch_fetch(tickers, include_news: bool):
        prof, pxs, nws = {}, {}, {}
        # Keep simple spinners per ticker for uncached runs
        for i, t in enumerate(tickers, 1):
            with st.spinner(f"Fetching {t} ({i}/{len(tickers)})"):
                try:
                    prof[t] = data.fetch_profile(t) or {}
                except Exception:
                    prof[t] = {}
                try:
                    pxs[t] = data.fetch_prices(t, period="5y")
                except Exception:
                    pxs[t] = None
                try:
                    nws[t] = news.fetch_news(t) if include_news else []
                except Exception:
                    nws[t] = []
        return prof, pxs, nws

    status = st.status("Preparing data…", expanded=True)
    status.update(label=f"Fetching {len(fetch_set)} symbols…", state="running")
    profiles_u, prices_u, news_u = discover_batch_fetch(fetch_set, fetch_news)

    # ── Benchmark mini-table (^GSPC) ────────────────────────────────────────────
    # Build compare table once (we'll reuse it for main table too)
    cmp_df = compare.build_compare_table(fetch_set, profiles_u, prices_u).copy()
    rename_map_cmp = {
        "ticker": "Ticker",
        "name": "Name",
        "type": "Type",
        "price": "Last Price ($)",
        "as_of": "As of",
        "return_ytd": "YTD Return",
        "return_1m": "1M Return",
        "return_3m": "3M Return",
        "return_1y": "1Y Return",
        "return_5y": "5Y Return",
        "vol_3m_ann": "Volatility (Ann., 3M)",
        "max_drawdown": "Max Drawdown",
        "expense_ratio": "Expense Ratio",
        "dividend_yield": "Dividend Yield",
        "aum": "AUM ($)",
    }
    cmp_df.rename(columns=rename_map_cmp, inplace=True)

    if "^GSPC" in cmp_df["Ticker"].values:
        g = cmp_df.loc[cmp_df["Ticker"] == "^GSPC", [
            "Ticker", "Last Price ($)", "YTD Return", "1M Return", "3M Return",
            "1Y Return", "5Y Return", "Volatility (Ann., 3M)", "Max Drawdown", "Dividend Yield"
        ]].copy()
        # Display as % for readability
        for c in ["YTD Return", "1M Return", "3M Return", "1Y Return", "5Y Return",
                  "Volatility (Ann., 3M)", "Max Drawdown", "Dividend Yield"]:
            if c in g.columns:
                g[c] = g[c] * 100.0

        st.markdown("**Benchmark: S&P 500 (^GSPC)**")
        st.dataframe(
            g,
            hide_index=True,
            width="stretch",
            column_config={
                "Last Price ($)": st.column_config.NumberColumn(format="$%.2f"),
                "YTD Return": st.column_config.NumberColumn(format="%.2f%%"),
                "1M Return": st.column_config.NumberColumn(format="%.2f%%"),
                "3M Return": st.column_config.NumberColumn(format="%.2f%%"),
                "1Y Return": st.column_config.NumberColumn(format="%.2f%%"),
                "5Y Return": st.column_config.NumberColumn(format="%.2f%%"),
                "Volatility (Ann., 3M)": st.column_config.NumberColumn(format="%.2f%%"),
                "Max Drawdown": st.column_config.NumberColumn(format="%.0f%%"),
                "Dividend Yield": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    # ── Score computation with progress bar ─────────────────────────────────────
    status.update(label="Computing scores…", state="running")
    progress = st.progress(0.0)

    rows = []
    for i, t in enumerate(uni, 1):
        pr = profiles_u.get(t) or {}
        px = prices_u.get(t)
        if isinstance(px, pd.DataFrame) and px.empty:
            px = None
        nw = news_u.get(t) or []
        try:
            s = score.compute_score(
                ticker=t,
                profile=pr,
                prices=px,
                news_items=nw,
                risk_tolerance=risk,    # reuse global sidebar control
                horizon=horizon,        # reuse global sidebar control
            )
            rows.append(s)
        except Exception:
            rows.append({"ticker": t, "score_0_100": float("nan")})
        progress.progress(i / max(1, len(uni)))

    if not rows:
        status.update(label="No results to display.", state="error")
        st.stop()

    df_score = pd.DataFrame(rows).set_index("ticker")

    # Add profile convenience columns
    def _name(t):
        p = profiles_u.get(t) or {}
        return p.get("longName") or p.get("shortName") or ""
    def _type(t):
        p = profiles_u.get(t) or {}
        return p.get("instrument_type") or ""
    def _aum(t):
        p = profiles_u.get(t) or {}
        return p.get("aum")

    # Score-side pretty names
    rename_map_score = {
        "score_0_100": "Score (0–100)",
        "cost_score": "Cost",
        "liquidity_score": "Liquidity",
        "volatility_score": "Volatility",
        "drawdown_score": "Drawdown",
        "structure_score": "Structure",
        "momentum_score": "Momentum",
        "tracking_score": "Tracking",
        "news_score": "News",
        "expense_ratio": "Expense Ratio",
        "avg_volume": "Avg Volume",
        "vol_annualized": "Vol (Ann.)",
        "max_drawdown": "Max DD",
    }
    df_score.rename(columns=rename_map_score, inplace=True)

    # Insert convenience columns at the front
    df_score.insert(0, "Name", [ _name(t) for t in df_score.index ])
    df_score.insert(1, "Type", [ _type(t) for t in df_score.index ])
    df_score.insert(2, "AUM ($)", [ _aum(t) for t in df_score.index ])

    # Keep only non-overlapping compare fields to merge
    cmp_keep = [
        "Ticker", "Last Price ($)", "As of",
        "YTD Return", "1M Return", "3M Return", "1Y Return", "5Y Return",
        "Volatility (Ann., 3M)", "Max Drawdown",
        "Dividend Yield",
    ]
    cmp_part = cmp_df[[c for c in cmp_keep if c in cmp_df.columns]].copy()

    # Merge score table with compare subset (left join on user universe)
    merged = (
        df_score
        .reset_index(names="Ticker")
        .merge(cmp_part, on="Ticker", how="left")
        .set_index("Ticker")
    )

    # Sort & rank
    sorted_df = merged.sort_values(by="Score (0–100)", ascending=False).copy()
    sorted_df.insert(0, "Rank", range(1, len(sorted_df) + 1))

    # Final visible columns (Ticker first)
    display_order = [
        "Rank",
        "Name",
        "Type",
        "Last Price ($)",
        "Score (0–100)",
        "YTD Return",
        "1M Return",
        "3M Return",
        "1Y Return",
        "5Y Return",
        "Volatility (Ann., 3M)",
        "Max Drawdown",
        "Dividend Yield",
    ]
    out = sorted_df.reset_index()
    final_cols = ["Ticker"] + [c for c in display_order if c in out.columns]
    out = out[final_cols].copy()

    # Format % columns for display
    for c in ["YTD Return", "1M Return", "3M Return", "1Y Return", "5Y Return",
              "Volatility (Ann., 3M)", "Max Drawdown", "Dividend Yield"]:
        if c in out.columns:
            out[c] = out[c] * 100.0

    status.update(label="Done.", state="complete")

    # ── Render main table ───────────────────────────────────────────────────────
    st.dataframe(
        out,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d"),
            "Last Price ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Score (0–100)": st.column_config.NumberColumn(format="%.0f"),
            "YTD Return": st.column_config.NumberColumn(format="%.2f%%"),
            "1M Return": st.column_config.NumberColumn(format="%.2f%%"),
            "3M Return": st.column_config.NumberColumn(format="%.2f%%"),
            "1Y Return": st.column_config.NumberColumn(format="%.2f%%"),
            "5Y Return": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatility (Ann., 3M)": st.column_config.NumberColumn(format="%.2f%%"),
            "Max Drawdown": st.column_config.NumberColumn(format="%.0f%%"),
            "Dividend Yield": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    st.download_button(
        "⬇️ Download ranked table (CSV)",
        out.to_csv(index=False).encode(),
        file_name="fund_rankings.csv",
        mime="text/csv",
        key="discover_download_csv",
    )

    st.caption(
        "© Fundamental — Educational tool only. Not investment advice. Data from public web sources (incl. Yahoo Finance via yfinance) "
        "may be delayed or incomplete. No warranty of accuracy or fitness. Past performance is not indicative of future results."
    )
