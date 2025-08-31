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

tab_overview, tab_perf, tab_news, tab_risk, tab_compare, tab_discover = st.tabs(
    ["Overview", "Performance", "News & Filings", "Risk & Score", "Compare", "Discover Top Funds"]
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
    - **Data sources.** Quotes, profiles, and history are fetched from **public endpoints** (e.g., Yahoo Finance via yfinance), plus public news feeds. Data can be **delayed, revised, or incomplete**.
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
# with tabs[2]:
#     st.subheader("Top Holdings (Under Construction)")
#     st.warning(
#         "Holdings data could not be displayed. "
#         "Unlike price, volume, and profile information (which Yahoo Finance exposes via their quote APIs), "
#         "constituent-level holdings are **not returned in the standard Yahoo endpoints**. "
#         "Yahoo migrated these holdings feeds behind gated, cookie-authenticated services. "
#         "At the same time, most ETF issuers (e.g. Global X, iShares, Vanguard) now inject their daily holdings tables "
#         "into the page dynamically using JavaScript. This means a simple HTTP request (like those used in `requests` or `yfinance`) "
#         "returns only the page shell — without the rendered table — so parsing yields an empty result. "

#     )

    # st.markdown("---")
    # st.caption("© Fundamental. For education only. Do your own research.")
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
        "Scoring system combining costs, liquidity, volatility/drawdown, structure, momentum, tracking (placeholder), and simple news sentiment."

    )
    st.caption("Note: Use risk slider to adjust score - Low should boost conservative funds (low vol / shallow drawdowns / low cost) and penalize high-octane names. \n"
        " High should reward momentum/liquidity and penalize volatility/drawdown more lightly."
    )
    rows = []
    for t in tickers:
        pr = profiles[t]
        px = prices[t]
        nw = news_items.get(t, [])
        s = score.compute_score(
            ticker=t,
            profile=pr,
            prices=px,
            news_items=nw,
            risk_tolerance=risk,
            horizon=horizon
        )
        rows.append(s)

    raw_df = pd.DataFrame(rows).set_index("ticker")

    # Prettify column headers
    rename_map = {
        "score_0_100": "Score (0–100)",
        "cost_score": "Cost Score",
        "liquidity_score": "Liquidity Score",
        "volatility_score": "Volatility Score",
        "drawdown_score": "Drawdown Score",
        "structure_score": "Structure Score",
        "momentum_score": "Momentum Score",
        "tracking_score": "Tracking Score",
        "news_score": "News Score",
        "expense_ratio": "Expense Ratio",
        "avg_volume": "Avg Volume",
        "vol_annualized": "Volatility (Ann.)",
        "max_drawdown": "Max Drawdown",
    }
    df = raw_df.rename(columns=rename_map)

    # Nicer column order (only keep those that exist)
    ordered_cols = [
        "Score (0–100)",
        "Cost Score",
        "Liquidity Score",
        "Volatility Score",
        "Drawdown Score",
        "Structure Score",
        "Momentum Score",
        "Tracking Score",
        "News Score",
        "Expense Ratio",
        "Avg Volume",
        "Volatility (Ann.)",
        "Max Drawdown",
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    st.dataframe(df, width='stretch')

    # Quick glossary
    st.markdown("### Glossary")
    st.markdown(
        """
- **Score (0–100):** Weighted blend of the factors below (educational, not advice).
- **Cost Score:** Lower **expense ratio** → higher score.
- **Liquidity Score:** Based on **average trading volume**; higher volume gets a higher score.
- **Volatility Score:** Lower **3-month annualized volatility** → higher score.
- **Drawdown Score:** Shallower **max drop from peak** over the window → higher score.
- **Structure Score:** Penalties for **ETNs** and **leveraged** products (e.g., 2×/3×).
- **Momentum Score:** Near-term trend (≈ 3-month) vs 12-month baseline.
- **Tracking Score:** Placeholder (currently neutral) until a benchmark is wired.
- **News Score:** Small boost/penalty from recent headline keywords.
- **Expense Ratio:** Annual fee charged by the fund.
- **Avg Volume:** Typical daily shares traded (liquidity proxy).
- **Volatility (Ann.):** Annualized stdev of recent daily returns (≈ last 3 months).
- **Max Drawdown:** Worst peak-to-trough drop in the window.
"""
    )
    st.markdown("---")
    st.caption("© Fundamental. For education only. Do your own research.")

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
        "expense_ratio": "Expense Ratio",
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
            "Expense Ratio": st.column_config.NumberColumn(format="%.2f%%"),
            "Dividend Yield": st.column_config.NumberColumn(format="%.2f%%"),
            "AUM ($)": st.column_config.NumberColumn(format="$%,d"),
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
# -------- Discover Top Funds --------
with tab_discover:
    import re
    from fundintel.universe import DEFAULT_UNIVERSE

    st.subheader("Top Funds by Transparent Score")
    st.caption(
        "Ranks a curated universe using the same factor model: cost, liquidity, volatility, drawdown, "
        "structure, momentum, tracking (placeholder), and news."
    )

    # Controls (unique keys)
    colf, coln = st.columns([3, 1])
    with colf:
        universe_text = st.text_area(
            "Universe (tickers, comma/space/newline separated)",
            value=",".join(DEFAULT_UNIVERSE),
            height=100,
            key="discover_universe_text",
        )
    with coln:
        fetch_news = st.checkbox(
            "Include News Boost",
            value=False,
            help="Slightly slower; small weight in score.",
            key="discover_include_news",
        )

    # Parse universe
    uni = [t.strip().upper() for t in re.split(r"[,\s]+", universe_text) if t.strip()]
    # Deduplicate while preserving order; cap to avoid silly large runs
    uni = list(dict.fromkeys(uni))[:150]

    if not uni:
        st.info("Add at least one ticker to the universe.")
        st.stop()

    # Batched fetch with cache (unique function name to avoid cache collisions)
    @st.cache_data(show_spinner=False, ttl=60 * 30)
    def discover_batch_fetch(tickers, include_news: bool):
        prof, pxs, nws = {}, {}, {}
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

    profiles_u, prices_u, news_u = discover_batch_fetch(uni, fetch_news)

    # Compute scores safely
    rows = []
    for t in uni:
        pr = profiles_u.get(t) or {}
        px = prices_u.get(t)
        # Normalize px for score function
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
        except Exception as e:
            # If a single ticker fails, keep going; record a minimal row
            rows.append({"ticker": t, "score_0_100": float("nan")})

    if not rows:
        st.info("No results to display.")
        st.stop()

    df = pd.DataFrame(rows).set_index("ticker")

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

    # Rename / pretty columns
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
        "expense_ratio": "Expense Ratio",
        "avg_volume": "Avg Volume",
        "vol_annualized": "Vol (Ann.)",
        "max_drawdown": "Max DD",
    }
    df = df.rename(columns=rename_map)

    # Insert convenience columns at the front
    df.insert(0, "Name", [ _name(t) for t in df.index ])
    df.insert(1, "Type", [ _type(t) for t in df.index ])
    df.insert(2, "AUM ($)", [ _aum(t) for t in df.index ])

    # Build sort options from existing columns
    candidate_sort_cols = [
        "Score (0–100)", "Momentum", "Cost", "Liquidity",
        "Volatility", "Drawdown", "Structure", "Tracking", "News"
    ]
    sort_options = [c for c in candidate_sort_cols if c in df.columns]

    # Unique key to avoid collisions
    sort_col = st.selectbox("Sort by", sort_options, index=0, key="discover_sort_select")

    # Sort (higher is better for all these)
    df_sorted = df.sort_values(by=sort_col, ascending=False)

    # Insert Rank (1..N) as first column (recomputed on every sort)
    df_sorted = df_sorted.copy()
    df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))

    # Display
    st.dataframe(
        df_sorted,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d"),
            "Expense Ratio": st.column_config.NumberColumn(format="%.2f%%"),
            "Vol (Ann.)": st.column_config.NumberColumn(format="%.2f%%"),
            "Max DD": st.column_config.NumberColumn(format="%.0f%%"),
            "Avg Volume": st.column_config.NumberColumn(format="%,.0f"),
            "AUM ($)": st.column_config.NumberColumn(format="$%,.0f"),
        },
    )


    # Download (unique key for button)
    st.download_button(
        "⬇️ Download ranked table (CSV)",
        df_sorted.reset_index().to_csv(index=False).encode(),
        file_name="fund_rankings.csv",
        mime="text/csv",
        key="discover_download_csv",
    )

    st.caption(
        "© Fundamental — Educational tool only. Not investment advice. Data from public web sources (incl. Yahoo Finance via yfinance) "
        "may be delayed or incomplete. No warranty of accuracy or fitness. Past performance is not indicative of future results."
    )