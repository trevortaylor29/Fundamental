import yfinance as yf
import pandas as pd

print("Checking ARGT...")

tk = yf.Ticker("ARGT")

# fund_top_holdings
ft = getattr(tk, "fund_top_holdings", None)
print("\nfund_top_holdings type:", type(ft))
if isinstance(ft, pd.DataFrame):
    print(ft.head())

# get_funds_data
try:
    gd = tk.get_funds_data()
    print("\nget_funds_data keys:", gd.keys())
    th = (gd.get("topHoldings") or {}).get("holdings")
    if th:
        print("topHoldings sample:", th[:3])
except Exception as e:
    print("\nget_funds_data error:", e)

# _fund_data
raw = getattr(tk, "_fund_data", None)
print("\n_fund_data type:", type(raw))
if isinstance(raw, dict):
    print("raw _fund_data keys:", list(raw.keys()))
