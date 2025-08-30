from typing import List, Dict, Any
import feedparser, datetime as dt, re

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

def fetch_news(ticker: str) -> List[Dict[str, Any]]:
    q = f"{ticker} ETF OR ETN OR fund"
    url = GOOGLE_NEWS_RSS.format(query=re.sub(r"\s+","+", q))
    try:
        parsed = feedparser.parse(url)
    except Exception:
        return []
    items = []
    for e in parsed.entries[:15]:
        try:
            published = dt.datetime(*e.published_parsed[:6])
        except Exception:
            published = None
        items.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": published,
            "source": getattr(e, "source", {}).get("title") if hasattr(e, "source") else "",
        })
    return items
