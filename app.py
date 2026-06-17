import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    import requests
except ImportError:
    requests = None


st.set_page_config(page_title="NASDAQ Bull Score Dashboard", layout="wide")


def get_news_api_key() -> str:
    try:
        return st.secrets.get("NEWS_API_KEY", "")
    except Exception:
        return os.getenv("NEWS_API_KEY", "")


@st.cache_data(show_spinner=False)
def load_nasdaq_tickers() -> List[str]:
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
        "AMD", "NFLX", "ADBE", "PEP", "CSCO", "INTU", "QCOM", "TXN", "AMAT", "BKNG",
        "AMGN", "HON", "SBUX", "ISRG", "VRTX", "ADP", "GILD", "MDLZ", "LRCX", "REGN",
        "MU", "PANW", "ADI", "KLAC", "SNPS", "CDNS", "MELI", "MAR", "CRWD", "PYPL",
        "ABNB", "MNST", "ORLY", "CTAS", "CSX", "PCAR", "ROP", "NXPI", "WDAY", "FTNT",
        "CHTR", "ROST", "PAYX", "AEP", "KDP", "TEAM", "MRVL", "DDOG", "CPRT", "FAST",
        "ODFL", "EA", "EXC", "VRSK", "KHC", "BKR", "XEL", "GEHC", "CCEP", "FANG",
        "LULU", "ZS", "TTD", "IDXX", "MCHP", "CSGP", "ON", "BIIB", "DXCM", "ANSS",
        "CDW", "ILMN", "WBD", "GFS", "MDB", "ARM", "SMCI", "DASH", "LIN", "AZN"
    ]


@st.cache_data(show_spinner=False)
def fetch_daily_stock_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in df.columns for col in needed):
        return pd.DataFrame()

    df = df[needed].dropna()
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def fetch_intraday_data(ticker: str, interval: str) -> pd.DataFrame:
    period_map = {
        "5m": "5d",
        "15m": "5d",
        "30m": "1mo",
        "60m": "1mo",
    }

    df = yf.download(
        ticker,
        period=period_map.get(interval, "5d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in df.columns for col in needed):
        return pd.DataFrame()

    df = df[needed].dropna()
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def fetch_news_headlines(ticker: str, api_key: str) -> List[str]:
    if requests is None or not api_key:
        return []

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": f"{ticker} stock OR {ticker} earnings OR {ticker} shares",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 8,
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [article.get("title", "") for article in articles if article.get("title")][:8]
    except Exception:
        return []


def sentiment_textblob(headlines: List[str]) -> float:
    if not headlines or TextBlob is None:
        return 0.0

    scores = []

    for headline in headlines:
        try:
            scores.append(TextBlob(headline).sentiment.polarity)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def sentiment_vader(headlines: List[str]) -> float:
    if not headlines or SentimentIntensityAnalyzer is None:
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    scores = []

    for headline in headlines:
        try:
            scores.append(analyzer.polarity_scores(headline)["compound"])
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def compute_sentiment(headlines: List[str], method: str) -> float:
    if method == "VADER":
        return sentiment_vader(headlines)

    return sentiment_textblob(headlines)


def calculate_features(ticker: str, df: pd.DataFrame, qqq_df: pd.DataFrame, sentiment_raw: float) -> Dict:
    close = df["Close"]
    volume = df["Volume"]

    qqq_return_20d = 0.0

    if not qqq_df.empty and len(qqq_df) >= 21:
        qqq_return_20d = (qqq_df["Close"].iloc[-1] / qqq_df["Close"].iloc[-21]) - 1

    return_5d = (close.iloc[-1] / close.iloc[-6]) - 1 if len(close) >= 6 else 0.0
    return_20d = (close.iloc[-1] / close.iloc[-21]) - 1 if len(close) >= 21 else 0.0
    return_60d = (close.iloc[-1] / close.iloc[-61]) - 1 if len(close) >= 61 else 0.0

    momentum_raw = (0.50 * return_5d) + (0.35 * return_20d) + (0.15 * return_60d)
    relative_strength_raw = return_20d - qqq_return_20d

    avg_volume_20d = volume.tail(20).mean()
    latest_volume = volume.iloc[-1]
    volume_surge_raw = latest_volume / avg_volume_20d if avg_volume_20d else 1.0

    daily_returns = close.pct_change().dropna()
    volatility_20d = daily_returns.tail(20).std() if len(daily_returns) >= 20 else 0.0

    momentum_score = max(0, min(100, 50 + momentum_raw * 500))
    relative_strength_score = max(0, min(100, 50 + relative_strength_raw * 500))
    volume_score = max(0, min(100, volume_surge_raw * 50))
    sentiment_score = max(0, min(100, 50 + sentiment_raw * 50))
    volatility_score = max(0, min(100, 100 - volatility_20d * 1000))

    bull_score = (
        0.40 * momentum_score
        + 0.20 * relative_strength_score
        + 0.15 * volume_score
        + 0.15 * sentiment_score
        + 0.10 * volatility_score
    )

    return {
        "Ticker": ticker,
        "Bull Score": bull_score,
        "Last Price": close.iloc[-1],
        "5D Return": return_5d,
        "20D Return": return_20d,
        "60D Return": return_60d,
        "Relative Strength vs QQQ": relative_strength_raw,
        "Volume Surge": volume_surge_raw,
        "Volatility 20D": volatility_20d,
        "Sentiment Raw": sentiment_raw,
        "Momentum Score": momentum_score,
        "Relative Strength Score": relative_strength_score,
        "Volume Score": volume_score,
        "Sentiment Score": sentiment_score,
        "Volatility Score": volatility_score,
        "Sentiment Label": "Positive" if sentiment_raw > 0.05 else "Negative" if sentiment_raw < -0.05 else "Neutral",
        "Momentum Label": "Strong" if momentum_score >= 70 else "Weak" if momentum_score <= 35 else "Average",
        "Volume Label": "Elevated" if volume_surge_raw >= 1.25 else "Light" if volume_surge_raw <= 0.75 else "Normal",
    }


def make_candlestick_chart(ticker: str, data: pd.DataFrame, interval: str):
    data = data.copy()

    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candles",
            increasing_line_width=4,
            decreasing_line_width=4,
            increasing_fillcolor="rgba(0, 200, 100, 0.85)",
            decreasing_fillcolor="rgba(220, 50, 50, 0.85)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], mode="lines", name="20 MA", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], mode="lines", name="50 MA", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MA200"], mode="lines", name="200 MA", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", opacity=0.35), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"{ticker} — {interval} Candlestick Chart",
            y=0.985,
            x=0.02,
            xanchor="left",
            yanchor="top",
            font=dict(size=22),
        ),
        height=780,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=25, r=25, t=125, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.075,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.5], pattern="hour"),
        ]
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def make_volume_profile_chart(ticker: str, data: pd.DataFrame):
    df = data.copy()

    if df.empty:
        return None

    df["Price Bin"] = pd.cut(df["Close"], bins=20)
    profile = df.groupby("Price Bin", observed=False)["Volume"].sum().reset_index()
    profile["Price"] = profile["Price Bin"].apply(lambda x: x.mid)

    fig = go.Figure(
        go.Bar(
            x=profile["Volume"],
            y=profile["Price"],
            orientation="h",
            name="Volume Profile",
        )
    )

    fig.update_layout(
        title=f"{ticker} Volume Profile",
        template="plotly_dark",
        height=450,
        margin=dict(l=25, r=25, t=60, b=25),
        xaxis_title="Volume",
        yaxis_title="Price Level",
    )

    return fig


def main():
    st.title("NASDAQ Bull Score Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")

    tickers = load_nasdaq_tickers()
    news_api_key = get_news_api_key()

    comparison_tickers = st.sidebar.multiselect(
        "Stocks to scan",
        options=tickers,
        default=tickers,
        help="These stocks will be ranked against each other by Bull Score.",
    )

    sentiment_method = st.sidebar.selectbox("Sentiment model", ["VADER", "TextBlob"])

    candle_interval = st.sidebar.selectbox(
        "Candlestick interval",
        ["5m", "15m", "30m", "60m"],
        index=1,
    )

    chart_count = st.sidebar.slider(
        "Number of top Bull Score charts",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    st.sidebar.markdown("### Score Weights")
    st.sidebar.write("Momentum: 40%")
    st.sidebar.write("Relative Strength vs QQQ: 20%")
    st.sidebar.write("Volume Surge: 15%")
    st.sidebar.write("News Sentiment: 15%")
    st.sidebar.write("Volatility Stability: 10%")

    if not comparison_tickers:
        st.warning("Please select at least one stock to scan.")
        return

    today = datetime.now().date()
    start_date = today - timedelta(days=365)
    end_date = today + timedelta(days=1)

    with st.spinner("Downloading market data and calculating Bull Scores..."):
        qqq_df = fetch_daily_stock_data("QQQ", start_date, end_date)

        stock_data = {}
        for ticker in comparison_tickers:
            df = fetch_daily_stock_data(ticker, start_date, end_date)
            if not df.empty:
                stock_data[ticker] = df

    if not stock_data:
        st.error("No usable price data found.")
        return

    feature_rows = []

    with st.spinner("Scoring sentiment and ranking stocks..."):
        for ticker, df in stock_data.items():
            sentiment_raw = 0.0

            if news_api_key:
                headlines = fetch_news_headlines(ticker, news_api_key)
                sentiment_raw = compute_sentiment(headlines, sentiment_method)

            feature_rows.append(calculate_features(ticker, df, qqq_df, sentiment_raw))

    comparison_df = (
        pd.DataFrame(feature_rows)
        .set_index("Ticker")
        .sort_values("Bull Score", ascending=False)
    )

    top_stock = comparison_df.index[0]
    top_metrics = comparison_df.loc[top_stock]
    top_bull_tickers = comparison_df.head(chart_count).index.tolist()

    st.subheader("Highest Bull Score Today")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top Stock", top_stock)
    col2.metric("Bull Score", f"{top_metrics['Bull Score']:.1f}/100")
    col3.metric("20D Return", f"{top_metrics['20D Return']:.2%}")
    col4.metric("Rel Strength vs QQQ", f"{top_metrics['Relative Strength vs QQQ']:.2%}")

    st.subheader("Bull Score Rankings")

    display_df = comparison_df[
        [
            "Bull Score",
            "Last Price",
            "5D Return",
            "20D Return",
            "60D Return",
            "Relative Strength vs QQQ",
            "Volume Surge",
            "Sentiment Raw",
            "Volatility 20D",
        ]
    ]

    st.dataframe(
        display_df.style.format(
            {
                "Bull Score": "{:.1f}",
                "Last Price": "${:.2f}",
                "5D Return": "{:.2%}",
                "20D Return": "{:.2%}",
                "60D Return": "{:.2%}",
                "Relative Strength vs QQQ": "{:.2%}",
                "Volume Surge": "{:.2f}x",
                "Sentiment Raw": "{:.3f}",
                "Volatility 20D": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Bull Score Comparison")
    st.bar_chart(comparison_df.head(20)[["Bull Score"]], use_container_width=True)

    st.subheader("Close Price Comparison")
    close_comparison = pd.DataFrame(
        {
            ticker: stock_data[ticker]["Close"]
            for ticker in top_bull_tickers
            if ticker in stock_data
        }
    )

    if not close_comparison.empty:
        st.line_chart(close_comparison, use_container_width=True)

    chart_tickers = st.sidebar.multiselect(
        "Candlestick charts shown",
        options=comparison_df.index.tolist(),
        default=top_bull_tickers,
        help="Defaults to the highest Bull Score stocks.",
    )

    if not chart_tickers:
        chart_tickers = top_bull_tickers

    st.subheader(f"Top {len(chart_tickers)} Bull Score Candlestick Charts")

    for ticker in chart_tickers:
        with st.spinner(f"Loading {ticker} {candle_interval} candlestick chart..."):
            intraday_df = fetch_intraday_data(ticker, candle_interval)

        if intraday_df.empty:
            st.warning(f"No {candle_interval} candlestick data available for {ticker}.")
            continue

        fig = make_candlestick_chart(ticker, intraday_df, candle_interval)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"{ticker} Volume Profile"):
            volume_profile = make_volume_profile_chart(ticker, intraday_df)
            if volume_profile:
                st.plotly_chart(volume_profile, use_container_width=True)

    st.subheader(f"{top_stock} Recent News")

    if news_api_key:
        top_headlines = fetch_news_headlines(top_stock, news_api_key)

        if top_headlines:
            for headline in top_headlines:
                st.write(f"- {headline}")
        else:
            st.info("No recent headlines found.")
    else:
        st.warning("NEWS_API_KEY is missing from Streamlit secrets.")

    st.markdown(
        "---\n"
        "**Disclaimer:** This dashboard is for informational and educational purposes only. "
        "It is not investment advice."
    )


if __name__ == "__main__":
    main()
