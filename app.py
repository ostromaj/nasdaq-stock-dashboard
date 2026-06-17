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


st.set_page_config(page_title="NASDAQ Stock Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_nasdaq_tickers() -> List[str]:
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "NVDA", "META",
        "ADBE", "CMCSA", "PEP", "COST", "CSCO", "AVGO", "INTC", "TMUS",
        "TXN", "QCOM", "AMD", "INTU", "NFLX", "AMAT", "MU", "LRCX",
        "PANW", "CRWD", "SHOP", "MELI", "ISRG", "BKNG"
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
def fetch_intraday_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="5d",
        interval="15m",
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


def normalize_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    series = series.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    if not higher_is_better:
        series = -series

    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(50, index=series.index)

    return ((series - min_val) / (max_val - min_val)) * 100


def calculate_features(stock_data: Dict[str, pd.DataFrame], qqq_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    qqq_return_20d = 0
    if not qqq_df.empty and len(qqq_df) >= 21:
        qqq_return_20d = (qqq_df["Close"].iloc[-1] / qqq_df["Close"].iloc[-21]) - 1

    for ticker, df in stock_data.items():
        if df.empty or len(df) < 30:
            continue

        close = df["Close"]
        volume = df["Volume"]

        return_5d = (close.iloc[-1] / close.iloc[-6]) - 1 if len(close) >= 6 else 0
        return_20d = (close.iloc[-1] / close.iloc[-21]) - 1 if len(close) >= 21 else 0
        return_60d = (close.iloc[-1] / close.iloc[-61]) - 1 if len(close) >= 61 else 0

        momentum_raw = (0.50 * return_5d) + (0.35 * return_20d) + (0.15 * return_60d)
        relative_strength_raw = return_20d - qqq_return_20d

        avg_volume_20d = volume.tail(20).mean()
        latest_volume = volume.iloc[-1]
        volume_surge_raw = latest_volume / avg_volume_20d if avg_volume_20d else 1

        daily_returns = close.pct_change().dropna()
        volatility_20d = daily_returns.tail(20).std() if len(daily_returns) >= 20 else 0

        rows.append(
            {
                "Ticker": ticker,
                "Last Price": close.iloc[-1],
                "5D Return": return_5d,
                "20D Return": return_20d,
                "60D Return": return_60d,
                "Momentum Raw": momentum_raw,
                "Relative Strength Raw": relative_strength_raw,
                "Volume Surge Raw": volume_surge_raw,
                "Volatility Raw": volatility_20d,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Ticker")

    df["Momentum Score"] = normalize_score(df["Momentum Raw"], True)
    df["Relative Strength Score"] = normalize_score(df["Relative Strength Raw"], True)
    df["Volume Score"] = normalize_score(df["Volume Surge Raw"], True)
    df["Volatility Score"] = normalize_score(df["Volatility Raw"], False)

    return df


def fetch_news_headlines(ticker: str, api_key: str) -> List[str]:
    if requests is None or not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 8,
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")][:8]
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


def make_professional_score(features_df: pd.DataFrame, sentiments: Dict[str, float]) -> pd.DataFrame:
    df = features_df.copy()

    sentiment_series = pd.Series(sentiments).reindex(df.index).fillna(0)
    df["Sentiment Raw"] = sentiment_series
    df["Sentiment Score"] = normalize_score(df["Sentiment Raw"], True)

    df["Bull Score"] = (
        0.40 * df["Momentum Score"]
        + 0.20 * df["Relative Strength Score"]
        + 0.15 * df["Volume Score"]
        + 0.15 * df["Sentiment Score"]
        + 0.10 * df["Volatility Score"]
    )

    df["Sentiment Label"] = df["Sentiment Raw"].apply(
        lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral"
    )

    df["Momentum Label"] = df["Momentum Score"].apply(
        lambda x: "Strong" if x >= 70 else "Weak" if x <= 35 else "Average"
    )

    df["Volume Label"] = df["Volume Surge Raw"].apply(
        lambda x: "Elevated" if x >= 1.25 else "Light" if x <= 0.75 else "Normal"
    )

    return df.sort_values("Bull Score", ascending=False)


def make_candlestick_chart(ticker: str, data: pd.DataFrame):
    data = data.copy()

    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
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
            increasing_line_width=3,
            decreasing_line_width=3,
            increasing_fillcolor="rgba(0,200,100,0.85)",
            decreasing_fillcolor="rgba(220,50,50,0.85)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA20"],
            mode="lines",
            name="20 MA",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA50"],
            mode="lines",
            name="50 MA",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            opacity=0.35,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} — 15 Minute Candlestick Chart",
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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


def main():
    st.title("NASDAQ Stock Dashboard")
    st.write(
        "Ranks NASDAQ stocks using momentum, relative strength, volume, volatility, and news sentiment. "
        "Educational only — not financial advice."
    )

    today = datetime.now().date()

    selected_date = st.sidebar.date_input(
        "Analysis date",
        value=today,
        max_value=today,
        min_value=today - timedelta(days=365),
    )

    tickers = load_nasdaq_tickers()

    selected_tickers = st.sidebar.multiselect(
        "Select stocks",
        options=tickers,
        default=tickers,
    )

    sentiment_method = st.sidebar.selectbox(
        "Sentiment model",
        ["TextBlob", "VADER"],
    )

    news_api_input = st.sidebar.text_input(
        "News API Key",
        value=os.getenv("NEWS_API_KEY", ""),
        type="password",
    )

    show_candles = st.sidebar.checkbox("Show 15-minute candlestick charts", value=True)

    st.sidebar.markdown("### Professional Score Weights")
    st.sidebar.write("Momentum: 40%")
    st.sidebar.write("Relative Strength: 20%")
    st.sidebar.write("Volume Surge: 15%")
    st.sidebar.write("News Sentiment: 15%")
    st.sidebar.write("Volatility Trend: 10%")

    if not selected_tickers:
        st.warning("Select at least one ticker.")
        return

    start_date = selected_date - timedelta(days=365)
    end_date = today + timedelta(days=1)

    stock_data = {}

    with st.spinner("Downloading daily stock data..."):
        for ticker in selected_tickers:
            df = fetch_daily_stock_data(ticker, start_date, end_date)
            if not df.empty:
                stock_data[ticker] = df

        qqq_df = fetch_daily_stock_data("QQQ", start_date, end_date)

    if not stock_data:
        st.error("No usable stock data returned.")
        return

    features_df = calculate_features(stock_data, qqq_df)

    if features_df.empty:
        st.error("Not enough historical data to calculate professional scores.")
        return

    sentiments = {}

    if news_api_input:
        with st.spinner("Fetching news headlines and scoring sentiment..."):
            for ticker in features_df.index:
                headlines = fetch_news_headlines(ticker, news_api_input)
                sentiments[ticker] = compute_sentiment(headlines, sentiment_method)
    else:
        st.info("No News API key entered. Sentiment will be treated as neutral.")
        sentiments = {ticker: 0.0 for ticker in features_df.index}

    ranking_df = make_professional_score(features_df, sentiments)

    top_n = ranking_df.head(10)

    display_cols = [
        "Bull Score",
        "Last Price",
        "5D Return",
        "20D Return",
        "60D Return",
        "Momentum Label",
        "Volume Label",
        "Sentiment Label",
        "Sentiment Raw",
        "Volume Surge Raw",
        "Volatility Raw",
    ]

    st.subheader("Top 10 NASDAQ Stocks")
    st.dataframe(
        top_n[display_cols].style.format(
            {
                "Bull Score": "{:.1f}",
                "Last Price": "${:.2f}",
                "5D Return": "{:.2%}",
                "20D Return": "{:.2%}",
                "60D Return": "{:.2%}",
                "Sentiment Raw": "{:.3f}",
                "Volume Surge Raw": "{:.2f}x",
                "Volatility Raw": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Score Breakdown")
    st.bar_chart(
        top_n[
            [
                "Momentum Score",
                "Relative Strength Score",
                "Volume Score",
                "Sentiment Score",
                "Volatility Score",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Daily Price History")
    daily_price_df = pd.DataFrame(
        {ticker: stock_data[ticker]["Close"] for ticker in top_n.index if ticker in stock_data}
    )
    st.line_chart(daily_price_df, use_container_width=True)

    if show_candles:
        st.subheader("15-Minute Candlestick Charts")

        for ticker in top_n.index:
            with st.spinner(f"Loading 15-minute chart for {ticker}..."):
                intraday_df = fetch_intraday_data(ticker)

            if intraday_df.empty:
                st.warning(f"No 15-minute candlestick data available for {ticker}.")
                continue

            fig = make_candlestick_chart(ticker, intraday_df)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "---\n"
        "**Disclaimer:** This dashboard is for informational and educational purposes only. "
        "It is not investment advice."
    )


if __name__ == "__main__":
    main()
