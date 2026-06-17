import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None

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
        "TXN", "QCOM", "AMD", "INTU"
    ]


@st.cache_data(show_spinner=False)
def fetch_price_data(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        price_df = pd.DataFrame()

        for field in ["Adj Close", "Close"]:
            if field in data.columns.get_level_values(0):
                price_df = data[field].copy()
                break

            if field in data.columns.get_level_values(1):
                price_df = data.xs(field, level=1, axis=1).copy()
                break

        if price_df.empty:
            return pd.DataFrame()
    else:
        if "Adj Close" in data.columns:
            price_df = pd.DataFrame({tickers[0]: data["Adj Close"]})
        elif "Close" in data.columns:
            price_df = pd.DataFrame({tickers[0]: data["Close"]})
        else:
            return pd.DataFrame()

    price_df.columns = [str(c).upper() for c in price_df.columns]
    wanted = [t.upper() for t in tickers]
    price_df = price_df[[c for c in wanted if c in price_df.columns]]

    return price_df.dropna(how="all").sort_index().ffill()


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty or len(price_df.dropna(how="all")) < 2:
        return pd.DataFrame()

    clean_prices = price_df.dropna(how="all").ffill()
    return clean_prices.pct_change().dropna(how="all")


@st.cache_data(show_spinner=False)
def fetch_intraday_data(ticker: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period="5d",
        interval="15m",
        auto_adjust=False,
        progress=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close"]
    if not all(col in data.columns for col in needed):
        return pd.DataFrame()

    return data.dropna(subset=needed)


def make_candlestick_chart(ticker: str, data: pd.DataFrame):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name=ticker,
            )
        ]
    )

    fig.update_layout(
        title=f"{ticker} — 15 Minute Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def fetch_news_headlines(ticker: str, date: datetime, api_key: str) -> List[str]:
    if requests is None or not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": (date - timedelta(days=3)).strftime("%Y-%m-%d"),
        "to": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 5,
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")][:5]
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


@st.cache_resource(show_spinner=False)
def load_finbert():
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model
    except Exception:
        return None, None


def sentiment_finbert(headlines: List[str]) -> float:
    if not headlines:
        return 0.0

    tokenizer, model = load_finbert()
    if tokenizer is None or model is None:
        return 0.0

    model.eval()
    scores = []

    for headline in headlines:
        try:
            inputs = tokenizer(headline, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            negative = probs[0].item()
            positive = probs[2].item()
            scores.append(positive - negative)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def compute_sentiment(headlines: List[str], method: str) -> float:
    if method == "VADER":
        return sentiment_vader(headlines)
    if method == "FinBERT":
        return sentiment_finbert(headlines)
    return sentiment_textblob(headlines)


def rank_stocks(
    returns_df: pd.DataFrame,
    sentiments: Dict[str, float],
    weight_price: float,
    weight_sentiment: float,
) -> pd.DataFrame:
    last_returns = returns_df.iloc[-1].fillna(0)

    ret_min = last_returns.min()
    ret_max = last_returns.max()

    if ret_max != ret_min:
        norm_returns = (last_returns - ret_min) / (ret_max - ret_min)
    else:
        norm_returns = last_returns * 0

    sentiment_series = pd.Series(sentiments).reindex(last_returns.index).fillna(0)

    sent_min = sentiment_series.min()
    sent_max = sentiment_series.max()

    if sent_max != sent_min:
        norm_sentiment = (sentiment_series - sent_min) / (sent_max - sent_min)
    else:
        norm_sentiment = sentiment_series * 0

    score = (weight_price * norm_returns) + (weight_sentiment * norm_sentiment)

    result = pd.DataFrame(
        {
            "Return": last_returns,
            "Sentiment": sentiment_series,
            "Score": score,
        }
    )

    return result.sort_values("Score", ascending=False)


def main():
    st.title("NASDAQ Stock Dashboard")
    st.write(
        "Ranks selected NASDAQ stocks using recent price movement and news sentiment. "
        "Educational only — not financial advice."
    )

    today = datetime.now().date()

    selected_date = st.date_input(
        "Select analysis date",
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
        ["TextBlob", "VADER", "FinBERT"],
    )

    weight_price = st.sidebar.slider(
        "Price weight",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
    )

    weight_sentiment = 1.0 - weight_price
    st.sidebar.write(f"Sentiment weight: {weight_sentiment:.2f}")

    show_candles = st.sidebar.checkbox("Show 15-minute candlestick charts", value=True)

    if not selected_tickers:
        st.warning("Select at least one ticker.")
        return

    if sentiment_method == "FinBERT" and (
        AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None
    ):
        st.sidebar.warning("FinBERT dependencies are not installed. Falling back to TextBlob.")
        sentiment_method = "TextBlob"

    start_date = selected_date - timedelta(days=60)
    end_date = today + timedelta(days=1)

    with st.spinner("Downloading price data..."):
        price_df = fetch_price_data(selected_tickers, start_date, end_date)

    if price_df.empty:
        st.error("No price data returned from Yahoo Finance.")
        return

    returns_df = compute_daily_returns(price_df)

    if returns_df.empty:
        st.error(
            "Still not enough usable price rows to calculate returns. "
            "Try rerunning the app or selecting fewer tickers."
        )
        st.write("Debug price data:")
        st.dataframe(price_df.tail(10))
        return

    available_tickers = [t.upper() for t in selected_tickers if t.upper() in returns_df.columns]

    if not available_tickers:
        st.error("None of the selected tickers had usable return data.")
        return

    news_api_key = os.getenv("NEWS_API_KEY", "")
    sentiments = {}

    if news_api_key:
        with st.spinner("Fetching headlines and scoring sentiment..."):
            for ticker in available_tickers:
                headlines = fetch_news_headlines(ticker, datetime.combine(selected_date, datetime.min.time()), news_api_key)
                sentiments[ticker] = compute_sentiment(headlines, sentiment_method)
    else:
        sentiments = {ticker: 0.0 for ticker in available_tickers}
        st.info("No NEWS_API_KEY found. Sentiment scores are neutral until you add one.")

    ranking_df = rank_stocks(
        returns_df[available_tickers],
        sentiments,
        weight_price,
        weight_sentiment,
    )

    top_n = ranking_df.head(10)

    st.subheader(f"Top {len(top_n)} Stocks")
    st.dataframe(
        top_n.style.format(
            {
                "Return": "{:.2%}",
                "Sentiment": "{:.3f}",
                "Score": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Daily Price History")
    st.line_chart(price_df[top_n.index], use_container_width=True)

    if show_candles:
        st.subheader("15-Minute Candlestick Charts")

        for ticker in top_n.index:
            with st.spinner(f"Loading 15-minute chart for {ticker}..."):
                candle_df = fetch_intraday_data(ticker)

            if candle_df.empty:
                st.warning(f"No 15-minute candlestick data available for {ticker}.")
                continue

            fig = make_candlestick_chart(ticker, candle_df)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "---\n"
        "**Disclaimer:** This dashboard is for informational and educational purposes only. "
        "It is not investment advice."
    )


if __name__ == "__main__":
    main()
