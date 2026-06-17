"""
Stock Dashboard Application

This Streamlit application provides a dashboard that ranks NASDAQ stocks based on
daily performance and a simple sentiment proxy derived from recent news headlines.
The goal of the app is educational – it demonstrates how to pull market data
and basic sentiment information in order to build a ranking of stocks. It is
**not** financial advice and should not be used to make investment decisions.

Features:
  * Pulls historical price data for a list of NASDAQ tickers using `yfinance`.
  * Calculates daily returns and ranks the stocks by performance.
  * Retrieves a handful of recent news headlines for each ticker from a free
    news API (if a key is provided via the `NEWS_API_KEY` environment variable).
  * Performs a naive sentiment analysis on the news using the `textblob` library.
  * Combines price performance and news sentiment into a simple score used to
    rank the top 10 stocks for a selected date.
  * Displays the results in an interactive table along with charts for each
    selected ticker.

To run this app locally:
    pip install -r requirements.txt
    streamlit run app.py

Environment variables:
    NEWS_API_KEY (optional): API key for NewsAPI.org to fetch real headlines.

Disclaimer:
    The rankings produced by this application are for demonstration purposes only.
    They do not constitute financial advice. Users should do their own research
    or consult a licensed professional before making any investment decisions.
"""

import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from textblob import TextBlob  # type: ignore
except ImportError:
    # If textblob isn't available, we'll define a dummy sentiment function.
    TextBlob = None  # type: ignore

# Optional sentiment libraries
try:
    # VADER sentiment analysis (lexicon and rule‐based)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
except ImportError:
    SentimentIntensityAnalyzer = None  # type: ignore

try:
    # HuggingFace Transformers for FinBERT
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch  # type: ignore
except ImportError:
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    torch = None  # type: ignore

try:
    import requests
except ImportError:
    requests = None  # type: ignore


@st.cache_data(show_spinner=False)
def load_nasdaq_tickers() -> List[str]:
    """
    Return a list of NASDAQ tickers. Ideally this would query an API or a
    maintained list. For the purposes of this demo we provide a curated
    subset of widely traded NASDAQ stocks to avoid excessive API calls.
    """
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "NVDA", "META",
        "ADBE", "CMCSA", "PEP", "COST", "CSCO", "AVGO", "INTC", "TMUS",
        "TXN", "QCOM", "AMD", "INTU"
    ]


@st.cache_data(show_spinner=False)
def fetch_price_data(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch adjusted close price data for the given tickers between start and end
    dates (inclusive of start, exclusive of end). Returns a DataFrame where
    columns are tickers and rows are dates.
    """
    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        progress=False
    )
    # yfinance returns a DataFrame with MultiIndex columns when multiple
    # tickers are supplied. In that case the first level of the column
    # index is the ticker and the second level is the price field (e.g.
    # 'Open', 'High', 'Low', 'Close', 'Adj Close', etc.). When only one
    # ticker is supplied, yfinance returns a single‐level DataFrame with
    # columns for each field.
    #
    # The previous implementation attempted to handle these two cases by
    # indexing into `data` with `(ticker, 'Adj Close')` for the multi‐index
    # case or `data['Adj Close'][ticker]` for the single‐index fallback.
    # However, this can fail with a KeyError when the structure of `data`
    # is different than expected. Instead, we explicitly check for a
    # MultiIndex and use `.xs()` to extract the 'Adj Close' rows. For the
    # single ticker case we rename the column to the ticker symbol.
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Extract the 'Adj Close' level for all tickers. This returns
            # a DataFrame with columns equal to the ticker symbols.
            price_df = data.xs('Adj Close', level=1, axis=1)
        except KeyError:
            # If 'Adj Close' is not present, fall back to 'Close'.
            price_df = data.xs('Close', level=1, axis=1)
    else:
        # Only a single ticker was downloaded. Extract the 'Adj Close'
        # column and rename it to the ticker symbol so that the caller
        # receives a consistent DataFrame regardless of the number of
        # tickers.
        if 'Adj Close' in data.columns:
            # Wrap in DataFrame to preserve two‐dimensional structure
            price_df = pd.DataFrame({tickers[0]: data['Adj Close']})
        elif 'Close' in data.columns:
            price_df = pd.DataFrame({tickers[0]: data['Close']})
        else:
            # Unexpected structure. Return empty DataFrame and let
            # downstream code handle the error gracefully.
            price_df = pd.DataFrame()
    # Ensure columns are ordered according to the tickers list. Missing
    # tickers will be dropped automatically.
    price_df = price_df[[c for c in tickers if c in price_df.columns]]
    return price_df


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from price DataFrame. The return is
    (today - previous_day) / previous_day.
    """
    return price_df.pct_change().dropna()


def fetch_news_headlines(ticker: str, date: datetime, api_key: str) -> List[str]:
    """
    Fetch recent news headlines for a given ticker. Uses the NewsAPI.org
    'everything' endpoint. Returns up to 3 headlines around the given date.

    This function requires the 'requests' library and a valid API key. If
    requests or an API key is not available, the function returns an empty
    list.
    """
    if requests is None or not api_key:
        return []

    # Query NewsAPI for articles containing the company name or ticker symbol.
    url = "https://newsapi.org/v2/everything"
    query_params = {
        "q": ticker,
        "from": (date - timedelta(days=3)).strftime("%Y-%m-%d"),
        "to": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 5,
        "apiKey": api_key,
    }
    try:
        response = requests.get(url, params=query_params, timeout=10)
        response.raise_for_status()
    except Exception:
        return []
    articles = response.json().get("articles", [])
    headlines = [article.get("title", "") for article in articles]
    return headlines[:3]


def sentiment_score_textblob(headlines: List[str]) -> float:
    """
    Compute a sentiment score using TextBlob. Returns the average polarity of
    the provided headlines. If TextBlob is unavailable or no headlines are
    provided, a neutral score of 0.0 is returned.
    """
    if not headlines or TextBlob is None:
        return 0.0
    polarities = []
    for headline in headlines:
        try:
            blob = TextBlob(headline)
            polarities.append(blob.sentiment.polarity)
        except Exception:
            polarities.append(0.0)
    return sum(polarities) / len(polarities) if polarities else 0.0


def sentiment_score_vader(headlines: List[str]) -> float:
    """
    Compute a sentiment score using the VADER model. The compound score is
    returned, averaged across all headlines. If VADER is unavailable or
    no headlines are provided, a neutral score of 0.0 is returned.
    """
    if not headlines or SentimentIntensityAnalyzer is None:
        return 0.0
    try:
        analyzer = SentimentIntensityAnalyzer()
    except Exception:
        return 0.0
    scores = []
    for headline in headlines:
        try:
            polarity = analyzer.polarity_scores(headline)
            scores.append(polarity.get("compound", 0.0))
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


@st.cache_resource(show_spinner=False)
def load_finbert_model():
    """
    Load the FinBERT tokenizer and model. This is cached so that the
    large model is loaded only once. Returns a tuple of (tokenizer, model)
    or (None, None) if FinBERT cannot be imported.
    """
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model
    except Exception:
        return None, None


def sentiment_score_finbert(headlines: List[str]) -> float:
    """
    Compute a sentiment score using the FinBERT model. For each headline
    FinBERT outputs probabilities for negative, neutral and positive classes.
    We convert these into a single score by subtracting the negative
    probability from the positive probability and averaging across all
    headlines. If FinBERT is unavailable or no headlines are provided,
    returns 0.0.
    """
    if not headlines:
        return 0.0
    tokenizer, model = load_finbert_model()
    if tokenizer is None or model is None:
        return 0.0
    model.eval()
    scores = []
    for headline in headlines:
        try:
            inputs = tokenizer(headline, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # FinBERT output logits correspond to [negative, neutral, positive]
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            pos = probabilities[2].item()
            neg = probabilities[0].item()
            scores.append(pos - neg)
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def compute_sentiment(headlines: List[str], method: str) -> float:
    """
    Compute sentiment based on the selected method. Supported methods are
    'TextBlob', 'VADER', and 'FinBERT'. Defaults to TextBlob when the
    method is not recognized or required libraries are missing.
    """
    if method == "VADER":
        return sentiment_score_vader(headlines)
    if method == "FinBERT":
        return sentiment_score_finbert(headlines)
    # Default to TextBlob
    return sentiment_score_textblob(headlines)


def rank_stocks(
    price_returns: pd.DataFrame,
    news_sentiments: Dict[str, float],
    weight_price: float = 0.7,
    weight_sentiment: float = 0.3
) -> pd.DataFrame:
    """
    Combine price performance and news sentiment into a composite score.

    The score is a weighted sum of normalized price return and normalized
    sentiment for each stock on the selected date. Price return and sentiment
    are normalized between 0 and 1. The DataFrame returned contains
    individual components and the final score.
    """
    # Extract the last row (most recent date) for returns
    last_returns = price_returns.iloc[-1]
    # Normalize returns and sentiments to [0, 1]
    ret_min, ret_max = last_returns.min(), last_returns.max()
    norm_returns = (last_returns - ret_min) / (ret_max - ret_min) if ret_max != ret_min else last_returns * 0
    sent_values = pd.Series(news_sentiments)
    sent_min, sent_max = sent_values.min(), sent_values.max()
    norm_sentiments = (sent_values - sent_min) / (sent_max - sent_min) if sent_max != sent_min else sent_values * 0
    # Combine scores
    score = weight_price * norm_returns + weight_sentiment * norm_sentiments
    result = pd.DataFrame({
        "Return": last_returns,
        "Sentiment": sent_values,
        "Score": score
    })
    return result.sort_values(by="Score", ascending=False)


def main() -> None:
    """
    Main entry point for the Streamlit dashboard.
    """
    st.set_page_config(page_title="NASDAQ Stock Dashboard", layout="wide")
    st.title("NASDAQ Stock Dashboard: Top Stocks by Performance and Sentiment")
    st.markdown(
        "This dashboard ranks selected NASDAQ stocks based on daily price performance "
        "and recent news sentiment. The analysis is educational and not intended as "
        "financial advice."
    )

    # Date selection
    today = datetime.now().date()
    default_date = today - timedelta(days=1)
    selected_date = st.date_input(
        "Select a date for analysis",
        value=default_date,
        max_value=today,
        min_value=today - timedelta(days=365)
    )

    # Parameters
    weight_price = st.slider("Weight for price performance", 0.0, 1.0, 0.7, 0.05)
    weight_sentiment = 1.0 - weight_price
    st.write(f"Weight for news sentiment: {weight_sentiment:.2f}")

    # Allow the user to choose which sentiment analysis model to use.  
    # TextBlob is the default and requires only the textblob package.  
    # VADER uses the vaderSentiment package and calculates a compound score.  
    # FinBERT uses a transformer model fine‑tuned for financial text and requires
    # the transformers and torch packages.  
    sentiment_method = st.sidebar.selectbox(
        "Sentiment analysis model",
        options=["TextBlob", "VADER", "FinBERT"],
        format_func=lambda m: m if (m != "FinBERT" or (AutoTokenizer is not None and AutoModelForSequenceClassification is not None and torch is not None)) else f"{m} (unavailable)"
    )
    # If FinBERT is selected but not available, notify the user
    if sentiment_method == "FinBERT" and (AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None):
        st.sidebar.warning("FinBERT is not installed. Please install transformers and torch to use this model.")
        sentiment_method = "TextBlob"

    # Load ticker list
    tickers = load_nasdaq_tickers()
    st.sidebar.header("Stock Universe")
    selected_tickers = st.sidebar.multiselect(
        "Select NASDAQ stocks to analyze",
        options=tickers,
        default=tickers
    )

    if not selected_tickers:
        st.warning("Please select at least one ticker.")
        return

    # Fetch data
    with st.spinner("Downloading price data..."):
        start_date = selected_date - timedelta(days=7)  # fetch a week of data around the date
        end_date = selected_date + timedelta(days=1)
        price_df = fetch_price_data(selected_tickers, start_date, end_date)
        if price_df.empty:
            st.error("No price data available for the selected date range.")
            return
        returns_df = compute_daily_returns(price_df)
        if returns_df.empty:
            st.error("Insufficient data to compute returns.")
            return

    # Fetch news sentiment
    news_api_key = os.getenv("NEWS_API_KEY", "")
    sentiments: Dict[str, float] = {}
    if news_api_key:
        with st.spinner("Fetching news headlines and computing sentiment..."):
            for ticker in selected_tickers:
                headlines = fetch_news_headlines(ticker, selected_date, news_api_key)
                sentiments[ticker] = compute_sentiment(headlines, sentiment_method)
    else:
        # Without an API key we assign neutral sentiment to all stocks
        sentiments = {ticker: 0.0 for ticker in selected_tickers}

    # Rank stocks
    ranking_df = rank_stocks(returns_df[selected_tickers], sentiments, weight_price, weight_sentiment)
    top_n = ranking_df.head(10)

    # Display results
    st.subheader(f"Top {len(top_n)} Stocks for {selected_date.strftime('%Y-%m-%d')}")
    st.dataframe(top_n.style.format({
        "Return": "{:.2%}",
        "Sentiment": "{:.2f}",
        "Score": "{:.2f}"
    }))

    # Show charts for top stocks
    st.subheader("Price Performance Charts")
    for ticker in top_n.index:
        chart_df = price_df[[ticker]].dropna()
        st.line_chart(chart_df, height=200, use_container_width=True)
        st.caption(f"Price history for {ticker}")

    st.markdown(
        "---\n"
        "**Disclaimer:** This dashboard is for informational purposes only and does not provide any investment advice. "
        "Please conduct your own research or consult a financial advisor before making investment decisions."
    )


if __name__ == "__main__":
    main()
