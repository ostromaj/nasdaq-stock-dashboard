import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    import requests
except Exception:
    requests = None


st.set_page_config(page_title="NASDAQ Bull Score Dashboard", layout="wide")


COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "META": "Meta Platforms", "GOOGL": "Alphabet Google", "GOOG": "Alphabet Google",
    "TSLA": "Tesla", "AVGO": "Broadcom", "COST": "Costco", "AMD": "Advanced Micro Devices",
    "NFLX": "Netflix", "ADBE": "Adobe", "PEP": "PepsiCo", "CSCO": "Cisco",
    "INTU": "Intuit", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "AMAT": "Applied Materials", "BKNG": "Booking Holdings", "MU": "Micron",
    "PANW": "Palo Alto Networks", "CRWD": "CrowdStrike", "PYPL": "PayPal",
    "SBUX": "Starbucks", "LRCX": "Lam Research", "MELI": "MercadoLibre",
    "MAR": "Marriott", "ABNB": "Airbnb", "MRVL": "Marvell", "DDOG": "Datadog",
    "SMCI": "Super Micro Computer", "ARM": "Arm Holdings", "DASH": "DoorDash",
}


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


def clean_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in df.columns for col in needed):
        return pd.DataFrame()

    df = df[needed].dropna()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    return df.sort_index()


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
    return clean_yfinance_df(df)


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
    return clean_yfinance_df(df)


@st.cache_data(show_spinner=False)
def fetch_news_headlines(ticker: str, api_key: str) -> List[str]:
    if requests is None:
        return ["ERROR: requests is not installed"]

    if not api_key:
        return ["ERROR: NEWS_API_KEY missing"]

    company = COMPANY_NAMES.get(ticker, ticker)
    query = f'("{company}" OR "{ticker}") AND (stock OR shares OR earnings OR analyst OR market OR revenue OR guidance)'

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": api_key,
            },
            timeout=10,
        )

        data = response.json()

        if data.get("status") == "error":
            return [f"NEWS API ERROR: {data.get('message', 'Unknown error')}"]

        articles = data.get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")][:10]

    except Exception as e:
        return [f"NEWS FETCH ERROR: {e}"]


def real_headlines_only(headlines: List[str]) -> List[str]:
    return [
        h for h in headlines
        if h
        and not h.startswith("ERROR")
        and not h.startswith("NEWS API ERROR")
        and not h.startswith("NEWS FETCH ERROR")
    ]


def sentiment_textblob(headlines: List[str]) -> float:
    headlines = real_headlines_only(headlines)

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
    headlines = real_headlines_only(headlines)

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
    return sentiment_vader(headlines) if method == "VADER" else sentiment_textblob(headlines)


def get_data_as_of(df: pd.DataFrame, selected_date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    selected_ts = pd.Timestamp(selected_date)

    try:
        selected_ts = selected_ts.tz_localize(None)
    except Exception:
        pass

    return df[df.index <= selected_ts].copy()


def get_next_trading_row(full_df: pd.DataFrame, selected_date) -> Optional[pd.Series]:
    if full_df is None or full_df.empty:
        return None

    selected_ts = pd.Timestamp(selected_date)

    try:
        selected_ts = selected_ts.tz_localize(None)
    except Exception:
        pass

    future = full_df[full_df.index > selected_ts]
    return None if future.empty else future.iloc[0]


def get_next_trading_date(full_df: pd.DataFrame, selected_date) -> str:
    selected_ts = pd.Timestamp(selected_date)

    if full_df is not None and not full_df.empty:
        future = full_df[full_df.index > selected_ts]
        if not future.empty:
            return future.index[0].strftime("%Y-%m-%d")

    next_day = selected_ts + pd.Timedelta(days=1)

    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)

    return next_day.strftime("%Y-%m-%d")


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    df = df.copy()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    df["H-L"] = high - low
    df["H-PC"] = (high - close.shift(1)).abs()
    df["L-PC"] = (low - close.shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(period).mean()

    atr = df["ATR"].iloc[-1]

    if pd.isna(atr) or atr <= 0:
        atr = (high.tail(period).max() - low.tail(period).min()) / period

    return float(atr) if not pd.isna(atr) else 0.0


def calculate_trade_plan(df: pd.DataFrame) -> Dict:
    latest_close = float(df["Close"].iloc[-1])
    atr = calculate_atr(df, 14)

    recent_support = float(df["Low"].tail(20).min())
    recent_resistance = float(df["High"].tail(20).max())

    entry_zone_low = max(recent_support, latest_close - (0.75 * atr))
    entry_zone_high = latest_close
    stop_loss = recent_support - (0.50 * atr)

    target_1 = recent_resistance
    target_2 = recent_resistance + atr

    risk = entry_zone_high - stop_loss
    rr_1 = (target_1 - entry_zone_high) / risk if risk > 0 else 0
    rr_2 = (target_2 - entry_zone_high) / risk if risk > 0 else 0

    return {
        "Suggested Entry Low": entry_zone_low,
        "Suggested Entry High": entry_zone_high,
        "Stop Loss": stop_loss,
        "Target 1": target_1,
        "Target 2": target_2,
        "Risk/Reward Target 1": rr_1,
        "Risk/Reward Target 2": rr_2,
        "ATR14": atr,
    }


def calculate_features(
    ticker: str,
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    selected_date,
    sentiment_raw: float,
) -> Dict:
    close = df["Close"]
    volume = df["Volume"]

    qqq_return_20d = 0.0

    if qqq_df is not None and not qqq_df.empty and len(qqq_df) >= 21:
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

    momentum_contribution = 0.40 * momentum_score
    relative_strength_contribution = 0.20 * relative_strength_score
    volume_contribution = 0.15 * volume_score
    sentiment_contribution = 0.15 * sentiment_score
    volatility_contribution = 0.10 * volatility_score

    bull_score = (
        momentum_contribution
        + relative_strength_contribution
        + volume_contribution
        + sentiment_contribution
        + volatility_contribution
    )

    trade_plan = calculate_trade_plan(df)
    latest_close = float(close.iloc[-1])
    atr_pct = trade_plan["ATR14"] / latest_close if latest_close > 0 else 0

    projected_next_close_return = ((bull_score - 50) / 50) * min(atr_pct, 0.05)
    projected_next_close_return = max(-0.05, min(0.05, projected_next_close_return))
    projected_next_close = latest_close * (1 + projected_next_close_return)

    if bull_score >= 75:
        prediction_label = "Strong Bullish"
    elif bull_score >= 65:
        prediction_label = "Bullish"
    elif bull_score >= 55:
        prediction_label = "Watchlist"
    elif bull_score >= 45:
        prediction_label = "Neutral"
    else:
        prediction_label = "Avoid"

    next_row = get_next_trading_row(full_df, selected_date)
    next_trading_date = get_next_trading_date(full_df, selected_date)

    actual_next_close_return = None
    actual_next_close = None
    actual_next_high = None
    actual_next_low = None
    hit_target_1 = None
    hit_target_2 = None
    hit_stop = None
    prediction_result = "Pending"

    today_date = pd.Timestamp.today().normalize()

    if next_row is not None and next_row.name.normalize() < today_date:
        actual_next_close = float(next_row["Close"])
        actual_next_high = float(next_row["High"])
        actual_next_low = float(next_row["Low"])
        actual_next_close_return = (actual_next_close / latest_close) - 1

        hit_target_1 = actual_next_high >= trade_plan["Target 1"]
        hit_target_2 = actual_next_high >= trade_plan["Target 2"]
        hit_stop = actual_next_low <= trade_plan["Stop Loss"]

        if prediction_label in ["Strong Bullish", "Bullish", "Watchlist"]:
            prediction_result = "Win" if actual_next_close_return > 0 else "Loss"
        else:
            prediction_result = "N/A"

    return {
        "Ticker": ticker,
        "Prediction Date": next_trading_date,
        "Prediction": prediction_label,
        "Prediction Result": prediction_result,
        "Bull Score": bull_score,
        "Last Price": latest_close,
        "Projected Next Close": projected_next_close,
        "Projected Next Close Return": projected_next_close_return,
        "Actual Next Close Return": actual_next_close_return,
        "Actual Next Close": actual_next_close,
        "Actual Next High": actual_next_high,
        "Actual Next Low": actual_next_low,
        "Hit Target 1": hit_target_1,
        "Hit Target 2": hit_target_2,
        "Hit Stop": hit_stop,
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
        "Momentum Contribution": momentum_contribution,
        "Relative Strength Contribution": relative_strength_contribution,
        "Volume Contribution": volume_contribution,
        "Sentiment Contribution": sentiment_contribution,
        "Volatility Contribution": volatility_contribution,
        **trade_plan,
    }


def safe_currency(value):
    return "—" if value is None or pd.isna(value) else f"${float(value):.2f}"


def safe_percent(value):
    return "—" if value is None or pd.isna(value) else f"{float(value):.2%}"


def safe_number(value, digits=1):
    return "—" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def safe_x(value):
    return "—" if value is None or pd.isna(value) else f"{float(value):.2f}x"


def make_display_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    table = df[cols].copy()

    currency_cols = [
        "Last Price", "Projected Next Close", "Actual Next Close",
        "Actual Next High", "Actual Next Low", "Suggested Entry Low",
        "Suggested Entry High", "Stop Loss", "Target 1", "Target 2"
    ]

    percent_cols = [
        "Projected Next Close Return", "Actual Next Close Return", "5D Return",
        "20D Return", "60D Return", "Relative Strength vs QQQ", "Volatility 20D"
    ]

    x_cols = ["Volume Surge", "Risk/Reward Target 1", "Risk/Reward Target 2"]

    for col in currency_cols:
        if col in table.columns:
            table[col] = table[col].apply(safe_currency)

    for col in percent_cols:
        if col in table.columns:
            table[col] = table[col].apply(safe_percent)

    if "Bull Score" in table.columns:
        table["Bull Score"] = table["Bull Score"].apply(lambda x: safe_number(x, 1))

    if "Sentiment Raw" in table.columns:
        table["Sentiment Raw"] = table["Sentiment Raw"].apply(lambda x: safe_number(x, 3))

    for col in x_cols:
        if col in table.columns:
            table[col] = table[col].apply(safe_x)

    return table.fillna("—")


def make_stacked_bull_score_chart(comparison_df: pd.DataFrame, top_n: int = 10):
    df = comparison_df.head(top_n).copy().sort_values("Bull Score", ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=df.index, y=df["Momentum Contribution"], name="Momentum", marker_color="#00cc96"))
    fig.add_trace(go.Bar(x=df.index, y=df["Relative Strength Contribution"], name="Relative Strength", marker_color="#636efa"))
    fig.add_trace(go.Bar(x=df.index, y=df["Volume Contribution"], name="Volume", marker_color="#ffa15a"))
    fig.add_trace(go.Bar(x=df.index, y=df["Sentiment Contribution"], name="Sentiment", marker_color="#ab63fa"))
    fig.add_trace(go.Bar(x=df.index, y=df["Volatility Contribution"], name="Volatility", marker_color="#ef553b"))

    fig.update_layout(
        title=f"Top {len(df)} Bull Score Breakdown — Stacked Contributions",
        template="plotly_dark",
        height=650,
        barmode="stack",
        yaxis_title="Bull Score Points",
        xaxis_title="Ticker",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        margin=dict(l=30, r=30, t=90, b=50),
    )

    fig.update_yaxes(range=[0, max(100, df["Bull Score"].max() * 1.15)])
    return fig


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
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], mode="lines", name="20 MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], mode="lines", name="50 MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MA200"], mode="lines", name="200 MA"), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", opacity=0.35), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — {interval} Candlestick Chart",
        height=780,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=25, r=25, t=125, b=25),
        legend=dict(orientation="h", y=1.075, x=0.01),
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.5], pattern="hour"),
        ]
    )

    return fig


def make_trade_plan_chart(ticker: str, data: pd.DataFrame, metrics: pd.Series, days: int = 90):
    data = data.copy().tail(days)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Recent Closing Price",
            line=dict(width=4),
        )
    )

    levels = [
        ("ENTRY ZONE LOW", metrics["Suggested Entry Low"], "Lower end of preferred buy zone", "dot", "#00cc96"),
        ("ENTRY ZONE HIGH", metrics["Suggested Entry High"], "Upper end of preferred buy zone", "dot", "#00cc96"),
        ("STOP LOSS", metrics["Stop Loss"], "Exit here if trade breaks down", "dash", "#ef553b"),
        ("TARGET 1", metrics["Target 1"], "First profit-taking area", "dash", "#ffa15a"),
        ("TARGET 2", metrics["Target 2"], "Stretch profit target", "dash", "#ab63fa"),
    ]

    for label, price, desc, dash, color in levels:
        fig.add_hline(
            y=price,
            line_dash=dash,
            line_color=color,
            line_width=2,
            annotation_text=f"{label} — ${price:.2f}<br>{desc}",
            annotation_position="right",
            annotation_font=dict(size=13, color=color),
        )

    fig.add_hrect(
        y0=metrics["Suggested Entry Low"],
        y1=metrics["Suggested Entry High"],
        fillcolor="#00cc96",
        opacity=0.15,
        line_width=0,
        annotation_text="Preferred Entry Zone",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"{ticker} Trade Plan — Last {days} Trading Days",
        template="plotly_dark",
        height=720,
        yaxis_title="Price",
        xaxis_title="Date",
        margin=dict(l=35, r=240, t=85, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

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
        xaxis_title="Volume",
        yaxis_title="Price Level",
    )

    return fig


def show_trade_plan_modal(ticker: str, comparison_df: pd.DataFrame, stock_data_asof: Dict[str, pd.DataFrame]):
    metrics = comparison_df.loc[ticker]
    df = stock_data_asof[ticker]

    @st.dialog(f"{ticker} Trade Plan", width="large")
    def trade_plan_popup():
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Bull Score", f"{metrics['Bull Score']:.1f}/100")
        col2.metric("Prediction", metrics["Prediction"])
        col3.metric(
            "Entry Zone",
            f"{safe_currency(metrics['Suggested Entry Low'])} - {safe_currency(metrics['Suggested Entry High'])}",
        )
        col4.metric(
            "Targets",
            f"{safe_currency(metrics['Target 1'])} / {safe_currency(metrics['Target 2'])}",
        )

        st.plotly_chart(make_trade_plan_chart(ticker, df, metrics, days=10), use_container_width=True)

        popup_cols = [
            "Prediction Date",
            "Prediction",
            "Prediction Result",
            "Last Price",
            "Projected Next Close",
            "Projected Next Close Return",
            "Suggested Entry Low",
            "Suggested Entry High",
            "Stop Loss",
            "Target 1",
            "Target 2",
            "Risk/Reward Target 1",
            "Risk/Reward Target 2",
        ]

        st.dataframe(make_display_table(comparison_df.loc[[ticker]], popup_cols), use_container_width=True)

    trade_plan_popup()


def main():
    st.title("NASDAQ Bull Score Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")

    tickers = load_nasdaq_tickers()
    news_api_key = get_news_api_key()
    today = datetime.now().date()

    selected_date = st.sidebar.date_input(
        "Analysis date",
        value=today,
        min_value=today - timedelta(days=365),
        max_value=today,
    )

    custom_ticker = st.sidebar.text_input(
        "Look up any stock",
        placeholder="Example: PLTR, SOFI, HOOD",
    ).upper().strip()

    scan_universe = tickers.copy()

    if custom_ticker:
        scan_universe.insert(0, custom_ticker)

    comparison_tickers = st.sidebar.multiselect(
        "Stocks to scan",
        options=scan_universe,
        default=scan_universe,
    )

    sentiment_method = st.sidebar.selectbox("Sentiment model", ["VADER", "TextBlob"])

    candle_interval = st.sidebar.selectbox(
        "Candlestick interval",
        ["5m", "15m", "30m", "60m"],
        index=1,
    )

    chart_count = st.sidebar.slider("Number of top Bull Score charts", 1, 10, 5)
    breakdown_count = st.sidebar.slider("Number of stocks in stacked Bull Score chart", 5, 20, 10)
    show_news_debug = st.sidebar.checkbox("Show news debug", value=False)

    if not comparison_tickers:
        st.warning("Please select at least one stock to scan.")
        return

    start_date = selected_date - timedelta(days=430)
    end_date = today + timedelta(days=10)

    with st.spinner("Downloading market data and calculating Bull Scores..."):
        qqq_full_df = fetch_daily_stock_data("QQQ", start_date, end_date)
        qqq_asof_df = get_data_as_of(qqq_full_df, selected_date)

        stock_data_full = {}
        stock_data_asof = {}

        for ticker in comparison_tickers:
            full_df = fetch_daily_stock_data(ticker, start_date, end_date)
            asof_df = get_data_as_of(full_df, selected_date)

            if not full_df.empty and not asof_df.empty and len(asof_df) >= 30:
                stock_data_full[ticker] = full_df
                stock_data_asof[ticker] = asof_df

    if not stock_data_asof:
        st.error("No usable price data found for the selected analysis date.")
        return

    feature_rows = []
    headlines_by_ticker = {}

    with st.spinner("Scoring sentiment and ranking stocks..."):
        for ticker, asof_df in stock_data_asof.items():
            sentiment_raw = 0.0
            headlines = []

            if news_api_key:
                headlines = fetch_news_headlines(ticker, news_api_key)
                headlines_by_ticker[ticker] = headlines
                sentiment_raw = compute_sentiment(headlines, sentiment_method)
            else:
                headlines_by_ticker[ticker] = ["ERROR: NEWS_API_KEY missing"]

            feature_rows.append(
                calculate_features(
                    ticker=ticker,
                    df=asof_df,
                    full_df=stock_data_full[ticker],
                    qqq_df=qqq_asof_df,
                    selected_date=selected_date,
                    sentiment_raw=sentiment_raw,
                )
            )

    comparison_df = pd.DataFrame(feature_rows).set_index("Ticker").sort_values("Bull Score", ascending=False)

    top_stock = comparison_df.index[0]
    top_metrics = comparison_df.loc[top_stock]
    top_bull_tickers = comparison_df.head(chart_count).index.tolist()

    main_col, right_col = st.columns([3.7, 0.95])

    with main_col:
        st.subheader(f"Highest Bull Score for {selected_date.strftime('%Y-%m-%d')}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Top Stock", top_stock)
        col2.metric("Bull Score", f"{top_metrics['Bull Score']:.1f}/100")
        col3.metric("Prediction", top_metrics["Prediction"])
        col4.metric("Prediction Date", top_metrics["Prediction Date"])

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Projected Next Close", safe_currency(top_metrics["Projected Next Close"]))
        col6.metric("Projected Close Return", safe_percent(top_metrics["Projected Next Close Return"]))
        col7.metric(
            "Entry Zone",
            f"{safe_currency(top_metrics['Suggested Entry Low'])} - {safe_currency(top_metrics['Suggested Entry High'])}",
        )
        col8.metric(
            "Target 1 / Target 2",
            f"{safe_currency(top_metrics['Target 1'])} / {safe_currency(top_metrics['Target 2'])}",
        )

        display_cols = [
            "Prediction Date",
            "Prediction",
            "Prediction Result",
            "Bull Score",
            "Last Price",
            "Projected Next Close",
            "Projected Next Close Return",
            "Actual Next Close Return",
            "Suggested Entry Low",
            "Suggested Entry High",
            "Stop Loss",
            "Target 1",
            "Target 2",
            "Risk/Reward Target 1",
            "Risk/Reward Target 2",
            "5D Return",
            "20D Return",
            "60D Return",
            "Relative Strength vs QQQ",
            "Volume Surge",
            "Sentiment Raw",
            "Volatility 20D",
        ]

        st.subheader("Prediction & Trade Plan Rankings")
        st.dataframe(make_display_table(comparison_df, display_cols), use_container_width=True)

        csv = comparison_df[display_cols].to_csv().encode("utf-8")
        st.download_button(
            "Download prediction log CSV",
            data=csv,
            file_name=f"bull_score_predictions_{selected_date.strftime('%Y_%m_%d')}.csv",
            mime="text/csv",
        )

        st.subheader("Stacked Bull Score Breakdown")
        st.plotly_chart(make_stacked_bull_score_chart(comparison_df, breakdown_count), use_container_width=True)

        st.subheader("Close Price Comparison")
        close_comparison = pd.DataFrame(
            {
                ticker: stock_data_asof[ticker]["Close"]
                for ticker in top_bull_tickers
                if ticker in stock_data_asof
            }
        )

        if not close_comparison.empty:
            st.line_chart(close_comparison, use_container_width=True)

        st.subheader(f"{top_stock} Entry / Exit Plan")
        st.plotly_chart(make_trade_plan_chart(top_stock, stock_data_asof[top_stock], top_metrics, days=90), use_container_width=True)

        chart_tickers = st.sidebar.multiselect(
            "Candlestick charts shown",
            options=comparison_df.index.tolist(),
            default=top_bull_tickers,
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

            st.plotly_chart(make_candlestick_chart(ticker, intraday_df, candle_interval), use_container_width=True)

            with st.expander(f"{ticker} Volume Profile"):
                volume_profile = make_volume_profile_chart(ticker, intraday_df)
                if volume_profile:
                    st.plotly_chart(volume_profile, use_container_width=True)

        st.subheader(f"{top_stock} Recent News")

        top_headlines = headlines_by_ticker.get(top_stock, [])

        if top_headlines:
            for headline in top_headlines:
                st.write(f"- {headline}")
        else:
            st.info("No recent headlines found.")

        if show_news_debug:
            st.subheader("News Debug")
            st.write("News API key found:", bool(news_api_key))
            st.write("VADER installed:", SentimentIntensityAnalyzer is not None)
            st.write("TextBlob installed:", TextBlob is not None)

            debug_rows = []
            for ticker in comparison_df.index[:10]:
                heads = headlines_by_ticker.get(ticker, [])
                real_heads = real_headlines_only(heads)
                debug_rows.append(
                    {
                        "Ticker": ticker,
                        "Raw Headlines Returned": len(heads),
                        "Usable Headlines": len(real_heads),
                        "Sentiment": comparison_df.loc[ticker, "Sentiment Raw"],
                        "Example Headline": heads[0] if heads else "None",
                    }
                )

            st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)

    with right_col:
        st.subheader("Top 5 Today")
        st.caption("Click to view trade plan.")

        top_5 = comparison_df.head(5)

        medals = ["🥇", "🥈", "🥉", "4.", "5."]

        for idx, (ticker, row) in enumerate(top_5.iterrows()):
            label = medals[idx]

            with st.container(border=True):
                st.markdown(f"### {label} {ticker}")
                st.write(f"**Bull:** {row['Bull Score']:.1f}")
                st.write(f"**{row['Prediction']}**")

                if st.button("View", key=f"open_trade_plan_{ticker}"):
                    show_trade_plan_modal(ticker, comparison_df, stock_data_asof)

        if custom_ticker and custom_ticker in comparison_df.index:
            st.divider()
            st.subheader("Lookup")
            row = comparison_df.loc[custom_ticker]

            with st.container(border=True):
                st.markdown(f"### {custom_ticker}")
                st.write(f"**Bull:** {row['Bull Score']:.1f}")
                st.write(f"**{row['Prediction']}**")

                if st.button("View lookup", key=f"open_lookup_{custom_ticker}"):
                    show_trade_plan_modal(custom_ticker, comparison_df, stock_data_asof)

    st.markdown(
        "---\n"
        "**Disclaimer:** This dashboard is for informational and educational purposes only. "
        "It is not investment advice, and suggested entry/exit zones are model-generated estimates."
    )


if __name__ == "__main__":
    main()
