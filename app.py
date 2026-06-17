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

    # Force a real datetime index.
    df.index = pd.to_datetime(df.index, errors="coerce")

    # Remove bad index rows.
    df = df[~df.index.isna()]

    if df.empty:
        return pd.DataFrame()

    # Remove timezone safely.
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    # If Yahoo returns a RangeIndex or anything weird, reject it.
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    return df.sort_index()
