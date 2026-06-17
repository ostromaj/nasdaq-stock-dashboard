# NASDAQ Stock Dashboard

This repository contains a Streamlit application that ranks a selection of NASDAQ stocks based on daily price performance and basic news sentiment. The app demonstrates how to combine financial data with recent headlines to generate a simple ranking. It is intended for educational purposes only.

## Features

- Fetches historical price data for a curated list of NASDAQ tickers using the `yfinance` library.
- Calculates daily percentage returns and ranks stocks by their performance on a selected date.
- (Optional) Retrieves recent news headlines for each stock via the NewsAPI.org service and applies a naive sentiment analysis using `textblob`.
- Combines price performance and sentiment into a composite score used to rank the top 10 stocks.
- Interactive Streamlit interface with sliders to adjust the weighting between price performance and news sentiment.
- Charts showing recent price history for the top-ranked stocks.

## Setup

1. Clone the repository or download the files.
2. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Obtain an API key from [NewsAPI.org](https://newsapi.org/) and set it as an environment variable to enable news sentiment analysis:

   ```bash
   export NEWS_API_KEY=your_api_key_here
   ```

5. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Usage

When you run the app, you will see a page where you can:

- Select a date for analysis (within the past year).
- Adjust the weighting between price performance and news sentiment.
- Choose which of the predefined NASDAQ tickers to analyze.

The app will display the top 10 stocks based on the selected criteria, along with their daily returns, sentiment scores, and composite scores. A line chart for each top-ranked stock shows its recent price history.

## Disclaimer

> **Important:** This application is for demonstration and educational purposes only. It does **not** constitute financial advice. The rankings and sentiment scores are simplistic and should not be relied upon for real investment decisions. Always perform your own research or consult a qualified financial advisor.

## Contributing

Contributions are welcome! If you have ideas for improving the ranking algorithm, adding more data sources, or enhancing the user interface, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.