import os
import requests
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()

client = Client(
    api_key=os.getenv('BINANCE_API_KEY'),
    api_secret=os.getenv('BINANCE_API_SECRET'),
    tld='com'
)

def get_historical_klines(symbol, interval, start_time, end_time=None, limit=1500):
    """
    Fetch historical candle data from Binance
    Parameters:
    - symbol: Trading pair (e.g., 'BTCUSDT')
    - interval: Candle interval (e.g., '1h', '4h', '1d')
    - start_time: Start datetime (datetime object or ISO string)
    - end_time: End datetime (default: now)
    - limit: Maximum number of candles to retrieve (max 1500)
    """

    # Convert datetime objects to formatted strings
    if isinstance(start_time, datetime):
        start_str = start_time.strftime('%d %b, %Y')
    else:
        start_str = start_time

    end_str = end_time.strftime('%d %b, %Y') if isinstance(end_time, datetime) else end_time

    try:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            limit=limit
        )
        return parse_klines(klines, symbol)  

    except Exception as e:
        print(f"Error fetching historical klines: {e}")
        return None
    
def get_order_book(symbol, depth=10):
    """
    Fetches order book data from Binance for a given symbol.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTCUSDT')
        depth (int): Number of order book levels to retrieve (default: 10)

    Returns:
        dict: Processed order book metrics including best bid/ask, spread, and order book imbalance.
    """

    try:
      url = f"{os.getenv('BINANCE_BASE_URL')}/api/v3/depth"
      params = {"symbol": symbol.upper(), "limit": depth}
      response = requests.get(url, params=params)
      order_book = response.json()

      # extract best bid and ask
      best_bid_price = float(order_book['bids'][0][0])
      best_bid_volume = float(order_book['bids'][0][1])
      best_ask_price = float(order_book['asks'][0][0])
      best_ask_volume = float(order_book['asks'][0][1])

      # calculate bid and ask spread
      bid_ask_spread = best_ask_price - best_bid_price
      
      # calculate order book imbalance
      order_book_imbalance = (best_bid_volume - best_ask_volume) / (best_bid_volume + best_ask_volume)

      # calculate total bid and ask volume at depth
      bid_volume_depth = sum(float(bid[1]) for bid in order_book['bids'])
      ask_volume_depth = sum(float(bid[1]) for bid in order_book['asks'])

      return {
          "best_bid_price": best_bid_price,
          "best_ask_price": best_ask_price,
          "best_bid_volume": best_bid_volume,
          "best_ask_volume": best_ask_volume,
          "bid_ask_spread": bid_ask_spread,
          "order_book_imbalance": order_book_imbalance,
          "bid_volume_depth": bid_volume_depth,
          "ask_volume_depth": ask_volume_depth
      }
    
    except Exception as e:
        print(f"Error fetching order book: {e}")
        return None
    
def parse_klines(klines, symbol):
    """
    Convert raw Kline data to a structured format and compute technical indicators.

    This function processes Binance Kline (candlestick) data and calculates key 
    technical indicators for time-series price analysis.

    **Technical Indicators Explained:**
    - **SMA (Simple Moving Average)**:
      - `sma_10`, `sma_50`: The average closing price over the last 10 and 50 periods.
      - Helps identify **general trends** and smooths out short-term fluctuations.

    - **EMA (Exponential Moving Average)**:
      - `ema_10`, `ema_50`: A weighted moving average that gives more importance to recent prices.
      - More responsive to price changes compared to SMA.

    - **RSI (Relative Strength Index)**:
      - `rsi_14`: Measures the magnitude of recent price changes on a scale of 0-100.
      - **Above 70** suggests overbought conditions, **below 30** suggests oversold conditions.

    - **Bollinger Bands**:
      - `bollinger_upper`, `bollinger_middle`, `bollinger_lower`: A volatility indicator based on a moving average.
      - The upper and lower bands represent **2 standard deviations** from the middle band (SMA 20).
      - Price touching the upper band may indicate overbought conditions, and the lower band may indicate oversold conditions.

    - **MACD (Moving Average Convergence Divergence)**:
      - `macd`: The difference between the 12-period EMA and the 26-period EMA.
      - `macd_signal`: A 9-period EMA of the MACD.
      - A bullish signal occurs when the MACD crosses **above** the signal line, and a bearish signal occurs when it crosses **below**.

    - **ATR (Average True Range)**:
      - `atr`: Measures market volatility by calculating the average range between high and low prices over 14 periods.
      - A higher ATR indicates **higher volatility**.

    - **OBV (On-Balance Volume)**:
      - `obv`: A momentum indicator that tracks cumulative buying and selling pressure using volume.
      - If price moves up and volume increases, OBV rises, indicating a **strong trend**.
    """

    # Ensure only the first 11 columns are processed (some responses may have extra columns)
    df = pd.DataFrame([k[:11] for k in klines], columns=[
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"
    ])

    # Convert necessary columns to float
    df[["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]] = \
        df[["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]].astype(float)

    # Compute technical indicators
    df["sma_10"] = ta.trend.sma_indicator(df["close"], window=10)
    df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["ema_10"] = ta.trend.ema_indicator(df["close"], window=10)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

    bollinger = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bollinger_upper"] = bollinger.bollinger_hband()
    df["bollinger_middle"] = bollinger.bollinger_mavg()
    df["bollinger_lower"] = bollinger.bollinger_lband()

    df["macd"] = ta.trend.macd(df["close"])
    df["macd_signal"] = ta.trend.macd_signal(df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

    order_book_data = get_order_book(symbol)

    for key, value in order_book_data.items():
        df[key] = value

    # Drop NaN values to remove incomplete indicator calculations
    # TODO: UNCOMMENT IF NECESSARY!!!
    df.dropna(inplace=True)

    # Convert DataFrame to list of dictionaries
    return df.to_dict(orient="records")