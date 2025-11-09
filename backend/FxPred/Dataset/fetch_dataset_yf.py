import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pytz

# use TwelveData
import os

# from twelvedata import TDClient
# from dotenv import load_dotenv
# load_dotenv()
#
# API_KEY = os.environ.get('TWELVEDATA_API_KEY')
# td = TDClient(apikey=API_KEY)
#
# def fetch_gold_price(interval, symbol, outputsize=5000):
#
#     ts = td.time_series(
#         symbol=symbol,
#         interval=interval,
#         outputsize=outputsize
#         # start_date=start_date,
#     )
#     df = ts.as_pandas()
#
#     df = df.reset_index()
#     date_col = 'date' if 'date' in df.columns else 'datetime'
#     df = df[[date_col, 'open', 'high', 'low', 'close']].dropna().copy()
#     df.columns = ['date', 'open', 'high', 'low', 'close']
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.sort_values('date', ascending=False).reset_index(drop=True)
#     return df

def fetch_gold_price(interval, start_date='2000-01-01', end_date=None, progress=False):
    try:
        tickers = ["GC=F", "IAU"]
        df = None
        finland_tz = pytz.timezone("Europe/Helsinki")
        now = datetime.now(tz=finland_tz)

        # Normalize end_date
        if end_date is None:
            end_date_dt = now
        elif isinstance(end_date, str):
            end_date_dt = finland_tz.localize(datetime.strptime(end_date, '%Y-%m-%d'))
        else:
            end_date_dt = end_date

        # Normalize start_date
        if isinstance(start_date, str):
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_date_dt = start_date

        # Set start_date based on interval
        if interval in ['15m', '30m']:
            start_date_dt = now - timedelta(days=60)  # yfinance limit for minute data
        elif interval in ['1h', '4h']:
            start_date_dt = now - timedelta(days=730)  # max 2 years for hourly
        elif interval in ['1d', '1wk']:
            # keep start_date as is
            start_date_dt = start_date_dt
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Convert start and end to UTC strings for yfinance
        start_str = start_date_dt.strftime('%Y-%m-%d')
        end_str = end_date_dt.strftime('%Y-%m-%d')

        # Try multiple tickers
        for ticker in tickers:
            try:
                print(f"Attempting to fetch data for ticker {ticker}...")
                if interval in ['15m', '30m']:
                    df = yf.download(ticker, period="5d", interval=interval, progress=progress)
                elif interval in ['1h', '4h', ]:
                    df = yf.download(ticker, period="30d", interval=interval, progress=progress)
                else:  # daily/weekly/monthly
                    df = yf.download(ticker, start=start_str, end=end_str, interval=interval, progress=progress)

                if df.empty:
                    continue

                # Convert index to Finland local time
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC').tz_convert(finland_tz)
                else:
                    df.index = df.index.tz_convert(finland_tz)

                print(f"Successfully fetched data from ticker {ticker}")
                break
            except Exception as e:
                print(f"Failed to fetch data for ticker {ticker}: {e}")
                continue
        else:
            raise ValueError("Failed to fetch data for all tickers.")

        # Reset index and clean dataframe
        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else 'Datetime'
        df = df[[date_col, 'Open', 'High', 'Low', 'Close']].dropna().copy()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        print(f"✓ Fetched {len(df)} rows of data from {df['Date'].min()} to {df['Date'].max()}")
        return df

    except Exception as e:
        print(f"Error initializing ticker: {e}")
        return None


def add_technical_indicators(df, extended=True):
    """Add technical indicators to the dataframe"""
    df = df.copy()

    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)

    # Momentum indicators
    df['Momentum'] = df['Close'].diff(periods=10)
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    df['ROC'] = df['Close'].pct_change(periods=10)

    if extended:
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-10))
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

        # Commodity Channel Index (CCI)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std() + 1e-10)

        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14 + 1e-10))

        # Price Change
        df['Price_Change'] = df['Close'].diff()

        # ADX (Average Directional Index)
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = true_range
        atr = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['ADX'] = dx.rolling(window=14).mean()

        # Rate of Change Ratio
        df['ROCR'] = df['Close'] / df['Close'].shift(10)

        # Price Channels
        df['Upper_Channel'] = df['High'].rolling(window=20).max()
        df['Lower_Channel'] = df['Low'].rolling(window=20).min()
        df['Channel_Position'] = (df['Close'] - df['Lower_Channel']) / (
                    df['Upper_Channel'] - df['Lower_Channel'] + 1e-10)

    # Remove infinite values and NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return df


def get_feature_columns(extended=True):
    """Get list of feature columns"""
    base_cols = ['Open', 'High', 'Low', 'Close',
                 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
                 'EMA_12', 'EMA_26', 'MACD', 'Signal', 'MACD_Hist', 'RSI',
                 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
                 'Momentum', 'Volatility', 'ROC', 'BB_Position']

    if extended:
        extended_cols = ['ATR', 'Stochastic_K', 'Stochastic_D', 'CCI',
                         'Williams_R', 'Price_Change', 'ADX', 'ROCR',
                         'Upper_Channel', 'Lower_Channel', 'Channel_Position']
        return base_cols + extended_cols

    return base_cols


def prepare_data(df, sequence_length=60, use_indicators=True, extended=True):
    """Prepare sequences for training"""
    if use_indicators:
        print("\nAdding technical indicators...")
        df = add_technical_indicators(df, extended=extended)

    feature_cols = get_feature_columns(extended=extended)

    data = df[feature_cols].values
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    # Create separate scalers
    scaler_features = MinMaxScaler()
    scaler_opens = MinMaxScaler()
    scaler_highs = MinMaxScaler()
    scaler_lows = MinMaxScaler()
    scaler_closes = MinMaxScaler()

    data_scaled = scaler_features.fit_transform(data)
    opens_scaled = scaler_opens.fit_transform(opens.reshape(-1, 1)).flatten()
    highs_scaled = scaler_highs.fit_transform(highs.reshape(-1, 1)).flatten()
    lows_scaled = scaler_lows.fit_transform(lows.reshape(-1, 1)).flatten()
    closes_scaled = scaler_closes.fit_transform(closes.reshape(-1, 1)).flatten()

    # Create sequences
    sequences = []
    targets = []
    for i in range(len(data_scaled) - sequence_length):
        seq = data_scaled[i:i + sequence_length]
        target = np.array([
            opens_scaled[i + sequence_length],
            highs_scaled[i + sequence_length],
            lows_scaled[i + sequence_length],
            closes_scaled[i + sequence_length]
        ])
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(f"✓ Prepared {len(sequences)} sequences of length {sequence_length}")
    print(f"✓ Feature dimensions: {len(feature_cols)}")

    scalers_ohlc = {
        'open': scaler_opens,
        'high': scaler_highs,
        'low': scaler_lows,
        'close': scaler_closes
    }

    ohlc_data = {
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    }

    return sequences, targets, scaler_features, scalers_ohlc, feature_cols, ohlc_data