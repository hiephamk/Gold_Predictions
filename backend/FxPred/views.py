from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.conf import settings
from rest_framework import status

from .save_gold_prices import *
from .Dataset.fetch_dataset_yf import (fetch_gold_price, add_technical_indicators, get_feature_columns)
from .Helper.helper import (get_interval_info, advance_datetime, format_datetime_for_interval)
from .gold_training import (ForexTransformer)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

from .serializers import *
from .models import *
from django.utils.timezone import make_aware
import pytz


@api_view(['GET'])
@permission_classes([AllowAny])
def PredictedData_15m_View(request):
    queryset = PredictedData_15m.objects.all()
    serializer = PredictedData_15m_Serializer(queryset, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([AllowAny])
def PredictedData_30m_View(request):
    queryset = PredictedData_30m.objects.all()
    serializer = PredictedData_30m_Serializer(queryset, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([AllowAny])
def PredictedData_1h_View(request):
    queryset = PredictedData_1h.objects.all()
    serializer = PredictedData_1h_Serializer(queryset, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([AllowAny])
def PredictedData_4h_View(request):
    queryset = PredictedData_4h.objects.all()
    serializer = PredictedData_4h_Serializer(queryset, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([AllowAny])
def PredictedData_1d_View(request):
    queryset = PredictedData_1d.objects.all()
    serializer = PredictedData_1d_Serializer(queryset, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([AllowAny])
def PredictedData_1wk_View(request):
    queryset = PredictedData_1wk.objects.all()
    serializer = PredictedData_1wk_Serializer(queryset, many=True)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([AllowAny])
def predict_future_prices(request):
    import os
    from pathlib import Path
    try:
        steps_ahead = 5
        sequence_length = 60
        extended = True
        interval = request.data.get('interval', '1h')

        # Map interval to corresponding model
        interval_model_map = {
            '15m': PredictedData_15m,
            '30m': PredictedData_30m,
            '1h': PredictedData_1h,
            '4h': PredictedData_4h,
            '1d': PredictedData_1d,
            '1wk': PredictedData_1wk,
        }

        # Get the appropriate model class
        if interval not in interval_model_map:
            return Response({
                "error": f"Invalid interval: {interval}. Must be one of: {', '.join(interval_model_map.keys())}"
            }, status=400)

        PredictedDataModel = interval_model_map[interval]

        BASE_DIR = Path(__file__).resolve().parent.parent  # → /app
        MODEL_DIR = BASE_DIR / "FxPred"  # → /app/FxPred

        # Option 2: Hardcode for Docker (simplest & safest)
        # MODEL_DIR = Path("/app/FxPred")

        model_path = MODEL_DIR / f"xauusd_model_best_{interval}.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        df = fetch_gold_price(interval)

        if df is None or df.empty:
            return Response({"error": "Failed to fetch data"}, status=400)

        df = add_technical_indicators(df, extended=extended)
        feature_cols = get_feature_columns(extended=extended)

        if not all(col in df.columns for col in feature_cols):
            missing = [col for col in feature_cols if col not in df.columns]
            return Response({"error": f"Missing columns: {missing}"}, status=400)

        scaler_features = MinMaxScaler()
        scalers_ohlc = {
            'open': MinMaxScaler(),
            'high': MinMaxScaler(),
            'low': MinMaxScaler(),
            'close': MinMaxScaler()
        }
        scaler_features.fit(df[feature_cols].values)
        # Output of predictions
        scalers_ohlc['open'].fit(df['Open'].values.reshape(-1, 1))
        scalers_ohlc['high'].fit(df['High'].values.reshape(-1, 1))
        scalers_ohlc['low'].fit(df['Low'].values.reshape(-1, 1))
        scalers_ohlc['close'].fit(df['Close'].values.reshape(-1, 1))

        data_scaled = scaler_features.transform(df[feature_cols].values)

        if len(data_scaled) < sequence_length:
            return Response({
                "error": f"Not enough data. Need at least {sequence_length} rows"
            }, status=400)

        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')

        model = ForexTransformer(input_dim=len(feature_cols), output_dim=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        current_sequence = data_scaled[-sequence_length:].copy()
        last_close = float(df['Close'].iloc[-1])
        last_date = pd.to_datetime(df['Date'].iloc[-1])

        future_ohlc = {'open': [], 'high': [], 'low': [], 'close': []}
        future_dates = []
        skip_weekends = interval in ['1d', '1wk']
        interval_info = get_interval_info(interval)

        with torch.no_grad():
            for _ in range(steps_ahead):
                seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
                pred_scaled = model(seq_tensor).cpu().numpy()[0]

                pred_open = scalers_ohlc['open'].inverse_transform([[pred_scaled[0]]])[0, 0]
                pred_high = scalers_ohlc['high'].inverse_transform([[pred_scaled[1]]])[0, 0]
                pred_low = scalers_ohlc['low'].inverse_transform([[pred_scaled[2]]])[0, 0]
                pred_close = scalers_ohlc['close'].inverse_transform([[pred_scaled[3]]])[0, 0]

                future_ohlc['open'].append(pred_open)
                future_ohlc['high'].append(pred_high)
                future_ohlc['low'].append(pred_low)
                future_ohlc['close'].append(pred_close)

                next_date = advance_datetime(last_date, interval, skip_weekends=skip_weekends)
                future_dates.append(next_date)
                last_date = next_date

                # Update sequence with all OHLC values
                new_row = current_sequence[-1].copy()
                new_row[feature_cols.index('Close')] = pred_scaled[3]
                if 'Open' in feature_cols:
                    new_row[feature_cols.index('Open')] = pred_scaled[0]
                if 'High' in feature_cols:
                    new_row[feature_cols.index('High')] = pred_scaled[1]
                if 'Low' in feature_cols:
                    new_row[feature_cols.index('Low')] = pred_scaled[2]

                current_sequence = np.vstack([current_sequence[1:], new_row])

        # Calculate additional metrics
        price_ranges = [h - l for h, l in zip(future_ohlc['high'], future_ohlc['low'])]
        price_changes = [c - last_close for c in future_ohlc['close']]

        # Calculate confidence scores based on Close prices
        confidence_scores = []
        for change in price_changes:
            confidence = max(0, 1 - abs(change) / last_close * 0.5) * 100
            confidence_scores.append(confidence)

        # Build results with OHLC data
        results = {
            'predictions': [
                {
                    'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    'formatted': format_datetime_for_interval(pd.to_datetime(date), interval),
                    'open': round(float(o), 2),
                    'high': round(float(h), 2),
                    'low': round(float(l), 2),
                    'close': round(float(c), 2),
                    'price_range': round(float(r), 2),
                    'change': round(float(ch), 2),
                    'confidence': round(float(conf), 2)
                }
                for date, o, h, l, c, r, ch, conf in zip(
                    future_dates,
                    future_ohlc['open'],
                    future_ohlc['high'],
                    future_ohlc['low'],
                    future_ohlc['close'],
                    price_ranges,
                    price_changes,
                    confidence_scores
                )
            ],
            'summary': {
                'last_close': round(float(df['Close'].iloc[-1]), 2),
                'last_open': round(float(df['Open'].iloc[-1]), 2),
                'last_high': round(float(df['High'].iloc[-1]), 2),
                'last_low': round(float(df['Low'].iloc[-1]), 2),
                'avg_predicted_close': round(float(np.mean(future_ohlc['close'])), 2),
                'high_predicted': round(float(np.max(future_ohlc['high'])), 2),
                'low_predicted': round(float(np.min(future_ohlc['low'])), 2),
                'avg_range': round(float(np.mean(price_ranges)), 2),
                'interval': interval_info['name']
            },
            'message': 'OHLC prediction successful!'
        }

        # Save to database with proper timezone handling
        # Clear old predictions for this interval
        # PredictedDataModel.objects.all().delete()

        for item in results['predictions']:
            # Parse the date - yfinance returns UTC timestamps
            ts = pd.to_datetime(item['date'])

            # Ensure timezone aware datetime in UTC
            if ts.tzinfo is None:
                # yfinance data should be UTC, so localize as UTC
                ts_aware = make_aware(ts, timezone=pytz.UTC)
            else:
                # Convert to UTC if it has a different timezone
                ts_utc = ts.astimezone(pytz.UTC)
                ts_aware = make_aware(ts_utc.replace(tzinfo=None), timezone=pytz.UTC)

            # Save to the appropriate model based on interval
            PredictedDataModel.objects.update_or_create(
                date=ts_aware,
                defaults={
                    "open": round(item["open"], 2),
                    "high": round(item["high"], 2),
                    "low": round(item["low"], 2),
                    "close": round(item["close"], 2),
                    "price_range": round(item["price_range"], 2),
                    "change": round(item["change"], 2),
                    "confidence": round(item["confidence"], 2),
                },
            )

        return Response(results, status=status.HTTP_200_OK)

    except FileNotFoundError:
        return Response({
            'error': f'Model file not found for interval {interval}. Please train the model first.'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        import traceback
        print("Error:", traceback.format_exc())
        return Response({
            'error': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# @api_view(['POST'])
# @permission_classes([AllowAny])
# def fetch_actual_data(request):
#     from datetime import datetime, timedelta
#     import pytz
#     import yfinance as yf
#     import pandas as pd
#     from rest_framework.response import Response
#     from rest_framework import status
#
#     try:
#         tickers = ["GOLD", "GC=F", "IAU"]
#         interval = request.data.get('interval', '1h')
#         progress = False
#
#         # === 1. Get current time in Finland (for logic only) ===
#         finland_tz = pytz.timezone("Europe/Helsinki")
#         now_local = datetime.now(tz=finland_tz)
#
#         # === 2. Set start_date based on interval (in local time) ===
#         if interval in ['1m', '5m', '15m', '30m']:
#             start_date_local = now_local - timedelta(days=60)
#         elif interval in ['1h', '4h']:
#             start_date_local = now_local - timedelta(days=730)
#         elif interval in ['1d', '1wk']:
#             start_date_local = datetime(2000, 1, 1)  # far back
#         else:
#             raise ValueError(f"Unsupported interval: {interval}")
#
#         # === 3. Convert start/end to UTC for yfinance ===
#         start_date_utc = start_date_local.astimezone(pytz.UTC)
#         end_date_utc = now_local.astimezone(pytz.UTC)
#
#         start_str = start_date_utc.strftime('%Y-%m-%d')
#         end_str = end_date_utc.strftime('%Y-%m-%d')
#
#         df = None
#         for ticker in tickers:
#             try:
#                 print(f"Fetching {ticker} @ {interval} from {start_str} to {end_str} (UTC)...")
#
#                 if interval in ['1m', '5m', '15m', '30m']:
#                     df = yf.download(ticker, period="5d", interval=interval, progress=progress)
#                 elif interval in ['1h', '4h']:
#                     df = yf.download(ticker, period="30d", interval=interval, progress=progress)
#                 else:
#                     df = yf.download(ticker, start=start_str, end=end_str, interval=interval, progress=progress)
#
#                 if not df.empty:
#                     print(f"Success with {ticker}")
#                     break
#             except Exception as e:
#                 print(f"{ticker} failed: {e}")
#                 continue
#         else:
#             raise ValueError("No data from any ticker")
#
#         # === 4. Ensure index is timezone-aware UTC ===
#         if df.index.tz is None:
#             df.index = df.index.tz_localize('UTC')
#         else:
#             df.index = df.index.tz_convert('UTC')  # Force UTC
#
#         # === 5. Reset index and select columns ===
#         df = df.reset_index()
#         date_col = 'Date' if 'Date' in df.columns else 'Datetime'
#         df = df[[date_col, 'Open', 'High', 'Low', 'Close']].dropna()
#         df = df.rename(columns={date_col: 'Date'})
#
#         # === 6. Convert Date to UTC string ===
#         df['Date'] = pd.to_datetime(df['Date,Date'])
#
#         records = []
#         for _, row in df.iterrows():
#             utc_dt = row['Date']
#             # Format as UTC string
#             utc_str = utc_dt.strftime('%Y-%m-%d %H:%M:%S')
#             records.append({
#                 'Date': utc_str,  # UTC time
#                 'Open': round(float(row['Open']), 2),
#                 'High': round(float(row['High']), 2),
#                 'Low': round(float(row['Low']), 2),
#                 'Close': round(float(row['Close']), 2),
#             })
#
#         return Response({
#             'data': records,
#             'timezone': 'UTC',
#             'interval': interval,
#         }, status=status.HTTP_200_OK)
#
#     except Exception as e:
#         import traceback
#         print("ERROR:", traceback.format_exc())
#         return Response(
#             {'error': str(e)},
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )



@api_view(['POST'])
@permission_classes([AllowAny])
def fetch_actual_data(request):
    from datetime import datetime, timedelta
    try:
        tickers = ["GC=F", "IAU"]
        interval = request.data.get('interval', '1h')
        start_date = '2000-01-01'
        end_date = None
        progress = False

        # df = None
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
        if interval in ['1m', '5m', '15m', '30m']:
            start_date_dt = now - timedelta(days=60)  # yfinance limit for minute data
        elif interval in ['1h', '4h']:
            start_date_dt = now - timedelta(days=730)  # max 2 years for hourly
        elif interval in ['1d', '1wk', '1mo']:
            # keep start_date as is
            start_date_dt = start_date_dt
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Convert start and end to UTC strings for yfinance
        start_str = start_date_dt.strftime('%Y-%m-%d')
        end_str = end_date_dt.strftime('%Y-%m-%d')


        for ticker in tickers:
            try:
                print(f"Attempting to fetch data for ticker {ticker}...")
                if interval in ['1m', '5m', '15m', '30m']:
                    df = yf.download(ticker, period="5d", interval=interval, progress=progress)
                elif interval in ['1h', '4h',]:
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
        df = df.sort_values('Date', ascending=False).reset_index(drop=True)

        # Flatten multi-index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Convert to list of dictionaries
        records = []
        for _, row in df.iterrows():
            records.append({
                'Date': row['Date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['Date'], 'strftime') else str(
                    row['Date']),
                # 'Date': row['Date'],
                'Open': round(float(row['Open']), 2),
                'High': round(float(row['High']), 2),
                'Low': round(float(row['Low']), 2),
                'Close': round(float(row['Close']), 2),
                # 'Volume': int(row['Volume'])
            })

        return Response({
            # 'ticker': ticker,
            # 'period': period,
            # 'interval': interval,
            'data': records
        }, status=status.HTTP_200_OK)

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Debug: print full error
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )