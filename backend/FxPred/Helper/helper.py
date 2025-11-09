import pandas as pd

def get_interval_info(interval):
    """Get information about time interval"""
    interval_map = {
        '15m': {'minutes': 15, 'name': '15 Minutes', 'trading_hours': 24},
        '30m': {'minutes': 30, 'name': '30 Minutes', 'trading_hours': 24},
        '1h': {'minutes': 60, 'name': '1 Hour', 'trading_hours': 24},
        '4h': {'minutes': 240, 'name': '4 Hours', 'trading_hours': 24},
        '1d': {'minutes': 1440, 'name': '1 Day', 'trading_hours': 24},
        '1wk': {'minutes': 10080, 'name': '1 Week', 'trading_hours': 24}
    }
    return interval_map.get(interval, {'minutes': 1440, 'name': 'Daily', 'trading_hours': 24})

def advance_datetime(dt, interval, skip_weekends=True):
    """Advance datetime by one interval, optionally skipping weekends"""
    info = get_interval_info(interval)

    if interval in ['1d', '1wk']:
        # For daily and above, skip weekends
        if interval == '1wk':
            next_dt = dt + pd.Timedelta(days=7)
        else:
            next_dt = dt + pd.Timedelta(days=1)

        if skip_weekends and interval in ['1d']:
            while next_dt.weekday() >= 5:  # Saturday=5, Sunday=6
                next_dt += pd.Timedelta(days=1)
        return next_dt
    else:
        # For intraday, just add the minutes (gold trades 24/7)
        return dt + pd.Timedelta(minutes=info['minutes'])

def format_datetime_for_interval(dt, interval):
    """Format datetime appropriately for the interval"""
    if interval in ['15m', '30m', '1h', '4h']:
        return dt.strftime('%Y-%m-%d %H:%M')
    elif interval in ['1d']:
        return dt.strftime('%Y-%m-%d')
    elif interval in ['1wk']:
        return dt.strftime('%Y-W%U')
    else:
        return dt.strftime('%Y-%m-%d')