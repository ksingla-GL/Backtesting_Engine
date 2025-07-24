"""
data_handler.py
Module for loading, merging, and preparing all financial data for backtesting.
This version uses a robust one-by-one loading approach and correctly handles 
the alignment of sparse daily/weekly indicators to the 1-minute timeframe.
"""
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

def connect_db(db_path):
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        logging.info(f"Successfully connected to database at {db_path}")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database: {e}")
        raise

def load_data_from_db(conn, table_name, **kwargs):
    """Loads data from a specified table in the database."""
    symbol = kwargs.get('symbol')
    indicator_name = kwargs.get('indicator_name')
    
    if table_name == 'price_data':
        query = """
        SELECT p.datetime, p.open, p.high, p.low, p.close, p.volume 
        FROM price_data p 
        JOIN symbol s ON p.symbol_id = s.symbol_id 
        WHERE s.ticker = ? AND p.timeframe = '1min'
        ORDER BY p.datetime
        """
        params = (symbol,)
    elif table_name == 'market_indicator':
        query = "SELECT datetime, value FROM market_indicator WHERE indicator_name = ? ORDER BY datetime"
        params = (indicator_name,)
    else:
        raise ValueError("Invalid table name specified.")

    df = pd.read_sql_query(query, conn, params=params)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Log data info
    logging.info(f"  Loaded {len(df)} records for {symbol or indicator_name}")
    if len(df) > 0:
        logging.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df

def create_market_hours_index(start_date, end_date):
    """Creates a 1-minute DatetimeIndex for US market hours."""
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    market_minutes = []
    for date in all_dates:
        day_start = pd.Timestamp.combine(date, MARKET_OPEN)
        day_end = pd.Timestamp.combine(date, MARKET_CLOSE)
        market_minutes.append(pd.date_range(start=day_start, end=day_end, freq='1min'))
    
    full_index = pd.DatetimeIndex(np.concatenate(market_minutes))
    logging.info(f"Created market hours index with {len(full_index)} timestamps")
    return full_index

def resample_to_1min(df, prefix=''):
    """Resamples OHLCV data to 1-minute frequency."""
    df = df.set_index('datetime')
    if df.empty:
        return pd.DataFrame()
    
    # Check if data is already 1-minute frequency
    time_diffs = df.index.to_series().diff().dropna()
    if len(time_diffs) > 0:
        median_diff = time_diffs.median()
        if median_diff == pd.Timedelta(minutes=1):
            logging.info(f"  Data for {prefix} is already at 1-minute frequency")
            resampled = df
        else:
            logging.info(f"  Resampling {prefix} data from {median_diff} to 1-minute frequency")
            resampled = df.resample('1min').agg({
                'open': 'first', 
                'high': 'max', 
                'low': 'min', 
                'close': 'last', 
                'volume': 'sum'
            })
    else:
        resampled = df
    
    # Rename columns with prefix
    resampled.columns = [f"{prefix}_{col}" for col in resampled.columns]
    
    # Log column info
    non_nan_counts = {col: resampled[col].notna().sum() for col in resampled.columns}
    logging.info(f"  Columns created: {list(resampled.columns)}")
    logging.info(f"  Non-NaN values per column: {non_nan_counts}")
    
    return resampled

def validate_data_quality(df, instrument_name):
    """Validate data quality and log any issues."""
    issues = []
    
    # Check for missing OHLC columns
    expected_cols = [f'{instrument_name}_open', f'{instrument_name}_high', 
                    f'{instrument_name}_low', f'{instrument_name}_close']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for NaN values in critical columns
    for col in expected_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            nan_pct = (nan_count / len(df)) * 100
            if nan_pct > 50:
                issues.append(f"{col} has {nan_pct:.1f}% NaN values")
    
    # Check if high >= low
    high_col = f'{instrument_name}_high'
    low_col = f'{instrument_name}_low'
    if high_col in df.columns and low_col in df.columns:
        invalid_rows = df[df[high_col] < df[low_col]]
        if len(invalid_rows) > 0:
            issues.append(f"{len(invalid_rows)} rows where high < low")
    
    if issues:
        logging.warning(f"Data quality issues for {instrument_name}:")
        for issue in issues:
            logging.warning(f"  - {issue}")
    else:
        logging.info(f"  Data quality check passed for {instrument_name}")

def get_merged_data(global_config):
    """
    Main data preparation function.
    """
    settings = global_config['settings']
    db_path = settings['db_path']
    start_date = settings['start_date']
    end_date = settings['end_date']
    
    conn = connect_db(db_path)
    
    logging.info("Creating master 1-minute time index...")
    master_index = create_market_hours_index(start_date, end_date)
    master_df = pd.DataFrame(index=master_index)
    
    # 1. Load Price Data
    instruments = ['ES', 'SPX', 'VIX', 'VX', 'TRIN']
    for inst in instruments:
        logging.info(f"Loading and processing {inst} data...")
        try:
            price_df = load_data_from_db(conn, 'price_data', symbol=inst)
            if price_df.empty:
                logging.warning(f"No data found for {inst}")
                continue
                
            resampled_df = resample_to_1min(price_df, prefix=inst)
            
            # Join with master_df
            before_join = len(master_df)
            master_df = master_df.join(resampled_df, how='left')
            
            # Validate the join
            validate_data_quality(master_df, inst)
            
        except Exception as e:
            logging.error(f"Could not process data for {inst}: {e}")
            import traceback
            logging.error(traceback.format_exc())

    # 2. Load Market Indicators
    indicators = ['VIX_VXV_RATIO', 'MARKET_BREADTH', 'NAAIM', 'FED_STANCE', 'CNN_FEAR_GREED', 'BUFFETT_INDICATOR']
    for ind_name in indicators:
        logging.info(f"Loading and processing {ind_name} data...")
        try:
            ind_df = load_data_from_db(conn, 'market_indicator', indicator_name=ind_name)
            if ind_df.empty:
                logging.warning(f"No data found for {ind_name}, filling with NaN")
                master_df[ind_name] = np.nan
                continue

            ind_df = ind_df.set_index('datetime')
            ind_df = ind_df[~ind_df.index.duplicated(keep='last')]
            
            if ind_name == 'FED_STANCE':
                stance_map = {'Dovish': 1.0, 'Neutral': 2.0, 'Hawkish': 3.0}
                ind_df['value'] = pd.to_numeric(ind_df['value'], errors='coerce').fillna(ind_df['value'].map(stance_map))
            else:
                ind_df['value'] = pd.to_numeric(ind_df['value'], errors='coerce')
            
            # Reindex and forward-fill
            master_df[ind_name] = ind_df['value'].reindex(master_df.index, method='ffill')
            
            # Log indicator stats
            non_nan = master_df[ind_name].notna().sum()
            logging.info(f"  {ind_name} has {non_nan} non-NaN values ({non_nan/len(master_df)*100:.1f}%)")

        except Exception as e:
            logging.error(f"Could not process indicator {ind_name}: {e}")
            master_df[ind_name] = np.nan

    # 3. Perform comprehensive data validation before forward fill
    logging.info("\nData summary before forward fill:")
    for col in master_df.columns:
        non_nan = master_df[col].notna().sum()
        nan_pct = (1 - non_nan/len(master_df)) * 100
        logging.info(f"  {col}: {non_nan:,} non-NaN ({nan_pct:.1f}% NaN)")
    
    # 4. Forward fill with limit to avoid propagating stale data too far
    logging.info("\nPerforming final forward-fill...")
    # First forward fill price data with a reasonable limit (e.g., 390 minutes = 1 trading day)
    price_columns = [col for inst in instruments for col in [f'{inst}_open', f'{inst}_high', f'{inst}_low', f'{inst}_close', f'{inst}_volume'] if col in master_df.columns]
    master_df[price_columns] = master_df[price_columns].ffill(limit=390)
    
    # Then forward fill indicators with no limit (they update less frequently)
    indicator_columns = [col for col in master_df.columns if col not in price_columns]
    master_df[indicator_columns] = master_df[indicator_columns].ffill()
    
    # 5. Final validation
    logging.info("\nFinal data summary after forward fill:")
    critical_issues = []
    for col in master_df.columns:
        non_nan = master_df[col].notna().sum()
        nan_pct = (1 - non_nan/len(master_df)) * 100
        logging.info(f"  {col}: {non_nan:,} non-NaN ({nan_pct:.1f}% NaN)")
        
        # Flag critical issues
        if col.startswith('ES_') and nan_pct > 10:
            critical_issues.append(f"{col} has {nan_pct:.1f}% NaN values")
    
    if critical_issues:
        logging.error("CRITICAL DATA ISSUES DETECTED:")
        for issue in critical_issues:
            logging.error(f"  - {issue}")
    
    conn.close()
    logging.info("Data preparation complete.")
    return master_df