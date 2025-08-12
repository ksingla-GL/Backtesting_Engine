"""
data_handler.py
Module for loading, merging, and preparing all financial data for backtesting.
Now uses pre-computed macro data for faster loading.
"""
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MARKET_OPEN = time(8, 0)
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

def get_date_range(conn, start_date=None, end_date=None):
    """Get actual date range from database if not specified."""
    if start_date and end_date:
        return start_date, end_date
    
    query = """
    SELECT MIN(datetime) as min_date, MAX(datetime) as max_date 
    FROM price_data 
    WHERE symbol_id = (SELECT symbol_id FROM symbol WHERE ticker = 'ES')
    """
    result = pd.read_sql_query(query, conn)
    
    db_start = result['min_date'].iloc[0] if not start_date else start_date
    db_end = result['max_date'].iloc[0] if not end_date else end_date
    
    logging.info(f"Using date range: {db_start} to {db_end}")
    return db_start, db_end

def extract_required_instruments(required_columns):
    """Extract instrument names from column list."""
    instruments = set()
    for col in required_columns:
        if '_' in col:
            prefix = col.split('_')[0]
            if prefix.upper() in ['ES', 'SPX', 'VIX', 'VX', 'TRIN', 'SPY']:
                instruments.add(prefix.upper())
    # Always include ES as primary instrument
    instruments.add('ES')
    return list(instruments)

def extract_required_indicators(required_columns):
    """Extract indicator names from column list."""
    # Standard indicators that map directly
    indicator_mapping = {
        'VIX_VXV_RATIO': 'VIX_VXV_RATIO',
        'MARKET_BREADTH': 'MARKET_BREADTH',
        'NAAIM': 'NAAIM',
        'FED_STANCE': 'FED_STANCE',
        'CNN_FEAR_GREED': 'CNN_FEAR_GREED',
        'BUFFETT_INDICATOR': 'BUFFETT_INDICATOR'
    }
    
    indicators = []
    for col in required_columns:
        if col in indicator_mapping:
            indicators.append(indicator_mapping[col])
    
    return indicators

def load_data_from_db(conn, table_name, start_date, end_date, **kwargs):
    """Loads data from a specified table in the database."""
    symbol = kwargs.get('symbol')
    indicator_name = kwargs.get('indicator_name')
    
    if table_name == 'price_data':
        query = """
        SELECT p.datetime, p.open, p.high, p.low, p.close, p.volume 
        FROM price_data p 
        JOIN symbol s ON p.symbol_id = s.symbol_id 
        WHERE s.ticker = ? AND p.timeframe = '1min'
        AND p.datetime >= ? AND p.datetime <= ?
        ORDER BY p.datetime
        """
        params = (symbol, start_date, end_date)
    elif table_name == 'market_indicator':
        query = """
        SELECT datetime, value FROM market_indicator 
        WHERE indicator_name = ? 
        AND datetime >= ? AND datetime <= ?
        ORDER BY datetime
        """
        params = (indicator_name, start_date, end_date)
    else:
        raise ValueError("Invalid table name specified.")

    df = pd.read_sql_query(query, conn, params=params)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
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

def load_precomputed_macro_data(conn, master_df, start_date, end_date, required_columns):
    """Load pre-computed macro data if needed."""
    # Check if any macro-related columns are required
    macro_keywords = ['cpi', 'fomc', 'fed', 'nfp', 'event', 'is_pre_', 'is_post_', '_inline', '_better', '_worse', 
                     '_pre_es_', '_pre_vix_', 'es_return_', 'vix_change_']
    needs_macro = any(keyword in col.lower() for col in required_columns for keyword in macro_keywords)
    
    if not needs_macro:
        logging.info("No macro event data required by strategy")
        return master_df
    
    logging.info("Loading pre-computed macro data...")
    
    # Check if pre-computed table exists
    table_check = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='macro_precomputed'", 
        conn
    )
    
    if table_check.empty:
        logging.warning("Pre-computed macro data table not found! Run precompute_macro_data.py first.")
        logging.warning("Falling back to on-the-fly calculation (slower)...")
        return load_macro_events_legacy(conn, master_df, start_date, end_date, required_columns)
    
    # Load pre-computed data
    try:
        # Get list of available columns
        cols_query = "PRAGMA table_info(macro_precomputed)"
        cols_df = pd.read_sql_query(cols_query, conn)
        available_cols = cols_df['name'].tolist()
        available_cols.remove('datetime')  # Remove datetime as it's the index
        
        # Filter to only columns that might be needed
        needed_cols = []
        for col in available_cols:
            if any(keyword in col.lower() for keyword in macro_keywords):
                needed_cols.append(col)
        
        if not needed_cols:
            logging.info("No relevant macro columns found for strategy")
            return master_df
        
        # Build query with only needed columns
        cols_str = ', '.join(needed_cols)
        query = f"""
        SELECT datetime, {cols_str}
        FROM macro_precomputed
        WHERE datetime >= ? AND datetime <= ?
        ORDER BY datetime
        """
        
        macro_df = pd.read_sql_query(query, conn, params=(start_date, end_date), parse_dates=['datetime'])
        macro_df.set_index('datetime', inplace=True)
        
        # Convert boolean columns back from int
        bool_columns = [col for col in needed_cols if col.startswith(('is_', 'cpi_', 'fomc_', 'nfp_')) and 
                       col not in ['cpi_pre_es_3d', 'cpi_pre_vix_max', 'cpi_pre_vix_rise',
                                  'fomc_pre_es_3d', 'fomc_pre_vix_max', 'fomc_pre_vix_rise']]
        for col in bool_columns:
            if col in macro_df.columns:
                macro_df[col] = macro_df[col].astype(bool)
        
        # Join with master dataframe
        master_df = master_df.join(macro_df, how='left')
        
        # Fill NaN values for boolean columns with False
        for col in bool_columns:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(False)
        
        logging.info(f"Loaded pre-computed macro data with {len(needed_cols)} columns")
        
        # Add ES return columns if needed and not in pre-computed
        if 'ES_return_1d' in required_columns and 'ES_return_1d' not in master_df.columns:
            if 'ES_close' in master_df.columns:
                master_df['ES_return_1d'] = master_df['ES_close'].pct_change(390) * 100
                
        if 'ES_return_3d' in required_columns and 'ES_return_3d' not in master_df.columns:
            if 'ES_close' in master_df.columns:
                master_df['ES_return_3d'] = master_df['ES_close'].pct_change(390 * 3) * 100
                
        if 'VIX_change_1d' in required_columns and 'VIX_change_1d' not in master_df.columns:
            if 'VIX_close' in master_df.columns:
                master_df['VIX_change_1d'] = master_df['VIX_close'].pct_change(390) * 100
        
    except Exception as e:
        logging.error(f"Error loading pre-computed macro data: {e}")
        logging.warning("Falling back to on-the-fly calculation...")
        return load_macro_events_legacy(conn, master_df, start_date, end_date, required_columns)
    
    return master_df

def load_macro_events_legacy(conn, master_df, start_date, end_date, required_columns):
    """Legacy function for loading macro events if pre-computed data not available."""
    logging.info("Loading macro events (legacy mode)...")
    
    try:
        # Load all macro events
        macro_df = pd.read_sql_query("""
            SELECT event_date, event_type, event_time, 
                   actual_value, forecast_value, surprise, surprise_pct,
                   is_inline, is_better, is_worse
            FROM macro_events
            WHERE event_date >= ? AND event_date <= ?
            ORDER BY event_date
        """, conn, params=(start_date, end_date))
        
        if not macro_df.empty:
            macro_df['event_datetime'] = pd.to_datetime(
                macro_df['event_date'] + ' ' + macro_df['event_time']
            )
            
            # Create event flags for each event type
            for event_type in ['CPI', 'FOMC', 'NFP']:
                event_data = macro_df[macro_df['event_type'] == event_type]
                
                if event_data.empty:
                    continue
                    
                # Initialize all flags
                master_df[f'is_{event_type.lower()}_day'] = False
                master_df[f'is_pre_{event_type.lower()}'] = False
                master_df[f'is_post_{event_type.lower()}'] = False
                
                # For CPI/FOMC, add surprise direction flags
                if event_type in ['CPI', 'FOMC']:
                    master_df[f'{event_type.lower()}_inline'] = False
                    master_df[f'{event_type.lower()}_better'] = False
                    master_df[f'{event_type.lower()}_worse'] = False
                
                for _, event in event_data.iterrows():
                    event_dt = event['event_datetime']
                    event_date = event_dt.date()
                    
                    # Mark event day
                    event_day_mask = master_df.index.date == event_date
                    master_df.loc[event_day_mask, f'is_{event_type.lower()}_day'] = True
                    
                    # Pre-event: previous trading day
                    pre_event_start = event_dt - pd.Timedelta(days=3)
                    pre_event_mask = (
                        (master_df.index >= pre_event_start) & 
                        (master_df.index < event_dt)
                    )
                    master_df.loc[pre_event_mask, f'is_pre_{event_type.lower()}'] = True
                    
                    # Post-event: after announcement time
                    post_event_mask = (
                        (master_df.index.date == event_date) & 
                        (master_df.index >= event_dt)
                    )
                    master_df.loc[post_event_mask, f'is_post_{event_type.lower()}'] = True
                    
                    # Add surprise direction for post-event periods
                    if event_type in ['CPI', 'FOMC'] and post_event_mask.any():
                        if event['is_inline']:
                            master_df.loc[post_event_mask, f'{event_type.lower()}_inline'] = True
                        elif event['is_better']:
                            master_df.loc[post_event_mask, f'{event_type.lower()}_better'] = True
                        elif event['is_worse']:
                            master_df.loc[post_event_mask, f'{event_type.lower()}_worse'] = True
            
            # Calculate pre-event market conditions if needed
            if any('pre_' in col for col in required_columns):
                logging.info("Calculating pre-event market conditions...")
                
                # ES returns
                if 'ES_close' in master_df.columns:
                    master_df['ES_return_1d'] = master_df['ES_close'].pct_change(390) * 100
                    master_df['ES_return_3d'] = master_df['ES_close'].pct_change(390 * 3) * 100
                
                # VIX changes
                if 'VIX_close' in master_df.columns:
                    master_df['VIX_change_1d'] = master_df['VIX_close'].pct_change(390) * 100
                
                # Store pre-event conditions
                for event_type in ['CPI', 'FOMC']:
                    pre_mask = master_df[f'is_pre_{event_type.lower()}']
                    if pre_mask.any() and 'ES_return_3d' in master_df.columns:
                        grouped = master_df[pre_mask].groupby(master_df[pre_mask].index.date)
                        
                        pre_conditions = grouped.agg({
                            'ES_return_3d': 'last',
                            'VIX_close': 'max' if 'VIX_close' in master_df.columns else 'first',
                            'VIX_change_1d': 'max' if 'VIX_change_1d' in master_df.columns else 'first'
                        })
                        
                        # Forward fill these conditions to the event day
                        for date, row in pre_conditions.iterrows():
                            next_day = date + pd.Timedelta(days=1)
                            event_mask = master_df.index.date == next_day
                            if event_mask.any():
                                master_df.loc[event_mask, f'{event_type.lower()}_pre_es_3d'] = row.get('ES_return_3d', np.nan)
                                master_df.loc[event_mask, f'{event_type.lower()}_pre_vix_max'] = row.get('VIX_close', np.nan)
                                master_df.loc[event_mask, f'{event_type.lower()}_pre_vix_rise'] = row.get('VIX_change_1d', np.nan)
            
            logging.info(f"Loaded {len(macro_df)} macro events with flags")
            
    except Exception as e:
        logging.error(f"Could not load macro events: {e}")
        # Create empty flags if loading fails
        for event_type in ['cpi', 'fomc', 'nfp']:
            master_df[f'is_{event_type}_day'] = False
            master_df[f'is_pre_{event_type}'] = False
            master_df[f'is_post_{event_type}'] = False
    
    return master_df

def get_merged_data(global_config, required_columns=None):
    """
    Main data preparation function.
    Now uses pre-computed macro data for faster loading.
    """
    settings = global_config['settings']
    db_path = settings['db_path']
    
    conn = connect_db(db_path)
    
    # Get actual date range from database
    start_date, end_date = get_date_range(
        conn, 
        settings.get('start_date'), 
        settings.get('end_date')
    )
    
    # If no required columns specified, load everything (backward compatibility)
    if not required_columns:
        logging.warning("No required columns specified, loading all data (slower)")
        required_instruments = ['ES', 'SPX', 'VIX', 'VX', 'TRIN']
        required_indicators = ['VIX_VXV_RATIO', 'MARKET_BREADTH', 'NAAIM', 'FED_STANCE', 'CNN_FEAR_GREED', 'BUFFETT_INDICATOR']
    else:
        # Extract what we need from required columns
        required_instruments = extract_required_instruments(required_columns)
        required_indicators = extract_required_indicators(required_columns)
        logging.info(f"Loading only required instruments: {required_instruments}")
        logging.info(f"Loading only required indicators: {required_indicators}")
    
    logging.info("Creating master 1-minute time index...")
    master_index = create_market_hours_index(start_date, end_date)
    master_df = pd.DataFrame(index=master_index)
    
    # 1. Load Price Data (only required instruments)
    for inst in required_instruments:
        logging.info(f"Loading and processing {inst} data...")
        try:
            price_df = load_data_from_db(conn, 'price_data', start_date, end_date, symbol=inst)
            if price_df.empty:
                logging.warning(f"No data found for {inst}")
                continue
                
            resampled_df = resample_to_1min(price_df, prefix=inst)
            
            # Join with master_df
            master_df = master_df.join(resampled_df, how='left')
            
        except Exception as e:
            logging.error(f"Could not process data for {inst}: {e}")

    # 2. Load Market Indicators (only required ones)
    for ind_name in required_indicators:
        logging.info(f"Loading and processing {ind_name} data...")
        try:
            ind_df = load_data_from_db(conn, 'market_indicator', start_date, end_date, indicator_name=ind_name)
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
    
    # 3. Load pre-computed macro data (FAST!)
    master_df = load_precomputed_macro_data(conn, master_df, start_date, end_date, required_columns or [])
    
    # 4. Perform comprehensive data validation before forward fill
    logging.info("\nData summary before forward fill:")
    for col in master_df.columns:
        non_nan = master_df[col].notna().sum()
        nan_pct = (1 - non_nan/len(master_df)) * 100
        logging.info(f"  {col}: {non_nan:,} non-NaN ({nan_pct:.1f}% NaN)")
    
    # 5. Forward fill with limit to avoid propagating stale data too far
    logging.info("\nPerforming final forward-fill...")
    # First forward fill price data with a reasonable limit (e.g., 390 minutes = 1 trading day)
    price_columns = [col for inst in required_instruments for col in [f'{inst}_open', f'{inst}_high', f'{inst}_low', f'{inst}_close', f'{inst}_volume'] if col in master_df.columns]
    if price_columns:
        master_df[price_columns] = master_df[price_columns].ffill(limit=390)
    
    # Then forward fill indicators with no limit (they update less frequently)
    indicator_columns = [col for col in master_df.columns if col not in price_columns]
    if indicator_columns:
        master_df[indicator_columns] = master_df[indicator_columns].ffill()
    
    # 6. Final validation
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
    logging.info(f"Data preparation complete. DataFrame shape: {master_df.shape}")
    return master_df