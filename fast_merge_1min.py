
"""
Data Loader and Merger Script V3 - FAST VERSION
Loads all data from SQLite database, merges to 1-minute frequency,
adds technical indicators, and handles US market hours only
Uses vectorized operations for fast processing of millions of records
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import logging
import warnings
import ta
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'backtesting_v2.db'
OUTPUT_PATH = 'merged_data_1min.parquet'
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loader_merger_v3_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and merges backtesting data to 1-minute frequency"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.gaps_log = []
        
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        logger.info("Connected to database")
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def load_price_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load all price data for a symbol"""
        query = """
        SELECT p.datetime, p.timeframe, p.open, p.high, p.low, p.close, p.volume
        FROM price_data p
        JOIN symbol s ON p.symbol_id = s.symbol_id
        WHERE s.ticker = ?
        """
        params = [symbol]
        
        if start_date:
            query += " AND p.datetime >= ?"
            params.append(start_date)
        if end_date:
            query += " AND p.datetime <= ?"
            params.append(end_date)
            
        query += " ORDER BY p.datetime"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        # Handle mixed datetime formats
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        except (ValueError, TypeError):
            df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
        
        return df
    
    def load_indicator_data(self, indicator_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load market indicator data"""
        query = """
        SELECT datetime, value
        FROM market_indicator
        WHERE indicator_name = ?
        """
        params = [indicator_name]
        
        if start_date:
            query += " AND datetime >= ?"
            params.append(start_date)
        if end_date:
            query += " AND datetime <= ?"
            params.append(end_date)
            
        query += " ORDER BY datetime"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        # Handle mixed datetime formats
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        except (ValueError, TypeError):
            df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
        
        # Don't convert to numeric for FED_STANCE
        if indicator_name != 'FED_STANCE':
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        return df
    
    def create_1min_index(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Create 1-minute index for US market hours only"""
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter for weekdays only (Mon=0, Sun=6)
        business_days = all_dates[all_dates.weekday < 5]
        
        # Create minute-level timestamps for market hours
        timestamps = []
        for date in business_days:
            market_open = pd.Timestamp.combine(date, MARKET_OPEN)
            market_close = pd.Timestamp.combine(date, MARKET_CLOSE)
            day_minutes = pd.date_range(start=market_open, end=market_close, freq='1min')[:-1]  # Exclude 16:00
            timestamps.extend(day_minutes)
        
        return pd.DatetimeIndex(timestamps)
    
    def merge_to_1min(self, df: pd.DataFrame, symbol: str, data_type: str = 'price') -> pd.DataFrame:
        """Merge different timeframe data to 1-minute frequency"""
        logger.info(f"Merging {symbol} {data_type} data to 1-minute frequency...")
        
        # Separate by timeframe
        dfs_by_timeframe = {}
        for timeframe in df['timeframe'].unique():
            tf_df = df[df['timeframe'] == timeframe].copy()
            tf_df = tf_df.set_index('datetime').sort_index()
            dfs_by_timeframe[timeframe] = tf_df
            logger.info(f"  {timeframe}: {len(tf_df)} records")
        
        # Start with 1min data if available
        if '1min' in dfs_by_timeframe:
            result = dfs_by_timeframe['1min'].copy()
        else:
            # Create empty 1min frame
            min_date = df['datetime'].min()
            max_date = df['datetime'].max()
            idx = self.create_1min_index(min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
            result = pd.DataFrame(index=idx)
        
        # Process each timeframe
        timeframe_order = ['daily', '1hour', '30min', '15min', '5min', '1min']
        
        for timeframe in timeframe_order:
            if timeframe not in dfs_by_timeframe or timeframe == '1min':
                continue
                
            tf_df = dfs_by_timeframe[timeframe]
            
            if data_type == 'price':
                # For price data, we need to handle OHLCV properly
                if timeframe == 'daily':
                    # For daily data, forward fill to all minutes of the trading day
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in tf_df.columns:
                            # Create a series that will be forward filled
                            daily_values = tf_df[col].copy()
                            
                            # For each day, set the value at market open
                            for date, value in daily_values.items():
                                if pd.notna(value):
                                    # Set at 9:30 AM of that day
                                    market_open_time = pd.Timestamp.combine(date.date(), MARKET_OPEN)
                                    if market_open_time in result.index:
                                        if col not in result.columns:
                                            result[col] = np.nan
                                        result.loc[market_open_time, col] = value
                else:
                    # For intraday timeframes, resample
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in tf_df.columns:
                            if col not in result.columns:
                                result[col] = np.nan
                            
                            # Resample to 1min
                            if col == 'open':
                                resampled = tf_df[col].resample('1min').first()
                            elif col == 'high':
                                resampled = tf_df[col].resample('1min').max()
                            elif col == 'low':
                                resampled = tf_df[col].resample('1min').min()
                            elif col == 'close':
                                resampled = tf_df[col].resample('1min').last()
                            elif col == 'volume':
                                resampled = tf_df[col].resample('1min').sum()
                            
                            # Update result where we have data
                            result[col].update(resampled)
        
        # Forward fill the data
        result = result.ffill()
        
        # Filter for market hours only
        result = result.between_time(MARKET_OPEN, time(15, 59))
        
        # Log gaps
        self.detect_and_log_gaps(result, symbol)
        
        return result
    
    def detect_and_log_gaps(self, df: pd.DataFrame, symbol: str):
        """Detect and log gaps in data"""
        # Check for missing minutes within trading days
        expected_minutes_per_day = 390  # 9:30 to 15:59
        
        by_date = df.groupby(df.index.date)
        for date, day_data in by_date:
            actual_minutes = len(day_data)
            if actual_minutes < expected_minutes_per_day:
                gap_info = {
                    'symbol': symbol,
                    'date': date,
                    'expected_minutes': expected_minutes_per_day,
                    'actual_minutes': actual_minutes,
                    'missing_minutes': expected_minutes_per_day - actual_minutes
                }
                self.gaps_log.append(gap_info)
                logger.warning(f"Gap detected: {symbol} on {date} has {actual_minutes}/{expected_minutes_per_day} minutes")
    
    def compute_es_indicators(self, es_df: pd.DataFrame) -> pd.DataFrame:
        """Compute ES indicators using daily open values to prevent lookahead"""
        logger.info("Computing ES indicators...")
        
        # First, create daily OHLC from 1-minute data
        daily_ohlc = es_df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Compute indicators on daily open values (no lookahead)
        daily_ohlc['SMA_50'] = daily_ohlc['open'].rolling(window=50).mean()
        daily_ohlc['EMA_9'] = daily_ohlc['open'].ewm(span=9, adjust=False).mean()
        daily_ohlc['EMA_15'] = daily_ohlc['open'].ewm(span=15, adjust=False).mean()
        
        # RSI on daily open
        delta = daily_ohlc['open'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
        rs = gain / loss
        daily_ohlc['RSI_2'] = 100 - (100 / (1 + rs))
        
        # 10-day high (using open prices)
        daily_ohlc['High_10D'] = daily_ohlc['open'].rolling(window=10).max()
        
        # Map daily indicators to 1-minute data
        # Set values at market open and forward fill
        for col in ['SMA_50', 'EMA_9', 'EMA_15', 'RSI_2', 'High_10D']:
            es_df[col] = np.nan
            for date, value in daily_ohlc[col].items():
                if pd.notna(value):
                    market_open_time = pd.Timestamp.combine(date, MARKET_OPEN)
                    if market_open_time in es_df.index:
                        es_df.loc[market_open_time, col] = value
            
            # Forward fill the indicator values throughout the day
            es_df[col] = es_df[col].ffill()
        
        return es_df
    
    def compute_spx_indicators(self, spx_df: pd.DataFrame) -> pd.DataFrame:
        """Compute SPX indicators using daily open values to prevent lookahead"""
        logger.info("Computing SPX indicators...")
        
        # First, create daily OHLC from 1-minute data
        daily_ohlc = spx_df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Compute 50-day SMA on daily open values (no lookahead)
        daily_ohlc['SMA_50'] = daily_ohlc['open'].rolling(window=50).mean()
        
        # Map daily indicator to 1-minute data
        spx_df['SMA_50'] = np.nan
        for date, value in daily_ohlc['SMA_50'].items():
            if pd.notna(value):
                market_open_time = pd.Timestamp.combine(date, MARKET_OPEN)
                if market_open_time in spx_df.index:
                    spx_df.loc[market_open_time, 'SMA_50'] = value
        
        # Forward fill the indicator values throughout the day
        spx_df['SMA_50'] = spx_df['SMA_50'].ffill()
        
        return spx_df
    
    def compute_vix_indicators(self, vix_df: pd.DataFrame) -> pd.DataFrame:
        """Compute VIX indicators"""
        logger.info("Computing VIX indicators...")
        
        # Create daily high values
        daily_high = vix_df['high'].resample('D').max()
        
        # Shift by 1 day to get previous day's high
        prev_day_high = daily_high.shift(1)
        
        # Map to 1-minute data
        vix_df['Prev_day_vix_high'] = np.nan
        for date, value in prev_day_high.items():
            if pd.notna(value):
                # Set for all minutes of the next trading day
                next_day_start = pd.Timestamp.combine(date, MARKET_OPEN)
                next_day_end = pd.Timestamp.combine(date, time(15, 59))
                mask = (vix_df.index >= next_day_start) & (vix_df.index <= next_day_end)
                vix_df.loc[mask, 'Prev_day_vix_high'] = value
        
        return vix_df
    
    def compute_vx_indicators(self, vx_df: pd.DataFrame) -> pd.DataFrame:
        """Compute VX indicators"""
        logger.info("Computing VX indicators...")
        
        # Create daily close values
        daily_close = vx_df['close'].resample('D').last()
        
        # Shift by 1 day to get previous day's close
        prev_day_close = daily_close.shift(1)
        
        # Map to 1-minute data
        vx_df['Prev_day_close'] = np.nan
        for date, value in prev_day_close.items():
            if pd.notna(value):
                # Set for all minutes of the next trading day
                next_day_start = pd.Timestamp.combine(date, MARKET_OPEN)
                next_day_end = pd.Timestamp.combine(date, time(15, 59))
                mask = (vx_df.index >= next_day_start) & (vx_df.index <= next_day_end)
                vx_df.loc[mask, 'Prev_day_close'] = value
        
        return vx_df
    
    def merge_indicator_fast(self, master_df: pd.DataFrame, indicator_name: str, 
                           ind_data: pd.DataFrame) -> pd.Series:
        """Fast vectorized merge of indicator data"""
        
        if ind_data.empty:
            return pd.Series(index=master_df.index, dtype=float)
        
        # Set datetime index and sort
        ind_data = ind_data.set_index('datetime').sort_index()
        
        # Remove duplicates (keep last)
        ind_data = ind_data[~ind_data.index.duplicated(keep='last')]
        
        # Get the value series
        ind_values = ind_data['value']
        
        # Choose method based on data size
        if len(ind_data) > 10000:  # Large dataset - use merge_asof
            logger.info(f"    Using merge_asof for {len(ind_data):,} records...")
            
            # Create temporary dataframe for merge_asof
            temp_df = pd.DataFrame({'datetime': master_df.index})
            
            # Reset index for merge_asof
            temp_df_reset = temp_df.reset_index(drop=True)
            ind_data_reset = ind_values.reset_index()
            
            # Merge asof - finds most recent past value
            merged = pd.merge_asof(
                temp_df_reset,
                ind_data_reset,
                on='datetime',
                direction='backward'
            )
            
            # Get the result series
            result = pd.Series(merged['value'].values, index=master_df.index)
            
        else:  # Smaller dataset - use reindex
            logger.info(f"    Using reindex for {len(ind_data):,} records...")
            
            # Reindex to master index with forward fill
            result = ind_values.reindex(master_df.index, method='ffill')
            
            # Backward fill for any remaining NaN at the beginning
            first_valid = result.first_valid_index()
            if first_valid is not None and first_valid > master_df.index[0]:
                first_value = result.loc[first_valid]
                result.loc[:first_valid] = first_value
        
        return result
    
    def merge_all_data(self, start_date: str = '2010-01-01', end_date: str = '2025-01-01'):
        """Main function to merge all data"""
        logger.info("="*60)
        logger.info("STARTING DATA MERGE PROCESS (FAST VERSION)")
        logger.info("="*60)
        
        # Create master 1-minute index
        master_index = self.create_1min_index(start_date, end_date)
        master_df = pd.DataFrame(index=master_index)
        
        # 1. Load and merge ES data
        logger.info("\n1. Processing ES data...")
        es_data = self.load_price_data('ES', start_date, end_date)
        if not es_data.empty:
            es_1min = self.merge_to_1min(es_data, 'ES')
            es_1min = self.compute_es_indicators(es_1min)
            
            # Add to master with ES_ prefix
            for col in es_1min.columns:
                master_df[f'ES_{col}'] = es_1min[col]
        
        # 2. Load and merge SPX data
        logger.info("\n2. Processing SPX data...")
        spx_data = self.load_price_data('SPX', start_date, end_date)
        if not spx_data.empty:
            spx_1min = self.merge_to_1min(spx_data, 'SPX')
            spx_1min = self.compute_spx_indicators(spx_1min)
            
            # Add to master with SPX_ prefix
            for col in spx_1min.columns:
                if col != 'timeframe':
                    master_df[f'SPX_{col}'] = spx_1min[col]
        
        # 3. Load and merge VIX data
        logger.info("\n3. Processing VIX data...")
        vix_data = self.load_price_data('VIX', start_date, end_date)
        if not vix_data.empty:
            vix_1min = self.merge_to_1min(vix_data, 'VIX')
            vix_1min = self.compute_vix_indicators(vix_1min)
            
            # Add to master with VIX_ prefix
            for col in vix_1min.columns:
                if col != 'timeframe':
                    master_df[f'VIX_{col}'] = vix_1min[col]
        
        # 4. Load and merge VX data
        logger.info("\n4. Processing VX data...")
        vx_data = self.load_price_data('VX', start_date, end_date)
        if not vx_data.empty:
            vx_1min = self.merge_to_1min(vx_data, 'VX')
            vx_1min = self.compute_vx_indicators(vx_1min)
            
            # Add to master with VX_ prefix
            for col in vx_1min.columns:
                if col != 'timeframe':
                    master_df[f'VX_{col}'] = vx_1min[col]
        
        # 5. Load and merge TRIN data (if available)
        logger.info("\n5. Processing TRIN data...")
        trin_data = self.load_price_data('TRIN', start_date, end_date)
        if not trin_data.empty:
            trin_1min = self.merge_to_1min(trin_data, 'TRIN')
            
            # Add to master with TRIN_ prefix
            for col in trin_1min.columns:
                if col != 'timeframe':
                    master_df[f'TRIN_{col}'] = trin_1min[col]
        
        # 6. Load and merge indicators - FAST VERSION
        indicator_list = [
            'VIX_VXV_RATIO',
            'MARKET_BREADTH',
            'MARKET_BREADTH_DAILY',
            'TRIN_DAILY',
            'NAAIM',
            'FED_STANCE',
            'CNN_FEAR_GREED',
            'BUFFETT_INDICATOR',
            'VXV'
        ]
        
        logger.info("\n6. Processing market indicators (FAST)...")
        for indicator in indicator_list:
            try:
                ind_data = self.load_indicator_data(indicator, start_date, end_date)
                if not ind_data.empty:
                    logger.info(f"  Processing {indicator}...")
                    
                    if indicator == 'FED_STANCE':
                        # Special handling for FED_STANCE
                        logger.info(f"    Raw FED_STANCE data shape: {ind_data.shape}")
                        
                        # Try numeric conversion first
                        ind_data['value_numeric'] = pd.to_numeric(ind_data['value'], errors='coerce')
                        
                        # If not numeric, map text values
                        if ind_data['value_numeric'].isna().all():
                            ind_data['value_clean'] = ind_data['value'].astype(str).str.strip()
                            
                            stance_mapping = {
                                'Dovish': 1.0, 'dovish': 1.0, 'DOVISH': 1.0,
                                'Neutral': 2.0, 'neutral': 2.0, 'NEUTRAL': 2.0,
                                'Hawkish': 3.0, 'hawkish': 3.0, 'HAWKISH': 3.0,
                            }
                            
                            ind_data['value_numeric'] = ind_data['value_clean'].map(stance_mapping)
                        
                        # Use the numeric values
                        ind_data['value'] = ind_data['value_numeric']
                        ind_data = ind_data[ind_data['value'].notna()]
                        
                        if not ind_data.empty:
                            # Use fast merge
                            master_df['FED_STANCE'] = self.merge_indicator_fast(master_df, indicator, ind_data)
                            
                            # Create text version
                            reverse_mapping = {1.0: 'Dovish', 2.0: 'Neutral', 3.0: 'Hawkish'}
                            master_df['FED_STANCE_TEXT'] = master_df['FED_STANCE'].map(reverse_mapping)
                            
                            non_null = master_df['FED_STANCE'].notna().sum()
                            logger.info(f"    FED_STANCE merged: {non_null:,} non-null values")
                    
                    else:
                        # Regular numeric indicators
                        # Remove any non-numeric values
                        ind_data = ind_data[ind_data['value'].notna()]
                        
                        if not ind_data.empty:
                            # Use fast merge
                            master_df[indicator] = self.merge_indicator_fast(master_df, indicator, ind_data)
                            
                            non_null = master_df[indicator].notna().sum()
                            if non_null > 0:
                                logger.info(f"    {indicator} merged: {non_null:,} non-null values")
                            else:
                                logger.warning(f"    WARNING: No data merged for {indicator}")
                            
            except Exception as e:
                logger.error(f"  Error processing {indicator}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 7. Final cleanup
        logger.info("\n7. Final cleanup...")
        
        # Remove rows where all price columns are NaN
        price_cols = [col for col in master_df.columns if any(x in col for x in ['open', 'high', 'low', 'close'])]
        master_df = master_df.dropna(subset=price_cols, how='all')
        
        # Log final status
        logger.info("\nFinal Data Status:")
        key_indicators = ['FED_STANCE', 'VIX_VXV_RATIO', 'MARKET_BREADTH', 'NAAIM', 'CNN_FEAR_GREED']
        for indicator in key_indicators:
            if indicator in master_df.columns:
                non_null = master_df[indicator].notna().sum()
                pct = (non_null / len(master_df)) * 100
                logger.info(f"  {indicator}: {non_null:,} values ({pct:.1f}%)")
        
        return master_df
    
    def save_gaps_log(self):
        """Save gaps log to file"""
        if self.gaps_log:
            gaps_df = pd.DataFrame(self.gaps_log)
            gaps_df.to_csv('data_gaps_log_v3_fast.csv', index=False)
            logger.info(f"Gaps log saved: {len(self.gaps_log)} gaps found")
    
    def run(self, start_date: str = '2010-01-01', end_date: str = '2025-01-01'):
        """Run the complete data loading and merging process"""
        try:
            self.connect()
            
            # Merge all data
            merged_df = self.merge_all_data(start_date, end_date)
            
            # Save to parquet
            merged_df.to_parquet(OUTPUT_PATH, compression='snappy')
            logger.info(f"\nMerged data saved to: {OUTPUT_PATH}")
            logger.info(f"Shape: {merged_df.shape}")
            logger.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
            
            # Save gaps log
            self.save_gaps_log()
            
            # Print summary statistics
            logger.info("\n" + "="*60)
            logger.info("SUMMARY STATISTICS")
            logger.info("="*60)
            
            # Count non-null values for each column
            non_null_counts = merged_df.count()
            total_rows = len(merged_df)
            
            # Key indicators summary
            logger.info("\nKey Indicators Coverage:")
            key_cols = ['ES_close', 'SPX_close', 'VIX_close', 'ES_SMA_50', 'SPX_SMA_50',
                       'FED_STANCE', 'VIX_VXV_RATIO', 'MARKET_BREADTH', 'NAAIM', 
                       'CNN_FEAR_GREED', 'BUFFETT_INDICATOR']
            
            for col in key_cols:
                if col in merged_df.columns:
                    count = non_null_counts[col]
                    pct = (count / total_rows) * 100
                    logger.info(f"  {col}: {count:,} ({pct:.1f}%)")
                else:
                    logger.warning(f"  {col}: NOT FOUND")
            
            # Memory usage
            memory_mb = merged_df.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"\nMemory usage: {memory_mb:.1f} MB")
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error during data merge: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            self.close()

def main():
    """Main execution"""
    loader = DataLoader(DB_PATH)
    
    # Run the merge process
    merged_data = loader.run(start_date='2010-01-01', end_date='2025-01-01')
    
    # Quick validation
    logger.info("\nSample Data Check:")
    check_cols = ['ES_close', 'VIX_VXV_RATIO', 'FED_STANCE', 'MARKET_BREADTH']
    available_cols = [col for col in check_cols if col in merged_data.columns]
    
    if available_cols:
        sample = merged_data[available_cols].dropna().head(10)
        logger.info("\nFirst 10 rows with all indicators:")
        logger.info(sample)

if __name__ == "__main__":
    main()