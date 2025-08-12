"""
precompute_macro_data.py
Pre-computes all macro event flags and market conditions once.
Fixed to create focused event windows instead of all-day flags.
"""
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import time, datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MARKET_OPEN = time(8, 00)
MARKET_CLOSE = time(16, 0)

class MacroDataPrecomputer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
    def create_precomputed_table(self):
        """Create table for pre-computed macro data."""
        logging.info("Creating pre-computed macro data table...")
        
        # Drop if exists
        self.cursor.execute("DROP TABLE IF EXISTS macro_precomputed")
        
        # Create new table
        self.cursor.execute("""
            CREATE TABLE macro_precomputed (
                datetime DATETIME PRIMARY KEY,
                -- Event flags
                is_cpi_day BOOLEAN DEFAULT FALSE,
                is_pre_cpi BOOLEAN DEFAULT FALSE,
                is_post_cpi BOOLEAN DEFAULT FALSE,
                is_cpi_window BOOLEAN DEFAULT FALSE,
                cpi_inline BOOLEAN DEFAULT FALSE,
                cpi_better BOOLEAN DEFAULT FALSE,
                cpi_worse BOOLEAN DEFAULT FALSE,
                
                is_fomc_day BOOLEAN DEFAULT FALSE,
                is_pre_fomc BOOLEAN DEFAULT FALSE,
                is_post_fomc BOOLEAN DEFAULT FALSE,
                is_fomc_window BOOLEAN DEFAULT FALSE,
                fomc_inline BOOLEAN DEFAULT FALSE,
                fomc_better BOOLEAN DEFAULT FALSE,
                fomc_worse BOOLEAN DEFAULT FALSE,
                
                is_nfp_day BOOLEAN DEFAULT FALSE,
                is_pre_nfp BOOLEAN DEFAULT FALSE,
                is_post_nfp BOOLEAN DEFAULT FALSE,
                is_nfp_window BOOLEAN DEFAULT FALSE,
                
                -- Pre-event conditions
                cpi_pre_es_3d REAL,
                cpi_pre_vix_max REAL,
                cpi_pre_vix_rise REAL,
                fomc_pre_es_3d REAL,
                fomc_pre_vix_max REAL,
                fomc_pre_vix_rise REAL,
                
                -- General market conditions (for event days)
                es_return_1d REAL,
                es_return_3d REAL,
                vix_change_1d REAL,
                vix_level REAL,
                
                -- Minutes since event
                mins_since_cpi REAL,
                mins_since_fomc REAL,
                mins_since_nfp REAL
            )
        """)
        
        # Create index for faster queries
        self.cursor.execute("CREATE INDEX idx_macro_datetime ON macro_precomputed(datetime)")
        self.conn.commit()
        
    def create_market_hours_index(self):
        """Create 1-minute market hours index."""
        # Get date range from price data
        query = """
        SELECT MIN(datetime) as min_date, MAX(datetime) as max_date 
        FROM price_data 
        WHERE symbol_id = (SELECT symbol_id FROM symbol WHERE ticker = 'ES')
        """
        result = pd.read_sql_query(query, self.conn)
        start_date = pd.to_datetime(result['min_date'].iloc[0])
        end_date = pd.to_datetime(result['max_date'].iloc[0])
        
        logging.info(f"Creating market hours index from {start_date} to {end_date}")
        
        all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='B')
        market_minutes = []
        for date in all_dates:
            day_start = pd.Timestamp.combine(date, MARKET_OPEN)
            day_end = pd.Timestamp.combine(date, MARKET_CLOSE)
            market_minutes.append(pd.date_range(start=day_start, end=day_end, freq='1min'))
        
        return pd.DatetimeIndex(np.concatenate(market_minutes))
    
    def load_price_data(self):
        """Load ES and VIX price data for calculations."""
        logging.info("Loading price data for calculations...")
        
        # ES data
        es_query = """
        SELECT p.datetime, p.close 
        FROM price_data p 
        JOIN symbol s ON p.symbol_id = s.symbol_id 
        WHERE s.ticker = 'ES' AND p.timeframe = '1min'
        ORDER BY p.datetime
        """
        es_df = pd.read_sql_query(es_query, self.conn, parse_dates=['datetime'])
        es_df.set_index('datetime', inplace=True)
        
        # VIX data
        vix_query = """
        SELECT p.datetime, p.close 
        FROM price_data p 
        JOIN symbol s ON p.symbol_id = s.symbol_id 
        WHERE s.ticker = 'VIX' AND p.timeframe = '1min'
        ORDER BY p.datetime
        """
        vix_df = pd.read_sql_query(vix_query, self.conn, parse_dates=['datetime'])
        vix_df.set_index('datetime', inplace=True)
        
        return es_df, vix_df
    
    def precompute_all(self):
        """Main pre-computation function."""
        logging.info("Starting macro data pre-computation...")
        
        # Create market hours index
        master_index = self.create_market_hours_index()
        
        # Initialize DataFrame
        df = pd.DataFrame(index=master_index)
        
        # Load price data
        es_df, vix_df = self.load_price_data()
        
        # Join price data
        df['es_close'] = es_df['close'].reindex(df.index, method='ffill')
        df['vix_close'] = vix_df['close'].reindex(df.index, method='ffill')
        
        # Calculate general market conditions
        logging.info("Calculating market conditions...")
        df['es_return_1d'] = df['es_close'].pct_change(390) * 100
        df['es_return_3d'] = df['es_close'].pct_change(390 * 3) * 100
        df['vix_change_1d'] = df['vix_close'].pct_change(390) * 100
        df['vix_level'] = df['vix_close']
        
        # Load macro events
        macro_query = """
        SELECT event_date, event_type, event_time, 
               is_inline, is_better, is_worse
        FROM macro_events
        ORDER BY event_date
        """
        macro_df = pd.read_sql_query(macro_query, self.conn)
        
        if not macro_df.empty:
            macro_df['event_datetime'] = pd.to_datetime(
                macro_df['event_date'] + ' ' + macro_df['event_time']
            )
            
            # Process each event type
            for event_type in ['CPI', 'FOMC', 'NFP']:
                logging.info(f"Processing {event_type} events...")
                event_data = macro_df[macro_df['event_type'] == event_type]
                
                if event_data.empty:
                    continue
                
                event_lower = event_type.lower()
                
                # Initialize flags
                df[f'is_{event_lower}_day'] = False
                df[f'is_pre_{event_lower}'] = False
                df[f'is_post_{event_lower}'] = False
                df[f'is_{event_lower}_window'] = False
                df[f'mins_since_{event_lower}'] = np.nan
                
                if event_type in ['CPI', 'FOMC']:
                    df[f'{event_lower}_inline'] = False
                    df[f'{event_lower}_better'] = False
                    df[f'{event_lower}_worse'] = False
                
                # Process each event
                for _, event in event_data.iterrows():
                    event_dt = event['event_datetime']
                    event_date = event_dt.date()
                    
                    # Event day (entire day)
                    event_day_mask = df.index.date == event_date
                    df.loc[event_day_mask, f'is_{event_lower}_day'] = True
                    
                    # Pre-event (3 days before to event time)
                    pre_event_start = event_dt - pd.Timedelta(days=3)
                    pre_event_mask = (df.index >= pre_event_start) & (df.index < event_dt)
                    df.loc[pre_event_mask, f'is_pre_{event_lower}'] = True
                    
                    # Post-event (after announcement time on event day)
                    post_event_mask = (df.index.date == event_date) & (df.index >= event_dt)
                    df.loc[post_event_mask, f'is_post_{event_lower}'] = True
                    
                    # Event window (focused 3-hour window after announcement)
                    event_window_end = event_dt + pd.Timedelta(hours=3)
                    event_window_mask = (df.index >= event_dt) & (df.index <= event_window_end)
                    df.loc[event_window_mask, f'is_{event_lower}_window'] = True
                    
                    # Minutes since event (for post-event period)
                    if post_event_mask.any():
                        mins_since = (df.index[post_event_mask] - event_dt).total_seconds() / 60
                        df.loc[post_event_mask, f'mins_since_{event_lower}'] = mins_since
                    
                    # Surprise directions (for entire post-event period)
                    if event_type in ['CPI', 'FOMC'] and post_event_mask.any():
                        if event['is_inline']:
                            df.loc[post_event_mask, f'{event_lower}_inline'] = True
                        elif event['is_better']:
                            df.loc[post_event_mask, f'{event_lower}_better'] = True
                        elif event['is_worse']:
                            df.loc[post_event_mask, f'{event_lower}_worse'] = True
                    
                    # Calculate pre-event conditions
                    if event_type in ['CPI', 'FOMC'] and pre_event_mask.any():
                        pre_data = df[pre_event_mask]
                        if len(pre_data) > 0:
                            # Get conditions from last pre-event day
                            last_pre_es_3d = pre_data['es_return_3d'].iloc[-1]
                            max_pre_vix = pre_data['vix_close'].max()
                            max_pre_vix_rise = pre_data['vix_change_1d'].max()
                            
                            # Store on event day and post-event period
                            event_and_post_mask = event_day_mask | post_event_mask
                            df.loc[event_and_post_mask, f'{event_lower}_pre_es_3d'] = last_pre_es_3d
                            df.loc[event_and_post_mask, f'{event_lower}_pre_vix_max'] = max_pre_vix
                            df.loc[event_and_post_mask, f'{event_lower}_pre_vix_rise'] = max_pre_vix_rise
        
        # Save to database
        logging.info("Saving pre-computed data to database...")
        
        # Name the index before resetting
        df.index.name = 'datetime'
        
        # Select columns to save (exclude temporary price columns)
        columns_to_save = [col for col in df.columns if col not in ['es_close', 'vix_close']]
        df_to_save = df[columns_to_save].copy()
        
        # Reset index to make datetime a column
        df_to_save = df_to_save.reset_index()
        
        # Convert boolean columns to int for SQLite
        bool_columns = [col for col in columns_to_save if col.startswith(('is_', 'cpi_', 'fomc_', 'nfp_')) and 
                       col not in ['cpi_pre_es_3d', 'cpi_pre_vix_max', 'cpi_pre_vix_rise',
                                  'fomc_pre_es_3d', 'fomc_pre_vix_max', 'fomc_pre_vix_rise',
                                  'mins_since_cpi', 'mins_since_fomc', 'mins_since_nfp']]
        for col in bool_columns:
            df_to_save[col] = df_to_save[col].astype(int)
        
        # Save in chunks
        chunk_size = 10000
        total_rows = len(df_to_save)
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df_to_save.iloc[start_idx:end_idx]
            
            # Use 'append' mode and no index
            chunk.to_sql('macro_precomputed', self.conn, if_exists='append', index=False)
            
            if start_idx % 100000 == 0:
                logging.info(f"  Saved {start_idx:,} / {total_rows:,} rows...")
        
        self.conn.commit()
        logging.info(f"Pre-computation complete! Saved {total_rows:,} rows.")
        
        # Show summary
        self.show_summary()
        
    def show_summary(self):
        """Display summary of pre-computed data."""
        logging.info("\nPre-computed data summary:")
        
        # Count event days
        for event_type in ['cpi', 'fomc', 'nfp']:
            count_query = f"SELECT COUNT(DISTINCT DATE(datetime)) FROM macro_precomputed WHERE is_{event_type}_day = 1"
            count = self.cursor.execute(count_query).fetchone()[0]
            logging.info(f"  {event_type.upper()} event days: {count}")
            
            # Count event windows
            window_query = f"SELECT COUNT(*) FROM macro_precomputed WHERE is_{event_type}_window = 1"
            window_count = self.cursor.execute(window_query).fetchone()[0]
            logging.info(f"  {event_type.upper()} window minutes: {window_count} (~{window_count/180:.0f} events)")
        
        # Sample data
        sample_query = """
        SELECT datetime, is_cpi_day, is_post_cpi, is_cpi_window, cpi_inline, 
               mins_since_cpi, cpi_pre_es_3d, es_return_3d
        FROM macro_precomputed
        WHERE is_cpi_window = 1
        LIMIT 10
        """
        sample_df = pd.read_sql_query(sample_query, self.conn)
        logging.info("\nSample CPI window data:")
        logging.info(sample_df)
        
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python precompute_macro_data.py <db_path>")
        print("Example: python precompute_macro_data.py ../backtesting_v2.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    precomputer = MacroDataPrecomputer(db_path)
    
    try:
        precomputer.create_precomputed_table()
        precomputer.precompute_all()
        print("\n✅ Pre-computation complete! The macro_precomputed table is ready for use.")
        print("Run this script again if you update the macro_events table.")
    finally:
        precomputer.close()

if __name__ == "__main__":
    main()