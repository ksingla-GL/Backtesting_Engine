
"""
Comprehensive Data Migration Script for Backtesting Database
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from typing import Dict, List, Tuple, Optional
import glob

# Configuration
DB_PATH = 'backtesting_v2.db'
DATA_ROOT = 'Backtesting_data'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_migration_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataMigrator:
    """Handles migration of financial data to SQLite database"""
    
    def __init__(self, db_path: str, data_root: str):
        self.db_path = db_path
        self.data_root = data_root
        self.conn = None
        self.cursor = None
        self.processed_files = set()  # Track processed files to avoid duplicates
        self.stats = {
            'price_records': 0,
            'indicator_records': 0,
            'errors': 0,
            'files_processed': 0,
            'files_skipped': 0
        }
        
    def connect(self):
        """Establish database connection with optimizations"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = 10000")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.cursor = self.conn.cursor()
        logger.info("Database connection established")
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def create_schema(self):
        """Create database schema if not exists"""
        logger.info("Creating database schema...")
        
        # Symbol table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol (
                symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker VARCHAR(10) UNIQUE NOT NULL,
                exchange VARCHAR(20) DEFAULT 'NYSE'
            )
        """)
        
        # Price data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                datetime DATETIME NOT NULL,
                symbol_id INTEGER NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER,
                PRIMARY KEY (datetime, symbol_id, timeframe),
                FOREIGN KEY (symbol_id) REFERENCES symbol(symbol_id)
            )
        """)
        
        # Market indicator table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_indicator (
                datetime DATETIME NOT NULL,
                indicator_name VARCHAR(50) NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (datetime, indicator_name)
            )
        """)
        
        # Create indexes
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_data(symbol_id, datetime)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_timeframe ON price_data(timeframe)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_date ON market_indicator(datetime)")
        
        self.conn.commit()
        logger.info("Schema created successfully")
    
    def get_or_create_symbol(self, ticker: str) -> Optional[int]:
        """Get symbol_id or create new symbol"""
        # Check if it's a tradeable symbol
        tradeable_symbols = ['ES', 'SPX', 'SPY', 'VIX', 'VX', 'VXV', 'TRIN']
        if ticker not in tradeable_symbols:
            return None
            
        self.cursor.execute("SELECT symbol_id FROM symbol WHERE ticker = ?", (ticker,))
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        else:
            self.cursor.execute("INSERT INTO symbol (ticker) VALUES (?)", (ticker,))
            self.conn.commit()
            return self.cursor.lastrowid
    
    def convert_european_number(self, value) -> float:
        """Convert European number format (1.234,56) to float"""
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        # Convert string: replace comma with dot
        return float(str(value).replace(',', '.'))
    
    def parse_datetime(self, date_str: str, time_str: str = None) -> Optional[str]:
        """Parse various datetime formats"""
        if time_str:
            datetime_str = f"{date_str} {time_str}"
        else:
            datetime_str = date_str
            
        # Try different formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%d.%m.%Y %H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d.%m.%Y'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(datetime_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                continue
                
        # Try pandas parser as last resort
        try:
            dt = pd.to_datetime(datetime_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return None
    
    def load_trin_file(self, filepath: str) -> int:
        """Special handler for TRIN files"""
        logger.info(f"Loading TRIN data from {os.path.basename(filepath)}...")
        
        symbol_id = self.get_or_create_symbol('TRIN')
        if not symbol_id:
            return 0
        
        try:
            # Read TRIN file with specific format
            df = pd.read_csv(filepath, sep=',')
            
            # TRIN files have format: Time,Open,High,Low,Last,Change,%Chg,Volume
            if 'Last' in df.columns:
                # This is the correct TRIN format
                records = []
                loaded = 0
                
                for idx, row in df.iterrows():
                    try:
                        # Parse datetime
                        dt = pd.to_datetime(row['Time'])
                        dt_formatted = dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Get OHLCV values (Last = Close)
                        open_val = float(row['Open'])
                        high_val = float(row['High'])
                        low_val = float(row['Low'])
                        close_val = float(row['Last'])
                        volume_val = int(row['Volume'])
                        
                        records.append((
                            dt_formatted, symbol_id, '1min',
                            open_val, high_val, low_val, close_val, volume_val
                        ))
                        
                        if len(records) >= 10000:
                            self.cursor.executemany("""
                                INSERT OR IGNORE INTO price_data 
                                (datetime, symbol_id, timeframe, open, high, low, close, volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, records)
                            loaded += len(records)
                            if loaded % 100000 == 0:
                                logger.info(f"  Loaded {loaded:,} TRIN records...")
                            records = []
                            
                    except Exception as e:
                        if idx < 5:
                            logger.error(f"Error on TRIN row {idx}: {e}")
                
                # Insert remaining records
                if records:
                    self.cursor.executemany("""
                        INSERT OR IGNORE INTO price_data 
                        (datetime, symbol_id, timeframe, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)
                    loaded += len(records)
                
                self.conn.commit()
                self.stats['price_records'] += loaded
                logger.info(f"  Successfully loaded {loaded:,} TRIN records")
                return loaded
            else:
                # Fall back to regular price file loading
                return self.load_price_file(filepath, 'TRIN', '1min')
                
        except Exception as e:
            logger.error(f"Failed to load TRIN file: {e}")
            return 0
    
    def load_price_file(self, filepath: str, symbol: str, timeframe: str) -> int:
        """Load a single price data file"""
        logger.info(f"Loading {os.path.basename(filepath)} for {symbol} {timeframe}...")
        
        symbol_id = self.get_or_create_symbol(symbol)
        if not symbol_id:
            logger.warning(f"Skipping {symbol} - not a tradeable symbol")
            return 0
            
        try:
            # Determine file format and read accordingly
            if filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                # Try different separators for text files
                for sep in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(filepath, sep=sep)
                        if len(df.columns) >= 5:  # Need at least OHLC
                            break
                    except:
                        continue
            
            # Handle different column structures
            if len(df.columns) == 7 and df.columns[0] != 'Date':
                # Likely Date, Time, OHLC, Volume
                df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            elif len(df.columns) == 6:
                # Could be DateTime, OHLC, Volume or Date, OHLC, Volume
                if 'time' in str(df.columns[0]).lower() or 'datetime' in str(df.columns[0]).lower():
                    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                else:
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            elif len(df.columns) == 5:
                df.columns = ['datetime', 'open', 'high', 'low', 'close']
                df['volume'] = 0
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Process records
            records = []
            records_count = 0
            total_rows = len(df)
            for idx, row in df.iterrows():
                try:
                    # Parse datetime
                    if 'datetime' in df.columns:
                        dt_str = self.parse_datetime(str(row['datetime']))
                    else:
                        dt_str = self.parse_datetime(str(row['date']), str(row.get('time', '')))
                    
                    if not dt_str:
                        continue
                    
                    # Handle European number format
                    open_val = self.convert_european_number(row['open'])
                    high_val = self.convert_european_number(row['high'])
                    low_val = self.convert_european_number(row['low'])
                    close_val = self.convert_european_number(row['close'])
                    volume_val = int(self.convert_european_number(row.get('volume', 0)))
                    
                    records.append((
                        dt_str, symbol_id, timeframe,
                        open_val, high_val, low_val, close_val, volume_val
                    ))
                    
                    # Bulk insert every 10000 records
                    if len(records) >= 10000:
                        self.cursor.executemany("""
                            INSERT OR IGNORE INTO price_data 
                            (datetime, symbol_id, timeframe, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, records)
                        records_count += len(records)
                        if records_count % 100000 == 0:
                            logger.info(f"  Progress: {records_count:,} / ~{total_rows:,} records loaded...")
                        records = []
                        
                except Exception as e:
                    self.stats['errors'] += 1
                    if self.stats['errors'] <= 10:
                        logger.error(f"Error processing row: {e}")
            
            # Insert remaining records
            if records:
                self.cursor.executemany("""
                    INSERT OR IGNORE INTO price_data 
                    (datetime, symbol_id, timeframe, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                records_count += len(records)
            
            self.conn.commit()
            self.stats['price_records'] += records_count
            logger.info(f"  Loaded {records_count} records")
            return records_count
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            self.stats['files_skipped'] += 1
            return 0
    
    def load_vix_vxv_ratio(self) -> int:
        """Special handler for VIX_VXV_Ratio.parquet"""
        logger.info("Loading VIX_VXV_RATIO data...")
        
        parquet_path = os.path.join(self.data_root, '1 min', 'VIX_VXV_Ratio.parquet')
        
        if not os.path.exists(parquet_path):
            logger.error(f"VIX_VXV_Ratio.parquet not found at {parquet_path}")
            return 0
        
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_path)
            logger.info(f"  Loaded {len(df)} rows from parquet")
            logger.info(f"  Columns found: {list(df.columns)}")
            
            # Combine date and time to create datetime
            # Handle the date format properly
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            
            # Use close_ratio as the value
            df['value'] = df['close_ratio']
            
            logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            logger.info(f"  Sample values: {df['value'].head()}")
            
            # Clear existing data
            self.cursor.execute("DELETE FROM market_indicator WHERE indicator_name = 'VIX_VXV_RATIO'")
            
            # Process in batches
            batch_size = 10000
            total_loaded = 0
            
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]
                
                records = []
                for _, row in batch.iterrows():
                    try:
                        dt_formatted = row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
                        value = float(row['value'])
                        records.append((dt_formatted, 'VIX_VXV_RATIO', str(value)))
                    except Exception as e:
                        if total_loaded == 0:  # Only log first error
                            logger.error(f"Error processing row: {e}")
                            logger.error(f"Row data: {row}")
                
                if records:
                    self.cursor.executemany("""
                        INSERT OR IGNORE INTO market_indicator 
                        (datetime, indicator_name, value)
                        VALUES (?, ?, ?)
                    """, records)
                    
                    total_loaded += len(records)
                    if total_loaded % 100000 == 0:
                        logger.info(f"    Loaded {total_loaded:,} records...")
            
            self.conn.commit()
            self.stats['indicator_records'] += total_loaded
            logger.info(f"  Successfully loaded {total_loaded:,} VIX_VXV_RATIO records")
            return total_loaded
            
        except Exception as e:
            logger.error(f"Failed to load VIX_VXV_RATIO: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
    
    def load_fed_stance(self, filepath: str) -> int:
        """Special handler for FedReserve stance.xlsx"""
        logger.info("Loading FED_STANCE data...")
        
        try:
            # Read the Excel file
            df = pd.read_excel(filepath)
            logger.info(f"  Loaded {len(df)} rows from Excel")
            logger.info(f"  Columns found: {list(df.columns)}")
            
            # The columns should be: time, Interest rate, mom, policy
            # We only need time and policy
            
            # Rename columns for clarity
            df.columns = [col.strip() for col in df.columns]
            
            # Find the time and policy columns
            time_col = None
            policy_col = None
            
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    time_col = col
                elif 'policy' in col.lower():
                    policy_col = col
            
            if not time_col or not policy_col:
                logger.error(f"Could not find required columns. Found: {list(df.columns)}")
                return 0
                
            logger.info(f"  Using columns: time='{time_col}', policy='{policy_col}'")
            
            # Clear existing data
            self.cursor.execute("DELETE FROM market_indicator WHERE indicator_name = 'FED_STANCE'")
            
            # Process records
            loaded = 0
            for _, row in df.iterrows():
                try:
                    # Parse the date - it's in format like "10/1/2023"
                    date_val = pd.to_datetime(row[time_col])
                    dt_formatted = date_val.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Get policy value
                    policy = str(row[policy_col]).strip()
                    
                    # Store the policy as value
                    self.cursor.execute("""
                        INSERT OR IGNORE INTO market_indicator 
                        (datetime, indicator_name, value)
                        VALUES (?, ?, ?)
                    """, (dt_formatted, 'FED_STANCE', policy))
                    loaded += 1
                    
                except Exception as e:
                    if loaded == 0:  # Only log first error
                        logger.error(f"Error processing row: {e}")
                        logger.error(f"Row data: {row}")
            
            self.conn.commit()
            self.stats['indicator_records'] += loaded
            logger.info(f"  Successfully loaded {loaded} FED_STANCE records")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load FED_STANCE: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
    
    def load_indicator_file(self, filepath: str, indicator_name: str) -> int:
        """Load indicator data from file - with special handling for FED_STANCE"""
        
        # Special handling for FedReserve stance
        if indicator_name == 'FED_STANCE' and 'FedReserve' in filepath:
            return self.load_fed_stance(filepath)
        
        # Rest of the original load_indicator_file code...
        logger.info(f"Loading {os.path.basename(filepath)} as {indicator_name}...")
        
        try:
            # Read file based on type
            if filepath.endswith(('.xlsx', '.xls')):
                # Try to read Excel file
                try:
                    xl = pd.ExcelFile(filepath)
                    # Try to find the right sheet
                    sheet_name = None
                    for name in xl.sheet_names:
                        if 'data' in name.lower() or len(xl.sheet_names) == 1:
                            sheet_name = name
                            break
                    if not sheet_name:
                        sheet_name = xl.sheet_names[0]
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                except:
                    # If Excel fails, it might be a different format
                    logger.warning(f"Could not read {filepath} as Excel, skipping")
                    self.stats['files_skipped'] += 1
                    return 0
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                # CSV/TXT - try different separators
                for sep in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(filepath, sep=sep)
                        if len(df.columns) >= 2:
                            break
                    except:
                        continue
            
            # Standardize column names
            df.columns = [str(col).strip().lower() for col in df.columns]
            
            # Find date and value columns
            date_col = None
            value_col = None
            
            # Look for date column
            for col in df.columns:
                if any(x in col for x in ['date', 'time', 'datetime']):
                    date_col = col
                    break
            
            # Look for value column
            value_keywords = ['value', 'index', 'fear', 'greed', 'buffett', 'ratio', 
                            'indicator', 'close', 'level', 'score']
            for col in df.columns:
                if col != date_col and any(keyword in col.lower() for keyword in value_keywords):
                    value_col = col
                    break
            
            # If no value column found, use the second column
            if not value_col and len(df.columns) >= 2:
                value_col = [col for col in df.columns if col != date_col][0]
            
            if not date_col or not value_col:
                logger.warning(f"Could not identify date/value columns in {filepath}")
                logger.warning(f"Columns found: {list(df.columns)}")
                self.stats['files_skipped'] += 1
                return 0
            
            logger.info(f"  Using columns: date='{date_col}', value='{value_col}'")
            
            # Clear existing data for this indicator
            self.cursor.execute("DELETE FROM market_indicator WHERE indicator_name = ?", (indicator_name,))
            
            # Process records
            loaded = 0
            for _, row in df.iterrows():
                try:
                    dt_str = self.parse_datetime(str(row[date_col]))
                    if dt_str and not pd.isna(row[value_col]):
                        self.cursor.execute("""
                            INSERT OR IGNORE INTO market_indicator 
                            (datetime, indicator_name, value)
                            VALUES (?, ?, ?)
                        """, (dt_str, indicator_name, str(row[value_col])))
                        loaded += 1
                except:
                    pass
            
            self.conn.commit()
            self.stats['indicator_records'] += loaded
            logger.info(f"  Loaded {loaded} records")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            self.stats['files_skipped'] += 1
            return 0
    
    def find_and_load_files(self, folder_path: str, pattern: str, symbol: str, timeframe: str):
        """Find and load files matching pattern"""
        files = glob.glob(os.path.join(folder_path, pattern))
        for filepath in files:
            # Skip if already processed
            if filepath in self.processed_files:
                logger.debug(f"Skipping already processed file: {os.path.basename(filepath)}")
                continue
                
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                self.processed_files.add(filepath)  # Mark as processed
                
                # Special handling for TRIN files
                if symbol == 'TRIN' and timeframe == '1min':
                    self.load_trin_file(filepath)
                else:
                    self.load_price_file(filepath, symbol, timeframe)
                
                self.stats['files_processed'] += 1
    
    def migrate_all_data(self):
        """Main migration function"""
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE DATA MIGRATION")
        logger.info("="*60)
        
        start_time = time.time()
        
        # 1. Load intraday price data
        logger.info("\n1. LOADING INTRADAY PRICE DATA")
        logger.info("-"*40)
        
        # Map of folder names to timeframes
        timeframe_folders = {
            '1 min': '1min',
            '5 min': '5min', 
            '15 min': '15min',
            '30 min': '30min',
            '1 hour': '1hour'
        }
        
        # Process each timeframe folder
        for folder_name, timeframe in timeframe_folders.items():
            folder_path = os.path.join(self.data_root, folder_name)
            if not os.path.exists(folder_path):
                logger.warning(f"Folder {folder_path} does not exist")
                continue
                
            logger.info(f"\nProcessing {folder_name} folder...")
            
            # Look for all possible files for each symbol
            symbols = ['ES', 'SPX', 'VIX', 'VX']
            for symbol in symbols:
                # Try different file patterns
                patterns = [
                    f"{symbol}_{timeframe}_*.txt",
                    f"{symbol}_{timeframe}_*.csv",
                    f"{symbol}_{timeframe}_*.xlsx",
                    f"{symbol}_{timeframe}_*.xls",
                    f"{symbol}_*.txt",  # Without timeframe in filename
                    f"{symbol}_*.csv",
                    f"{symbol}_*.xlsx"
                ]
                
                for pattern in patterns:
                    self.find_and_load_files(folder_path, pattern, symbol, timeframe)
            
            # Special handling for TRIN 1-minute data
            if timeframe == '1min':
                logger.info("  Looking for TRIN 1-minute data...")
                trin_patterns = ['TRIN_1min*.txt', 'TRIN_1min*.csv', 'TRIN*.txt']
                for pattern in trin_patterns:
                    self.find_and_load_files(folder_path, pattern, 'TRIN', '1min')
        
        # 2. Load daily price data
        logger.info("\n2. LOADING DAILY PRICE DATA")
        logger.info("-"*40)
        
        daily_folder = os.path.join(self.data_root, 'Daily')
        if os.path.exists(daily_folder):
            # Process all files in daily folder
            for filename in os.listdir(daily_folder):
                filepath = os.path.join(daily_folder, filename)
                
                # Skip if already processed
                if filepath in self.processed_files:
                    continue
                    
                if os.path.isfile(filepath) and not filename.startswith('~'):  # Skip temp files
                    # Extract symbol from filename
                    symbol = None
                    for s in ['ES', 'SPX', 'VIX', 'VX', 'VXV', 'TRIN']:
                        if s in filename.upper():
                            symbol = s
                            break
                    
                    # Check if it's a price file or indicator
                    if symbol and symbol not in ['VXV']:  # Regular price data
                        self.processed_files.add(filepath)
                        self.load_price_file(filepath, symbol, 'daily')
                        self.stats['files_processed'] += 1
                    elif symbol == 'VXV':
                        # Load VXV as an indicator
                        self.processed_files.add(filepath)
                        self.load_indicator_file(filepath, 'VXV')
                        self.stats['files_processed'] += 1
                    elif any(ind in filename for ind in ['CNN', 'Buffett', 'TRIN']):
                        # These are indicators
                        if 'CNN' in filename:
                            self.processed_files.add(filepath)
                            self.load_indicator_file(filepath, 'CNN_FEAR_GREED')
                            self.stats['files_processed'] += 1
                        elif 'Buffett' in filename:
                            self.processed_files.add(filepath)
                            self.load_indicator_file(filepath, 'BUFFETT_INDICATOR')
                            self.stats['files_processed'] += 1
                        elif 'TRIN' in filename and 'daily' in filename.lower():
                            self.processed_files.add(filepath)
                            self.load_indicator_file(filepath, 'TRIN_DAILY')
                            self.stats['files_processed'] += 1
        
        # 3. Load market indicators
        logger.info("\n3. LOADING MARKET INDICATORS")
        logger.info("-"*40)
        
        # Root folder indicators
        root_indicators = {
            'NAAIM.xlsx': 'NAAIM',
            'FedReserve stance.xlsx': 'FED_STANCE',
            'CNN Fear and Greed Index.xlsx': 'CNN_FEAR_GREED',
            'Buffett Indicator.xlsx': 'BUFFETT_INDICATOR',
            'Data_Summary.xlsx': 'DATA_SUMMARY'
        }
        
        for filename, indicator_name in root_indicators.items():
            filepath = os.path.join(self.data_root, filename)
            if os.path.exists(filepath) and filepath not in self.processed_files:
                self.processed_files.add(filepath)
                self.load_indicator_file(filepath, indicator_name)
                self.stats['files_processed'] += 1
        
        # Market breadth parquet file
        market_breadth_path = os.path.join(self.data_root, 'Market_breadth($S5FI)_2011-2025.parquet')
        if os.path.exists(market_breadth_path) and market_breadth_path not in self.processed_files:
            self.processed_files.add(market_breadth_path)
            self.load_indicator_file(market_breadth_path, 'MARKET_BREADTH')
            self.stats['files_processed'] += 1
        
        # VIX_VXV_Ratio - special handling
        vix_vxv_loaded = self.load_vix_vxv_ratio()
        if vix_vxv_loaded > 0:
            self.stats['files_processed'] += 1
        
        # Daily folder indicators
        if os.path.exists(daily_folder):
            daily_indicators = {
                'TRIN_2013-2025.csv': 'TRIN_DAILY',
                'Market breadth (_ above 50SMA)_2007-2025.csv': 'MARKET_BREADTH_DAILY',
                'VXV.xlsx': 'VXV',
                'VXV': 'VXV',  # In case it doesn't have extension
                'CNN Fear and Greed Index.xlsx': 'CNN_FEAR_GREED',
                'Buffett Indicator.xlsx': 'BUFFETT_INDICATOR'
            }
            
            for filename, indicator_name in daily_indicators.items():
                filepath = os.path.join(daily_folder, filename)
                if os.path.exists(filepath) and filepath not in self.processed_files:
                    self.processed_files.add(filepath)
                    self.load_indicator_file(filepath, indicator_name)
                    self.stats['files_processed'] += 1
        
        # 4. Show final summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("MIGRATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Duplicate files avoided: {len(self.processed_files) - self.stats['files_processed']}")
        logger.info(f"Price records loaded: {self.stats['price_records']:,}")
        logger.info(f"Indicator records loaded: {self.stats['indicator_records']:,}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        # Database summary
        self.show_database_summary()
    
    def show_database_summary(self):
        """Display comprehensive database summary"""
        logger.info("\n" + "="*60)
        logger.info("DATABASE SUMMARY")
        logger.info("="*60)
        
        # Price data summary
        query = """
        SELECT s.ticker, p.timeframe, COUNT(*) as records,
               MIN(p.datetime) as start_date, MAX(p.datetime) as end_date
        FROM price_data p
        JOIN symbol s ON p.symbol_id = s.symbol_id
        GROUP BY s.ticker, p.timeframe
        ORDER BY s.ticker, 
                 CASE p.timeframe 
                    WHEN '1min' THEN 1
                    WHEN '5min' THEN 2
                    WHEN '15min' THEN 3
                    WHEN '30min' THEN 4
                    WHEN '1hour' THEN 5
                    WHEN 'daily' THEN 6
                    ELSE 7
                 END
        """
        
        logger.info("\nPRICE DATA:")
        current_symbol = None
        for row in self.cursor.execute(query):
            if current_symbol != row[0]:
                current_symbol = row[0]
                logger.info(f"\n{current_symbol}:")
            logger.info(f"  {row[1]:6s}: {row[2]:>10,} records ({row[3]} to {row[4]})")
        
        # Indicator summary
        query = """
        SELECT indicator_name, COUNT(*) as records,
               MIN(datetime) as start_date, MAX(datetime) as end_date
        FROM market_indicator
        GROUP BY indicator_name
        ORDER BY indicator_name
        """
        
        logger.info("\nMARKET INDICATORS:")
        for row in self.cursor.execute(query):
            logger.info(f"  {row[0]:25s}: {row[1]:>10,} records ({row[2]} to {row[3]})")
        
        # Specific checks for TRIN and VIX_VXV_RATIO
        logger.info("\n" + "-"*40)
        logger.info("CRITICAL DATA VERIFICATION:")
        
        # Check TRIN
        self.cursor.execute("""
            SELECT COUNT(*) FROM price_data p
            JOIN symbol s ON p.symbol_id = s.symbol_id
            WHERE s.ticker = 'TRIN' AND p.timeframe = '1min'
        """)
        trin_count = self.cursor.fetchone()[0]
        logger.info(f"  TRIN 1min records: {trin_count:,}")
        
        # Check VIX_VXV_RATIO
        self.cursor.execute("""
            SELECT COUNT(*), MIN(datetime), MAX(datetime)
            FROM market_indicator
            WHERE indicator_name = 'VIX_VXV_RATIO'
        """)
        vix_count, vix_min, vix_max = self.cursor.fetchone()
        logger.info(f"  VIX_VXV_RATIO records: {vix_count:,} ({vix_min} to {vix_max})")
        
        # Check FED_STANCE
        self.cursor.execute("""
            SELECT COUNT(*), MIN(datetime), MAX(datetime)
            FROM market_indicator
            WHERE indicator_name = 'FED_STANCE'
        """)
        fed_count, fed_min, fed_max = self.cursor.fetchone()
        logger.info(f"  FED_STANCE records: {fed_count:,} ({fed_min} to {fed_max})")
        
        # Total counts
        self.cursor.execute("SELECT COUNT(*) FROM price_data")
        total_price = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM market_indicator")
        total_indicators = self.cursor.fetchone()[0]
        
        # Get database file size
        db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # Convert to MB
        
        logger.info(f"\nTOTAL RECORDS: {total_price + total_indicators:,}")
        logger.info(f"DATABASE SIZE: {db_size:.1f} MB")

def main():
    """Main execution function"""
    migrator = DataMigrator(DB_PATH, DATA_ROOT)
    
    try:
        migrator.connect()
        migrator.create_schema()
        migrator.migrate_all_data()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        migrator.close()

if __name__ == "__main__":
    main()
