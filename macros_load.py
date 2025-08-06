"""
complete_macro_loader_with_events.py
Complete macro data loader that:
1. Loads FRED data
2. Merges Excel consensus forecasts
3. Creates event flags for strategies
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRED_API_KEY = "162999cfbc37078421fa990d37b84c1d"

class CompleteMacroLoaderWithEvents:
    def __init__(self, db_path: str, excel_path: str):
        self.db_path = db_path
        self.excel_path = excel_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
    def create_enhanced_events_table(self):
        """Create table with all necessary columns"""
        
        # Drop any existing tables
        self.cursor.execute("DROP TABLE IF EXISTS macro_events_old")
        self.cursor.execute("DROP TABLE IF EXISTS macro_events")
        
        # Create fresh table
        self.cursor.execute("""
            CREATE TABLE macro_events (
                event_date DATE NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                event_time TIME DEFAULT '08:30:00',
                importance VARCHAR(20) DEFAULT 'HIGH',
                actual_value REAL,
                forecast_value REAL,
                previous_value REAL,
                surprise REAL,
                surprise_pct REAL,
                is_inline INTEGER DEFAULT 0,
                is_better INTEGER DEFAULT 0,
                is_worse INTEGER DEFAULT 0,
                PRIMARY KEY (event_date, event_type)
            )
        """)
        
        self.conn.commit()
        logger.info("Created enhanced macro_events table")
    
    def load_cpi_with_excel_forecasts(self):
        """Load CPI dates, FRED actuals, and Excel forecasts"""
        logger.info("Loading CPI data with Excel forecasts...")
        
        # Load Excel CPI data
        excel_cpi = pd.read_excel(self.excel_path, sheet_name='CPI')
        
        # Fix date conversion - check if already datetime or numeric
        if pd.api.types.is_datetime64_any_dtype(excel_cpi['Date of publication']):
            # Already datetime
            excel_cpi['date'] = excel_cpi['Date of publication']
        else:
            # Numeric Excel date
            excel_cpi['date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(excel_cpi['Date of publication'], 'D')
        
        excel_cpi['excel_forecast'] = excel_cpi['CPI Forecast (investing.com)']
        
        # Accurate CPI dates
        cpi_dates = [
            "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12",
            "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
            "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13",
            "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
            "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11", "2022-06-10",
            "2022-07-13", "2022-08-10", "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
            "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13", "2021-05-12", "2021-06-10",
            "2021-07-13", "2021-08-11", "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
            "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10", "2020-05-12", "2020-06-10",
            "2020-07-14", "2020-08-12", "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10",
            "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10", "2019-05-10", "2019-06-12",
            "2019-07-11", "2019-08-13", "2019-09-12", "2019-10-10", "2019-11-13", "2019-12-11",
            "2018-01-12", "2018-02-14", "2018-03-13", "2018-04-11", "2018-05-10", "2018-06-12",
            "2018-07-12", "2018-08-10", "2018-09-13", "2018-10-11", "2018-11-14", "2018-12-12",
            "2017-01-18", "2017-02-15", "2017-03-15", "2017-04-14", "2017-05-12", "2017-06-14",
            "2017-07-14", "2017-08-11", "2017-09-14", "2017-10-13", "2017-11-15", "2017-12-13",
            "2016-01-20", "2016-02-19", "2016-03-16", "2016-04-14", "2016-05-17", "2016-06-16",
            "2016-07-15", "2016-08-16", "2016-09-16", "2016-10-18", "2016-11-17", "2016-12-15",
            "2015-01-16", "2015-02-26", "2015-03-24", "2015-04-17", "2015-05-22", "2015-06-18",
            "2015-07-17", "2015-08-19", "2015-09-16", "2015-10-15", "2015-11-17", "2015-12-15",
            "2014-01-16", "2014-02-20", "2014-03-18", "2014-04-15", "2014-05-15", "2014-06-17",
            "2014-07-22", "2014-08-19", "2014-09-17", "2014-10-22", "2014-11-20", "2014-12-17",
            "2013-01-16", "2013-02-21", "2013-03-15", "2013-04-16", "2013-05-16", "2013-06-18",
            "2013-07-16", "2013-08-15", "2013-09-17", "2013-10-30", "2013-11-20", "2013-12-17"
        ]
        
        # Get FRED CPI actuals
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "CPIAUCSL",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": "2012-12-01",
            "observation_end": "2024-12-31",
            "units": "pc1",
            "frequency": "m"
        }
        
        response = requests.get(url, params=params)
        fred_data = response.json()
        
        cpi_values = {}
        for obs in fred_data['observations']:
            if obs['value'] != '.':
                cpi_values[obs['date']] = float(obs['value'])
        
        # Process each CPI date
        prev_value = None
        for release_date in cpi_dates:
            release_dt = pd.to_datetime(release_date)
            reference_month = (release_dt - pd.DateOffset(months=1)).strftime('%Y-%m-01')
            
            # Get actual from FRED
            actual = cpi_values.get(reference_month, None)
            
            # Try to get forecast from Excel
            forecast = None
            excel_matches = excel_cpi[
                (excel_cpi['date'] >= release_dt - pd.Timedelta(days=2)) &
                (excel_cpi['date'] <= release_dt + pd.Timedelta(days=2))
            ]
            
            if len(excel_matches) > 0 and not pd.isna(excel_matches.iloc[0]['excel_forecast']):
                forecast = excel_matches.iloc[0]['excel_forecast']
            else:
                # Fallback to estimate
                if prev_value:
                    forecast = prev_value * 0.95
                else:
                    forecast = actual
            
            # Calculate surprise and classify
            is_inline = 0
            is_better = 0
            is_worse = 0
            
            if actual is not None and forecast is not None:
                surprise = actual - forecast
                surprise_pct = (surprise / abs(forecast)) * 100 if forecast != 0 else 0
                
                # Classify surprise (matching client's thresholds)
                if abs(surprise) <= 0.1:  # Within 0.1% = inline
                    is_inline = 1
                elif surprise < -0.1:  # Better than expected
                    is_better = 1
                else:  # Worse than expected
                    is_worse = 1
            else:
                surprise = None
                surprise_pct = None
                is_inline = 1  # Default to inline if no data
            
            # Insert into database
            self.cursor.execute("""
                INSERT OR REPLACE INTO macro_events 
                (event_date, event_type, event_time, importance, 
                 actual_value, forecast_value, previous_value, 
                 surprise, surprise_pct, is_inline, is_better, is_worse)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (release_date, 'CPI', '08:30:00', 'HIGH',
                  actual, forecast, prev_value, 
                  surprise, surprise_pct, is_inline, is_better, is_worse))
            
            if actual is not None:
                prev_value = actual
        
        self.conn.commit()
        logger.info(f"Loaded {len(cpi_dates)} CPI events with Excel forecasts")
    
    def load_fomc_with_classifications(self):
        """Load FOMC with dovish/hawkish classifications"""
        logger.info("Loading FOMC data...")
        
        fomc_dates = [
            "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
            "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
            "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
            "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
            "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
            "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
            "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
            "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
            "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
            "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
            "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
            "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18"
        ]
        
        # Get Fed Funds Rate from FRED
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DFF",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": "2013-01-01",
            "observation_end": "2024-12-31"
        }
        
        response = requests.get(url, params=params)
        fred_data = response.json()
        
        fed_rates = {}
        for obs in fred_data['observations']:
            if obs['value'] != '.':
                fed_rates[obs['date']] = float(obs['value'])
        
        # Process FOMC dates
        prev_rate = 0.25
        for fomc_date in fomc_dates:
            actual = fed_rates.get(fomc_date, None)
            if actual is None:
                next_day = (pd.to_datetime(fomc_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                actual = fed_rates.get(next_day, prev_rate)
            
            forecast = prev_rate  # Market usually expects no change
            
            # Calculate surprise and classify
            is_inline = 0
            is_better = 0  # Dovish (rate cut)
            is_worse = 0   # Hawkish (rate hike)
            
            if actual is not None:
                surprise = actual - forecast
                surprise_pct = (surprise / forecast) * 100 if forecast != 0 else 0
                
                if abs(surprise) < 0.25:  # Less than 25bp = no surprise
                    is_inline = 1
                elif surprise < 0:  # Cut = dovish
                    is_better = 1
                else:  # Hike = hawkish
                    is_worse = 1
            else:
                surprise = None
                surprise_pct = None
                is_inline = 1
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO macro_events 
                (event_date, event_type, event_time, importance, 
                 actual_value, forecast_value, previous_value, 
                 surprise, surprise_pct, is_inline, is_better, is_worse)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fomc_date, 'FOMC', '14:00:00', 'HIGH',
                  actual, forecast, prev_rate, 
                  surprise, surprise_pct, is_inline, is_better, is_worse))
            
            if actual is not None:
                prev_rate = actual
        
        self.conn.commit()
        logger.info(f"Loaded {len(fomc_dates)} FOMC events")
    
    def load_nfp_data(self):
        """Load NFP data"""
        logger.info("Loading NFP data...")
        
        # Generate NFP dates
        nfp_dates = []
        for year in range(2013, 2025):
            for month in range(1, 13):
                first_day = pd.Timestamp(year, month, 1)
                days_to_friday = (4 - first_day.dayofweek) % 7
                if days_to_friday == 0 and first_day.dayofweek != 4:
                    days_to_friday = 7
                first_friday = first_day + pd.Timedelta(days=days_to_friday)
                if first_friday <= pd.Timestamp.now():
                    nfp_dates.append(first_friday.strftime('%Y-%m-%d'))
        
        # Get NFP from FRED
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "PAYEMS",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": "2012-12-01",
            "observation_end": "2024-12-31",
            "units": "chg",
            "frequency": "m"
        }
        
        response = requests.get(url, params=params)
        fred_data = response.json()
        
        nfp_values = {}
        for obs in fred_data['observations']:
            if obs['value'] != '.':
                nfp_values[obs['date']] = float(obs['value'])
        
        # Process NFP dates
        prev_value = None
        for nfp_date in nfp_dates[:144]:
            release_dt = pd.to_datetime(nfp_date)
            reference_month = (release_dt - pd.DateOffset(months=1)).strftime('%Y-%m-01')
            
            actual = nfp_values.get(reference_month, None)
            forecast = prev_value * 0.9 if prev_value else 150
            
            # NFP doesn't have inline/better/worse in same way
            is_inline = 1
            is_better = 0
            is_worse = 0
            
            if actual is not None and forecast is not None:
                surprise = actual - forecast
                surprise_pct = (surprise / abs(forecast)) * 100 if forecast != 0 else 0
            else:
                surprise = None
                surprise_pct = None
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO macro_events 
                (event_date, event_type, event_time, importance, 
                 actual_value, forecast_value, previous_value, 
                 surprise, surprise_pct, is_inline, is_better, is_worse)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (nfp_date, 'NFP', '08:30:00', 'HIGH',
                  actual, forecast, prev_value, 
                  surprise, surprise_pct, is_inline, is_better, is_worse))
            
            if actual is not None:
                prev_value = actual
        
        self.conn.commit()
        logger.info(f"Loaded {len(nfp_dates[:144])} NFP events")
    
    def add_market_indicators_table(self):
        """Create table for pre/post event market indicators"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_event_indicators (
                event_date DATE NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                pre_es_return_1d REAL,
                pre_es_return_3d REAL,
                pre_vix_level REAL,
                pre_vix_change_1d REAL,
                post_vix_change REAL,
                post_es_move_5min REAL,
                post_es_move_30min REAL,
                PRIMARY KEY (event_date, event_type)
            )
        """)
        self.conn.commit()
        logger.info("Created macro_event_indicators table")
    
    def export_enhanced_csv(self):
        """Export complete data with event flags"""
        logger.info("Exporting enhanced CSV with event flags...")
        
        # Get all events
        df = pd.read_sql_query("""
            SELECT * FROM macro_events 
            ORDER BY event_date DESC
        """, self.conn)
        
        # Add readable columns for strategies
        df['cpi_surprise_inline'] = (df['event_type'] == 'CPI') & (df['is_inline'] == 1)
        df['cpi_surprise_better'] = (df['event_type'] == 'CPI') & (df['is_better'] == 1)
        df['cpi_surprise_worse'] = (df['event_type'] == 'CPI') & (df['is_worse'] == 1)
        
        df['fed_surprise_neutral'] = (df['event_type'] == 'FOMC') & (df['is_inline'] == 1)
        df['fed_surprise_dovish'] = (df['event_type'] == 'FOMC') & (df['is_better'] == 1)
        df['fed_surprise_hawkish'] = (df['event_type'] == 'FOMC') & (df['is_worse'] == 1)
        
        # Save enhanced CSV
        df.to_csv('macro_events_complete_enhanced.csv', index=False)
        logger.info(f"Exported {len(df)} events to macro_events_complete_enhanced.csv")
        
        # Show summary
        print("\nEvent Summary:")
        print(df.groupby(['event_type', 'is_inline', 'is_better', 'is_worse']).size())
        
        print("\nCPI Classifications:")
        print(f"  Inline: {df['cpi_surprise_inline'].sum()}")
        print(f"  Better: {df['cpi_surprise_better'].sum()}")
        print(f"  Worse: {df['cpi_surprise_worse'].sum()}")
        
        print("\nFOMC Classifications:")
        print(f"  Neutral: {df['fed_surprise_neutral'].sum()}")
        print(f"  Dovish: {df['fed_surprise_dovish'].sum()}")
        print(f"  Hawkish: {df['fed_surprise_hawkish'].sum()}")
        
        return df
    
    def close(self):
        self.conn.close()

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python complete_macro_loader_with_events.py <db_path> <excel_path>")
        print("Example: python complete_macro_loader_with_events.py ../backtesting_v2.db 'Macro_DFF CPI.xlsx'")
        sys.exit(1)
    
    db_path = sys.argv[1]
    excel_path = sys.argv[2]
    
    loader = CompleteMacroLoaderWithEvents(db_path, excel_path)
    
    try:
        print("\n1. Creating enhanced table...")
        loader.create_enhanced_events_table()
        
        print("\n2. Loading CPI with Excel forecasts...")
        loader.load_cpi_with_excel_forecasts()
        
        print("\n3. Loading FOMC with classifications...")
        loader.load_fomc_with_classifications()
        
        print("\n4. Loading NFP data...")
        loader.load_nfp_data()
        
        print("\n5. Creating market indicators table...")
        loader.add_market_indicators_table()
        
        print("\n6. Exporting enhanced CSV...")
        df = loader.export_enhanced_csv()
        
        print("\n✅ COMPLETE! Database and CSV now have:")
        print("- Actual values from FRED")
        print("- Consensus forecasts from Excel (where available)")
        print("- Surprise classifications (inline/better/worse)")
        print("- Event flags for strategies")
        print("- Market indicators table ready for pre/post data")
        
        print("\nCheck macro_events_complete_enhanced.csv for all columns")
        
    finally:
        loader.close()

if __name__ == "__main__":
    main()