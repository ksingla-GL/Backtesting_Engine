"""
Daily Data Handler for Testing Environment
Loads and manages daily price data for backtesting
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, date
import logging

class DailyDataHandler:
    def __init__(self):
        self.data = None
        self.symbols = ['ES', 'VIX', 'TRIN']
        self.logger = self._setup_logger()
        self.us_holidays = self._get_us_market_holidays()
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _get_us_market_holidays(self):
        """Get list of US market holidays for common years"""
        holidays = set()
        
        # Common years to cover typical backtesting periods
        for year in range(2020, 2026):
            # New Year's Day (observed)
            new_years = date(year, 1, 1)
            if new_years.weekday() == 5:  # Saturday
                holidays.add(date(year, 1, 3))  # Monday
            elif new_years.weekday() == 6:  # Sunday
                holidays.add(date(year, 1, 2))  # Monday
            else:
                holidays.add(new_years)
            
            # Martin Luther King Jr. Day (3rd Monday in January)
            holidays.add(self._get_nth_weekday(year, 1, 0, 3))
            
            # Presidents Day (3rd Monday in February)
            holidays.add(self._get_nth_weekday(year, 2, 0, 3))
            
            # Good Friday (Friday before Easter) - approximate
            easter = self._get_easter_date(year)
            good_friday = easter - pd.Timedelta(days=2)
            holidays.add(good_friday.date())
            
            # Memorial Day (last Monday in May)
            holidays.add(self._get_last_weekday(year, 5, 0))
            
            # Juneteenth (June 19, observed if weekend)
            juneteenth = date(year, 6, 19)
            if juneteenth.weekday() == 5:  # Saturday
                holidays.add(date(year, 6, 18))  # Friday
            elif juneteenth.weekday() == 6:  # Sunday
                holidays.add(date(year, 6, 20))  # Monday
            else:
                holidays.add(juneteenth)
            
            # Independence Day (July 4, observed if weekend)
            july4 = date(year, 7, 4)
            if july4.weekday() == 5:  # Saturday
                holidays.add(date(year, 7, 3))  # Friday
            elif july4.weekday() == 6:  # Sunday
                holidays.add(date(year, 7, 5))  # Monday
            else:
                holidays.add(july4)
            
            # Labor Day (1st Monday in September)
            holidays.add(self._get_nth_weekday(year, 9, 0, 1))
            
            # Thanksgiving (4th Thursday in November)
            holidays.add(self._get_nth_weekday(year, 11, 3, 4))
            
            # Christmas Day (December 25, observed if weekend)
            christmas = date(year, 12, 25)
            if christmas.weekday() == 5:  # Saturday
                holidays.add(date(year, 12, 24))  # Friday
            elif christmas.weekday() == 6:  # Sunday
                holidays.add(date(year, 12, 26))  # Monday
            else:
                holidays.add(christmas)
        
        return holidays
    
    def _get_nth_weekday(self, year, month, weekday, n):
        """Get the nth occurrence of weekday in given month/year"""
        first_day = date(year, month, 1)
        # Find first occurrence of weekday
        days_ahead = weekday - first_day.weekday()
        if days_ahead < 0:
            days_ahead += 7
        first_occurrence = pd.Timestamp(first_day) + pd.Timedelta(days=days_ahead)
        # Get nth occurrence
        nth_occurrence = first_occurrence + pd.Timedelta(days=7 * (n - 1))
        return nth_occurrence.date()
    
    def _get_last_weekday(self, year, month, weekday):
        """Get the last occurrence of weekday in given month/year"""
        # Start from last day of month and work backwards
        if month == 12:
            last_day = pd.Timestamp(date(year + 1, 1, 1)) - pd.Timedelta(days=1)
        else:
            last_day = pd.Timestamp(date(year, month + 1, 1)) - pd.Timedelta(days=1)
        
        days_back = (last_day.weekday() - weekday) % 7
        last_occurrence = last_day - pd.Timedelta(days=days_back)
        return last_occurrence.date()
    
    def _get_easter_date(self, year):
        """Calculate Easter date using algorithm"""
        # Using the algorithm for Western Easter
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        n = (h + l - 7 * m + 114) // 31
        p = (h + l - 7 * m + 114) % 31
        return pd.Timestamp(year, n, p + 1)
    
    def _is_weekend(self, date_obj):
        """Check if date is weekend (Saturday=5, Sunday=6)"""
        return date_obj.weekday() >= 5
    
    def _is_holiday(self, date_obj):
        """Check if date is a US market holiday"""
        return date_obj.date() in self.us_holidays
    
    def _is_trading_day(self, date_obj):
        """Check if date is a valid trading day (not weekend or holiday)"""
        return not (self._is_weekend(date_obj) or self._is_holiday(date_obj))
        
    def load_data(self, file_path='top_v2_daily_data_2022.csv', skip_weekends_holidays=False):
        """Load daily data from CSV file with optional weekend/holiday filtering"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Data file not found: {file_path}")
                return None
                
            self.data = pd.read_csv(file_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
            
            initial_count = len(self.data)
            
            # Filter out weekends and holidays if requested
            if skip_weekends_holidays:
                trading_days_mask = self.data['date'].apply(self._is_trading_day)
                filtered_data = self.data[trading_days_mask].copy()
                
                removed_count = len(self.data) - len(filtered_data)
                self.data = filtered_data.reset_index(drop=True)
                
                self.logger.info(f"Filtered out {removed_count} non-trading days (weekends/holidays)")
            
            self.logger.info(f"Loaded {len(self.data)} trading days of data")
            self.logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
    
    def get_data(self):
        """Get the loaded data"""
        return self.data
    
    def get_date_range(self):
        """Get the date range of loaded data"""
        if self.data is None:
            return None, None
        return self.data['date'].min(), self.data['date'].max()
    
    def get_data_summary(self):
        """Get summary of data completeness"""
        if self.data is None:
            return None
            
        summary = {}
        for col in self.data.columns:
            if col != 'date':
                non_null = self.data[col].notna().sum()
                total = len(self.data)
                pct = (non_null / total) * 100 if total > 0 else 0
                summary[col] = {
                    'non_null': non_null,
                    'total': total,
                    'completeness_pct': pct
                }
        
        return summary
    
    def validate_data(self, required_columns):
        """Validate that required columns exist and have data"""
        if self.data is None:
            self.logger.error("No data loaded")
            return False
            
        missing_columns = []
        for col in required_columns:
            if col not in self.data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for reasonable amount of data in required columns
        for col in required_columns:
            non_null_pct = (self.data[col].notna().sum() / len(self.data)) * 100
            if non_null_pct < 50:  # Less than 50% data availability
                self.logger.warning(f"Column {col} has only {non_null_pct:.1f}% data availability")
        
        self.logger.info("Data validation passed")
        return True
    
    def get_column_data(self, column_name):
        """Get data for a specific column"""
        if self.data is None or column_name not in self.data.columns:
            return None
        return self.data[column_name]
    
    def forward_fill_missing_data(self, columns=None):
        """Forward fill missing data for specified columns"""
        if self.data is None:
            return
            
        if columns is None:
            # Forward fill all numeric columns except date
            columns = [col for col in self.data.columns if col != 'date']
        
        for col in columns:
            if col in self.data.columns:
                before_fill = self.data[col].isna().sum()
                self.data[col] = self.data[col].fillna(method='ffill')
                after_fill = self.data[col].isna().sum()
                
                if before_fill > after_fill:
                    self.logger.info(f"Forward filled {before_fill - after_fill} missing values in {col}")
    
    def add_column(self, column_name, values):
        """Add a new column to the data"""
        if self.data is None:
            self.logger.error("No data loaded")
            return False
            
        if len(values) != len(self.data):
            self.logger.error(f"Length mismatch: data has {len(self.data)} rows, values has {len(values)}")
            return False
            
        self.data[column_name] = values
        self.logger.info(f"Added column: {column_name}")
        return True
    
    def get_data_for_date_range(self, start_date, end_date):
        """Get data for specific date range"""
        if self.data is None:
            return None
            
        mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
        return self.data[mask].copy()
    
    def export_data(self, file_path, columns=None):
        """Export data to CSV"""
        if self.data is None:
            self.logger.error("No data to export")
            return False
            
        try:
            if columns:
                export_data = self.data[columns]
            else:
                export_data = self.data
                
            export_data.to_csv(file_path, index=False)
            self.logger.info(f"Data exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
    
    def get_filtered_dates_info(self, file_path='top_v2_daily_data_2022.csv'):
        """Get information about which dates were filtered out"""
        try:
            # Load raw data without filtering
            raw_data = pd.read_csv(file_path)
            raw_data['date'] = pd.to_datetime(raw_data['date'])
            
            weekends = []
            holidays = []
            
            for _, row in raw_data.iterrows():
                date_obj = row['date']
                if self._is_weekend(date_obj):
                    weekends.append(date_obj.strftime('%Y-%m-%d'))
                elif self._is_holiday(date_obj):
                    holidays.append(date_obj.strftime('%Y-%m-%d'))
            
            return {
                'weekends_filtered': weekends,
                'holidays_filtered': holidays,
                'total_filtered': len(weekends) + len(holidays)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting filtered dates info: {e}")
            return None

if __name__ == "__main__":
    # Test the data handler
    handler = DailyDataHandler()
    data = handler.load_data()
    
    if data is not None:
        print("Data loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        summary = handler.get_data_summary()
        print("\nData Summary:")
        for col, info in summary.items():
            print(f"  {col}: {info['completeness_pct']:.1f}% complete ({info['non_null']}/{info['total']})")
    else:
        print("Failed to load data")