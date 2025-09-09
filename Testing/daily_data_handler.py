"""
Daily Data Handler for Testing Environment
Loads and manages daily price data for backtesting
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

class DailyDataHandler:
    def __init__(self):
        self.data = None
        self.symbols = ['ES', 'VIX', 'TRIN']
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
        
    def load_data(self, file_path='top_v2_daily_data_2022.csv'):
        """Load daily data from CSV file"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Data file not found: {file_path}")
                return None
                
            self.data = pd.read_csv(file_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(self.data)} days of data")
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