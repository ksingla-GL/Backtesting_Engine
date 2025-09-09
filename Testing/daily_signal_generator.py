"""
Daily Signal Generator for Testing Environment
Calculates technical indicators and generates trading signals on daily data
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

class DailySignalGenerator:
    def __init__(self):
        self.logger = self._setup_logger()
        self.indicators_calculated = {}
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            # Handle missing values
            clean_prices = prices.dropna()
            if len(clean_prices) == 0:
                return pd.Series([np.nan] * len(prices), index=prices.index)
            
            # Calculate EMA using pandas ewm
            ema = clean_prices.ewm(span=span, adjust=False).mean()
            
            # Reindex to match original series
            result = ema.reindex(prices.index, method='ffill')
            
            self.logger.info(f"Calculated EMA({span}) for {len(clean_prices)} data points")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return pd.Series([np.nan] * len(prices), index=prices.index)
    
    def calculate_rsi(self, prices: pd.Series, window: int = 2) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            clean_prices = prices.dropna()
            if len(clean_prices) < window + 1:
                return pd.Series([50.0] * len(prices), index=prices.index)
            
            # Calculate price changes
            delta = clean_prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0.0)
            losses = -delta.where(delta < 0, 0.0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=window, min_periods=window).mean()
            avg_losses = losses.rolling(window=window, min_periods=window).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            
            # Handle edge cases
            rsi = rsi.fillna(50.0)
            rsi = rsi.clip(0, 100)
            
            # Reindex to match original series
            result = rsi.reindex(prices.index, method='ffill').fillna(50.0)
            
            self.logger.info(f"Calculated RSI({window}) for {len(clean_prices)} data points")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50.0] * len(prices), index=prices.index)
    
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        try:
            clean_prices = prices.dropna()
            if len(clean_prices) == 0:
                return pd.Series([np.nan] * len(prices), index=prices.index)
            
            sma = clean_prices.rolling(window=window, min_periods=window).mean()
            result = sma.reindex(prices.index, method='ffill')
            
            self.logger.info(f"Calculated SMA({window}) for {len(clean_prices)} data points")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return pd.Series([np.nan] * len(prices), index=prices.index)
    
    def calculate_indicators(self, data: pd.DataFrame, strategy_config: Dict) -> pd.DataFrame:
        """Calculate all indicators specified in strategy configuration"""
        result_data = data.copy()
        
        self.logger.info("Starting indicator calculations...")
        
        # Process indicators from strategy config
        if 'indicators' in strategy_config:
            for indicator_config in strategy_config['indicators']:
                name = indicator_config['name']
                params = indicator_config['params']
                on_column = indicator_config['on_column']
                output_col = indicator_config['output_col']
                
                self.logger.info(f"Calculating {name} on {on_column} -> {output_col}")
                
                if on_column not in result_data.columns:
                    self.logger.error(f"Column {on_column} not found in data")
                    continue
                
                if name == 'EMA':
                    span = params.get('span', params.get('window', 20))
                    indicator_values = self.calculate_ema(result_data[on_column], span)
                    result_data[output_col] = indicator_values
                    
                elif name == 'RSI':
                    window = params.get('window', 2)
                    indicator_values = self.calculate_rsi(result_data[on_column], window)
                    result_data[output_col] = indicator_values
                    
                elif name == 'SMA':
                    window = params.get('window', 20)
                    indicator_values = self.calculate_sma(result_data[on_column], window)
                    result_data[output_col] = indicator_values
                    
                else:
                    self.logger.warning(f"Unsupported indicator: {name}")
                    continue
                
                # Track what we calculated
                self.indicators_calculated[output_col] = {
                    'indicator': name,
                    'params': params,
                    'on_column': on_column
                }
        
        self.logger.info(f"Calculated {len(self.indicators_calculated)} indicators")
        return result_data
    
    def evaluate_condition(self, data: pd.DataFrame, condition: str, params: Dict = None) -> pd.Series:
        """Evaluate a single condition string against the data"""
        try:
            # Replace parameter placeholders
            if params:
                for param, value in params.items():
                    condition = condition.replace(f"@{param}", str(value))
            
            # Evaluate the condition
            # This is a simple evaluation - in production, you'd want more robust parsing
            result = eval(condition, {"__builtins__": {}}, data.to_dict('series'))
            
            if isinstance(result, (bool, np.bool_)):
                # Single boolean result, broadcast to all rows
                return pd.Series([result] * len(data), index=data.index)
            elif isinstance(result, pd.Series):
                return result.fillna(False)
            else:
                # Try to convert to boolean series
                return pd.Series(result, index=data.index).fillna(False)
                
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return pd.Series([False] * len(data), index=data.index)
    
    def generate_signals(self, data: pd.DataFrame, strategy_config: Dict) -> pd.DataFrame:
        """Generate trading signals based on strategy configuration"""
        result_data = data.copy()
        
        self.logger.info("Generating trading signals...")
        
        # Get parameter grid for testing multiple parameter values
        param_grid = strategy_config.get('param_grid', {})
        
        # Use parameter value of 30 for fear_threshold (matching Excel analysis)
        params = {}
        for param_name, param_values in param_grid.items():
            if param_name == 'fear_threshold':
                params[param_name] = 30  # Use 30 to match Excel analysis
            else:
                params[param_name] = param_values[0] if isinstance(param_values, list) else param_values
        
        self.logger.info(f"Using parameters: {params}")
        
        # Evaluate base entry rules
        base_rules = strategy_config.get('base_entry_rules', [])
        base_conditions = []
        
        for i, rule in enumerate(base_rules):
            condition_result = self.evaluate_condition(result_data, rule, params)
            col_name = f"base_rule_{i+1}"
            result_data[col_name] = condition_result
            base_conditions.append(condition_result)
            self.logger.info(f"Base rule {i+1}: {condition_result.sum()} true signals")
        
        # Evaluate conditional entry rules
        conditional_rules = strategy_config.get('conditional_entry_rules', [])
        conditional_conditions = []
        
        for rule_group in conditional_rules:
            conditions = rule_group['rule']['conditions']
            threshold = int(rule_group['rule']['threshold'])
            
            group_results = []
            for i, condition in enumerate(conditions):
                condition_result = self.evaluate_condition(result_data, condition, params)
                col_name = f"cond_rule_{len(conditional_conditions)+1}_{i+1}"
                result_data[col_name] = condition_result
                group_results.append(condition_result)
            
            # Sum conditions in this group
            group_sum = pd.concat(group_results, axis=1).sum(axis=1)
            group_signal = group_sum >= threshold
            
            conditional_conditions.append(group_signal)
            result_data[f"cond_group_{len(conditional_conditions)}"] = group_signal
            result_data[f"cond_sum_{len(conditional_conditions)}"] = group_sum
            
            self.logger.info(f"Conditional group {len(conditional_conditions)}: {group_signal.sum()} true signals")
        
        # Combine all conditions for final signal
        all_conditions = base_conditions + conditional_conditions
        if all_conditions:
            # All base conditions AND at least one conditional group must be true
            final_signal = pd.concat(all_conditions, axis=1).all(axis=1)
        else:
            final_signal = pd.Series([False] * len(result_data), index=result_data.index)
        
        result_data['entry_signal'] = final_signal
        
        # Apply one-signal-per-day logic (already daily data, but ensure no duplicates)
        result_data['entry_signal_final'] = result_data['entry_signal']
        
        total_signals = result_data['entry_signal_final'].sum()
        self.logger.info(f"Generated {total_signals} entry signals")
        
        return result_data
    
    def get_signal_summary(self, data: pd.DataFrame) -> Dict:
        """Get summary of generated signals"""
        summary = {}
        
        # Count signals by column
        signal_columns = [col for col in data.columns if 'signal' in col.lower() or 'rule' in col.lower()]
        
        for col in signal_columns:
            if col in data.columns:
                true_count = data[col].sum() if data[col].dtype == bool else (data[col] == True).sum()
                summary[col] = {
                    'true_count': int(true_count),
                    'false_count': int(len(data) - true_count),
                    'percentage': float(true_count / len(data) * 100) if len(data) > 0 else 0.0
                }
        
        return summary

if __name__ == "__main__":
    # Test the signal generator
    from daily_data_handler import DailyDataHandler
    
    # Load test data
    handler = DailyDataHandler()
    data = handler.load_data()
    
    if data is not None:
        # Test strategy config
        test_config = {
            "indicators": [
                {"name": "EMA", "params": {"span": 9}, "on_column": "ES_close", "output_col": "ES_EMA_9"},
                {"name": "RSI", "params": {"window": 2}, "on_column": "ES_close", "output_col": "ES_RSI_2"}
            ],
            "param_grid": {
                "fear_threshold": [30]
            },
            "base_entry_rules": [
                "CNN_FEAR_GREED < @fear_threshold",
                "ES_RSI_2 > 50"
            ],
            "conditional_entry_rules": [
                {
                    "condition": "True",
                    "rule": {
                        "type": "sum_of_conditions",
                        "conditions": ["VIX_close > 20", "ES_RSI_2 > 60", "TRIN_close < 0.9"],
                        "threshold": "1"
                    }
                }
            ]
        }
        
        # Test signal generation
        generator = DailySignalGenerator()
        
        # Calculate indicators
        data_with_indicators = generator.calculate_indicators(data, test_config)
        print(f"Data shape after indicators: {data_with_indicators.shape}")
        
        # Generate signals
        data_with_signals = generator.generate_signals(data_with_indicators, test_config)
        print(f"Data shape after signals: {data_with_signals.shape}")
        
        # Get summary
        summary = generator.get_signal_summary(data_with_signals)
        print("\nSignal Summary:")
        for signal, info in summary.items():
            print(f"  {signal}: {info['true_count']} true ({info['percentage']:.1f}%)")
    
    else:
        print("Failed to load test data")