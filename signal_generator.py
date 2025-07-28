"""
signal_generator.py
Module to dynamically calculate ALL technical indicators and generate trading signals.
Enhanced to support both long and short positions.
"""
import pandas as pd
import numpy as np
import logging
import ta
import re

def calculate_indicator(df, indicator_config):
    """Calculates a single technical indicator based on config."""
    name = indicator_config['name']
    params = indicator_config.get('params', {})
    on_col = indicator_config.get('on_column')
    out_col = indicator_config['output_col']
    
    if out_col in df.columns and df[out_col].notna().all():
        return df

    logging.info(f"Calculating {name} -> {out_col}")
    
    if name not in ['IsUpDay', 'MACD'] and on_col and on_col not in df.columns:
        logging.error(f"Input column '{on_col}' not found for indicator {name}")
        df[out_col] = np.nan
        return df
    
    if name not in ['IsUpDay', 'MACD'] and on_col and df[on_col].isna().all():
        logging.error(f"Input column '{on_col}' is all NaN for indicator {name}")
        df[out_col] = np.nan
        return df
    
    try:
        if name == 'SMA':
            window = params.get('window', 50)
            daily_close = df[on_col].resample('D').last().dropna()
            daily_sma = daily_close.rolling(window=window, min_periods=window).mean()
            df[out_col] = daily_sma.reindex(df.index, method='ffill').ffill()
                
        elif name == 'EMA':
            span = params.get('span', params.get('window'))
            daily_close = df[on_col].resample('D').last().dropna()
            daily_ema = daily_close.ewm(span=span, adjust=False).mean()
            df[out_col] = daily_ema.reindex(df.index, method='ffill').ffill()
                
        elif name == 'RSI':
            window = params.get('window', 2)
            daily_close = df[on_col].resample('D').last().dropna()
            daily_rsi = ta.momentum.rsi(daily_close, window=window)
            df[out_col] = daily_rsi.reindex(df.index, method='ffill').ffill()
        
        elif name == 'RollingHigh':
            daily_data = df[on_col].resample('D').max()
            window_size = params.get('window', 10)
            rolling_data = daily_data.rolling(window=window_size, min_periods=1).max()
            df[out_col] = rolling_data.reindex(df.index, method='ffill').ffill()
        
        elif name == 'RollingLow':
            daily_data = df[on_col].resample('D').min()
            window_size = params.get('window', 10)
            rolling_data = daily_data.rolling(window=window_size, min_periods=1).min()
            df[out_col] = rolling_data.reindex(df.index, method='ffill').ffill()
        
        elif name == 'PrevDayClose':
            daily_data = df[on_col].resample('D').last()
            df[out_col] = daily_data.shift(1).reindex(df.index, method='ffill').ffill()

        elif name == 'VIXSpike':
            prev_close_col = indicator_config['prev_close_col']
            if prev_close_col not in df.columns:
                df[out_col] = np.nan
                return df
            df[out_col] = (df[on_col] / df[prev_close_col] - 1) * 100
            
        elif name == 'DeclineFromPeak':
            peak_col = indicator_config['peak_column']
            if peak_col not in df.columns or df[peak_col].isna().all():
                df[out_col] = np.nan
                return df
            df[out_col] = np.where(df[peak_col] > 0, (df[on_col] / df[peak_col] - 1) * 100, np.nan)

        elif name == 'IsUpDay':
            close_col = f"{params['instrument']}_close"
            open_col = f"{params['instrument']}_open"
            if close_col not in df.columns or open_col not in df.columns:
                 logging.error(f"Could not find {close_col} or {open_col} for IsUpDay indicator")
                 df[out_col] = False
                 return df
            df[out_col] = df[close_col] > df[open_col]

        elif name == 'MACD':
            # MACD calculation
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            signal = params.get('signal', 9)
            
            # Calculate on daily data
            daily_close = df[on_col].resample('D').last().dropna()
            
            # Calculate MACD line
            ema_fast = daily_close.ewm(span=fast, adjust=False).mean()
            ema_slow = daily_close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculate histogram
            macd_histogram = macd_line - signal_line
            
            # Reindex to 1-minute
            df[out_col] = macd_histogram.reindex(df.index, method='ffill').ffill()

        else:
            logging.warning(f"Indicator '{name}' is not recognized.")
            
    except Exception as e:
        logging.error(f"Failed to calculate indicator {name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        df[out_col] = np.nan
        
    return df

def get_columns_from_rules(rules):
    """Parses rule strings to find all column names used."""
    columns = set()
    for rule in rules:
        if isinstance(rule, str):
            columns.update(re.findall(r'\b(?<!@)([a-zA-Z_][a-zA-Z0-9_]*)\b', rule))
        elif isinstance(rule, dict) and 'conditions' in rule:
            columns.update(get_columns_from_rules(rule['conditions']))
            
    keywords = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'}
    return [col for col in columns if col not in keywords and not col.isnumeric()]

def generate_signals(df, strategy_config):
    """Calculates indicators, handles NaNs, and generates entry signals."""
    logging.info("--- Starting Signal Generation ---")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Check if this is a short strategy
    position_type = strategy_config.get('position_type', 'long')
    signal_value = -1 if position_type == 'short' else 1
    
    for ind_config in strategy_config['indicators']:
        df = calculate_indicator(df, ind_config)

    if 'base_entry_rules' in strategy_config or 'conditional_entry_rules' in strategy_config:
        base_rules = strategy_config.get('base_entry_rules', [])
        conditional_rules_list = strategy_config.get('conditional_entry_rules', [])
        all_possible_rules = base_rules + [item['rule'] for item in conditional_rules_list]
    else:
        base_rules = strategy_config.get('entry_rules', [])
        conditional_rules_list = []
        all_possible_rules = list(base_rules)

    required_columns = get_columns_from_rules(all_possible_rules)
    logging.info(f"Columns required by strategy rules: {required_columns}")
    
    # Drop NaNs after all calculations are done
    df = df.ffill().dropna(subset=required_columns)
    logging.info(f"Data cleaned. {len(df)} valid rows remaining.")
    
    if df.empty:
        logging.error("No valid data rows remaining. Cannot generate signals.")
        return df

    param_grid = strategy_config['param_grid']
    param_name = list(param_grid.keys())[0]
    param_values = param_grid[param_name]
    
    for value in param_values:
        logging.info(f"--- Generating signals for {param_name} = {value} ---")
        local_vars = {param_name: value}
        
        current_rules = list(base_rules)
        if conditional_rules_list:
            for cond_rule in conditional_rules_list:
                condition_str = cond_rule['condition'].replace(f"@{param_name}", str(value))
                if eval(condition_str):
                    current_rules.append(cond_rule['rule'])
        
        logging.info(f"Evaluating final rule set for {param_name}={value}")
        
        if not current_rules:
            df[f"entry_signal_{value}"] = 0
            continue

        all_conditions = []
        for i, rule in enumerate(current_rules):
            try:
                if isinstance(rule, str):
                    condition = df.eval(rule, local_dict=local_vars)
                    logging.info(f"  Rule {i+1:02d} (Simple) '{rule}': Met {condition.sum():,} times.")
                
                elif isinstance(rule, dict) and rule.get('type') == 'sum_of_conditions':
                    sub_conditions = []
                    for sub_rule in rule['conditions']:
                        sub_conditions.append(df.eval(sub_rule, local_dict=local_vars))
                    
                    summed_conds = pd.concat(sub_conditions, axis=1).sum(axis=1)
                    
                    threshold_str = str(rule['threshold']).replace(f"@{param_name}", str(value))
                    threshold = int(threshold_str)
                    
                    condition = (summed_conds >= threshold)
                    logging.info(f"  Rule {i+1:02d} (Cluster): '{len(rule['conditions'])} sub-rules, threshold >= {threshold}': Met {condition.sum():,} times.")

                all_conditions.append(condition)
            except Exception as e:
                logging.error(f"  Rule {i+1:02d} '{rule}': FAILED TO EVALUATE. Error: {e}")
                all_conditions.append(pd.Series(False, index=df.index))
        
        final_entry_condition = pd.concat(all_conditions, axis=1).all(axis=1)
        
        signal_col_name = f"entry_signal_{value}"
        df[signal_col_name] = np.where(final_entry_condition, signal_value, 0)
        
        # Fix: For shorts, we need to handle the one-signal-per-day logic differently
        if position_type == 'short':
            df[signal_col_name] = df[signal_col_name].groupby(df.index.date).transform(
                lambda x: (x.cumsum() == -1) & (x == -1)
            ).astype(int) * -1
        else:
            df[signal_col_name] = df[signal_col_name].groupby(df.index.date).transform(
                lambda x: (x.cumsum() == 1) & (x == 1)
            ).astype(int)
        
        logging.info(f"Generated {abs(df[signal_col_name]).sum()} final signals for {signal_col_name}")
        logging.info(f"Position type: {position_type} (signal value: {signal_value})")

    return df