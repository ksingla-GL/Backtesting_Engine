"""
signal_generator.py
Module to dynamically calculate ALL technical indicators and generate trading signals.
Enhanced to support both long and short positions and handle new strategies.
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
    
    if name not in ['IsUpDay', 'MACD', 'ConsecutiveGreenDays', 'BreakoutRetest', 'OBVHighN'] and on_col and on_col not in df.columns:
        logging.error(f"Input column '{on_col}' not found for indicator {name}")
        df[out_col] = np.nan
        return df
    
    if name not in ['IsUpDay', 'MACD', 'ConsecutiveGreenDays', 'BreakoutRetest', 'OBVHighN'] and on_col and df[on_col].isna().all():
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

        # NEW INDICATORS START HERE
        elif name == 'ADX':
            window = params.get('window', 14)
            # Need high, low, close for ADX calculation
            instrument = on_col.split('_')[0]
            daily_high = df[f"{instrument}_high"].resample('D').max().dropna()
            daily_low = df[f"{instrument}_low"].resample('D').min().dropna()
            daily_close = df[on_col].resample('D').last().dropna()
            daily_adx = ta.trend.adx(daily_high, daily_low, daily_close, window=window)
            df[out_col] = daily_adx.reindex(df.index, method='ffill').ffill()

        elif name == 'EMASlope':
            span = params.get('span')
            lookback = params.get('lookback', 1)
            daily_close = df[on_col].resample('D').last().dropna()
            daily_ema = daily_close.ewm(span=span, adjust=False).mean()
            daily_slope = (daily_ema - daily_ema.shift(lookback)) / daily_ema.shift(lookback)
            df[out_col] = daily_slope.reindex(df.index, method='ffill').ffill()

        elif name == 'VIXChange':
            lookback = params.get('lookback', 20)
            daily_vix = df[on_col].resample('D').last().dropna()
            vix_sma = daily_vix.rolling(window=lookback).mean()
            vix_declining = daily_vix < vix_sma
            df[out_col] = vix_declining.reindex(df.index, method='ffill').ffill()

        elif name == 'SMASlope':
            window = params.get('window', 50)
            lookback = params.get('lookback', 1)
            daily_close = df[on_col].resample('D').last().dropna()
            daily_sma = daily_close.rolling(window=window, min_periods=window).mean()
            sma_slope = (daily_sma - daily_sma.shift(lookback)) > 0
            df[out_col] = sma_slope.reindex(df.index, method='ffill').ffill()

        elif name == 'RealizedVolatility':
            window = params.get('window', 10)
            daily_close = df[on_col].resample('D').last().dropna()
            daily_returns = daily_close.pct_change()
            realized_vol = daily_returns.rolling(window=window).std() * np.sqrt(252) * 100
            df[out_col] = realized_vol.reindex(df.index, method='ffill').ffill()

        elif name == 'ConsecutiveGreenDays':
            instrument = params.get('instrument', 'SPX')
            close_col = f"{instrument}_close"
            open_col = f"{instrument}_open"
            
            # Calculate daily green/red
            daily_open = df[open_col].resample('D').first()
            daily_close = df[close_col].resample('D').last()
            is_green = daily_close > daily_open
            
            # Count consecutive green days
            consecutive = is_green.astype(int).groupby((~is_green).cumsum()).cumsum()
            df[out_col] = consecutive.reindex(df.index, method='ffill').ffill()

        elif name == 'VolumeRatio':
            window = params.get('window', 20)
            daily_volume = df[on_col].resample('D').sum()
            avg_volume = daily_volume.rolling(window=window).mean()
            volume_ratio = daily_volume / avg_volume
            df[out_col] = volume_ratio.reindex(df.index, method='ffill').ffill()

        elif name == 'OBV':
            # On Balance Volume
            instrument = on_col.split('_')[0]
            close_col = f"{instrument}_close"
            
            daily_close = df[close_col].resample('D').last()
            daily_volume = df[on_col].resample('D').sum()
            
            # Calculate OBV
            close_diff = daily_close.diff()
            obv = daily_volume.copy()
            obv[close_diff < 0] *= -1
            obv[close_diff == 0] = 0
            obv = obv.cumsum()
            
            df[out_col] = obv.reindex(df.index, method='ffill').ffill()

        elif name == 'OBVHighN':
            # OBV is at N-day high
            window = params.get('window', 20)
            obv_col = params.get('obv_column')
            
            if obv_col not in df.columns:
                df[out_col] = False
                return df
                
            daily_obv = df[obv_col].resample('D').last()
            rolling_max = daily_obv.rolling(window=window).max()
            is_new_high = daily_obv >= rolling_max
            df[out_col] = is_new_high.reindex(df.index, method='ffill').fillna(False)

        elif name == 'BreakoutRetest':
            # Breakout and retest logic based on user's specification
            instrument = params.get('instrument', 'ES')
            high_col = f"{instrument}_high"
            low_col = f"{instrument}_low"
            close_col = f"{instrument}_close"
            breakout_window = params.get('breakout_window', 20)
            retest_window = params.get('retest_window', 7)
            
            # Check if columns exist
            if high_col not in df.columns or low_col not in df.columns:
                logging.error(f"Required columns {high_col} or {low_col} not found!")
                df[out_col] = False
                return df
            
            # Calculate on daily data - drop NaN values to handle weekends/holidays
            daily_high = df[high_col].resample('D').max().dropna()
            daily_low = df[low_col].resample('D').min().dropna()
            daily_close = df[close_col].resample('D').last().dropna()
            
            # Debug: Check data quality
            logging.info(f"    BreakoutRetest Debug - Daily data shape after dropping NaN: {len(daily_high)} days")
            logging.info(f"    First few daily highs: {daily_high.head()}")
            logging.info(f"    Daily high range: {daily_high.min():.2f} to {daily_high.max():.2f}")
            
            # Step 1: Calculate 20-day rolling high (not including today)
            # Use min_periods=1 to handle the beginning of the series
            rolling_20d_high = daily_high.rolling(window=breakout_window, min_periods=min(breakout_window, len(daily_high))).max().shift(1)
            
            # Debug: Check rolling calculation
            non_nan_rolling = rolling_20d_high.dropna()
            logging.info(f"    Rolling 20d high - non-NaN values: {len(non_nan_rolling)}")
            if len(non_nan_rolling) > 0:
                logging.info(f"    Rolling 20d high range: {non_nan_rolling.min():.2f} to {non_nan_rolling.max():.2f}")
                
                # Check if highs ever exceed rolling highs
                daily_high_aligned = daily_high.reindex(non_nan_rolling.index)
                potential_breakouts = daily_high_aligned > non_nan_rolling
                logging.info(f"    Days where high > 20d high: {potential_breakouts.sum()}")
                
                # Show a few examples
                if potential_breakouts.sum() > 0:
                    examples = potential_breakouts[potential_breakouts].head(5)
                    for date in examples.index:
                        logging.info(f"      {date.date()}: High={daily_high_aligned[date]:.2f} > 20d_high={non_nan_rolling[date]:.2f}")
                else:
                    # Show why no breakouts - look at some recent data
                    sample_dates = non_nan_rolling.index[-10:-5] if len(non_nan_rolling) > 10 else non_nan_rolling.index[:5]
                    logging.info("    Sample of recent data (no breakouts found):")
                    for date in sample_dates:
                        if date in daily_high_aligned.index:
                            logging.info(f"      {date.date()}: High={daily_high_aligned[date]:.2f}, 20d_high={non_nan_rolling[date]:.2f}")
            
            # Continue with original logic but with better alignment
            # Only proceed if we have valid rolling data
            if len(non_nan_rolling) == 0:
                logging.warning("    No valid rolling 20d high data - setting all signals to False")
                df[out_col] = False
                return df
            
            # Align all series to the same index
            valid_dates = non_nan_rolling.index
            daily_high = daily_high.reindex(valid_dates)
            daily_low = daily_low.reindex(valid_dates)
            daily_close = daily_close.reindex(valid_dates)
            rolling_20d_high = non_nan_rolling
            
            # Step 2: Identify breakouts - when today's high breaks above 20-day high
            is_breakout_day = daily_high > rolling_20d_high
            breakout_levels = rolling_20d_high.where(is_breakout_day, np.nan)
            
            # Step 3: Track breakout levels for retest_window days using forward fill
            # This remembers the breakout level for up to 7 trading days
            breakout_memory = breakout_levels.ffill(limit=retest_window-1)
            
            # Step 4: Identify successful retests
            has_past_breakout = breakout_memory.notna()
            low_above_support = daily_low <= breakout_memory * 1.01
            
            retest_signals = has_past_breakout & low_above_support & (~is_breakout_day)
            
            # Log results
            total_breakouts = is_breakout_day.sum()
            total_retests = retest_signals.sum()
            logging.info(f"  BreakoutRetest: Found {total_breakouts} breakout days and {total_retests} retest signals")
            
            # If we found breakouts but no retests, show why
            if total_breakouts > 0 and total_retests == 0:
                logging.info(f"    Days with breakout memory: {has_past_breakout.sum()}")
                logging.info(f"    Days where low > breakout level: {(has_past_breakout & low_above_support).sum()}")
                
                # Show what happened after first breakout
                first_breakout = is_breakout_day[is_breakout_day].index[0]
                logging.info(f"    First breakout on {first_breakout.date()}, level={rolling_20d_high[first_breakout]:.2f}")
                
                # Show next 7 trading days
                start_idx = valid_dates.get_loc(first_breakout)
                for i in range(1, min(8, len(valid_dates) - start_idx)):
                    date = valid_dates[start_idx + i]
                    low = daily_low[date]
                    breakout_mem = breakout_memory[date]
                    logging.info(f"      {date.date()}: Low={low:.2f}, Breakout_level={breakout_mem:.2f if pd.notna(breakout_mem) else 'None'}, "
                               f"Low>Level={low > breakout_mem if pd.notna(breakout_mem) else 'N/A'}")
            
            # Convert to boolean and reindex to original dataframe
            retest_signals = retest_signals.fillna(False).astype(bool)
            
            # Reindex to intraday data
            df[out_col] = retest_signals.reindex(df.index, method='ffill').fillna(False)
            
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
        
    # Special handling for LiquidityDrift strategy - delay signals by 1 day
    if strategy_config['strategy_name'] == 'LiquidityDrift':
        for value in param_values:
            signal_col_name = f"entry_signal_{value}"
            if signal_col_name in df.columns:
                # Shift signals by 1 day
                daily_signals = df[signal_col_name].resample('D').max()
                next_day_signals = daily_signals.shift(1)
                df[signal_col_name] = next_day_signals.reindex(df.index, method='ffill').fillna(0).astype(int)
                
                # Re-apply one signal per day logic
                df[signal_col_name] = df[signal_col_name].groupby(df.index.date).transform(
                    lambda x: (x.cumsum() == 1) & (x == 1)
                ).astype(int)
                
                logging.info(f"Delayed signals for {signal_col_name} - now {df[signal_col_name].sum()} signals")

    return df