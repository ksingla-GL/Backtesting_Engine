"""
main_backtester.py
The main script to run the backtesting engine. 
Optimized to extract required columns from strategy and pass to data_handler.
"""
import json
import pandas as pd
import numpy as np
import logging
import argparse
import os
import sys
import inspect
import traceback
import re

# Import the refactored modules
import data_handler
import signal_generator
import backtest_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_available_strategies():
    """Lists all available .json files in the strategies directory."""
    strategy_dir = 'strategies'
    if not os.path.isdir(strategy_dir):
        return []
    return [f.replace('.json', '') for f in os.listdir(strategy_dir) if f.endswith('.json')]

def calculate_multiperiod_returns(trade_log, df_with_signals, position_type='long'):
    """
    Calculate 1, 3, 5, and 10-day holding period returns for each trade.
    Returns are calculated from entry_price to the close price N business days later.
    """
    if trade_log.empty:
        return trade_log
    
    # Create a copy to avoid modifying original
    enhanced_log = trade_log.copy()
    
    # Initialize new columns
    for period in [1, 3, 5, 10]:
        enhanced_log[f'return_{period}d'] = np.nan
    
    # Create daily close price series (end-of-day prices only)
    # Resample to daily frequency and take last close price of each day
    daily_closes = df_with_signals['ES_close'].resample('D').last().dropna()
    
    for idx, trade in enhanced_log.iterrows():
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        
        # Find future close prices for each holding period
        for period in [1, 3, 5, 10]:
            try:
                # Calculate target date using business days from entry_time
                target_date = entry_time + pd.tseries.offsets.BDay(period)
                
                # Find the close price on or after the target business day
                future_closes = daily_closes[daily_closes.index >= target_date]
                
                if len(future_closes) > 0:
                    exit_price = future_closes.iloc[0]
                    
                    # Calculate return based on position type
                    if position_type == 'long':
                        return_pct = (exit_price / entry_price - 1) * 100
                    else:  # short
                        return_pct = (entry_price / exit_price - 1) * 100
                    
                    enhanced_log.loc[idx, f'return_{period}d'] = return_pct
                    
            except Exception as e:
                # If can't find future price, leave as NaN
                continue
    
    return enhanced_log


def calculate_regime_analysis(trade_log, results_df):
    """
    Perform rolling 2-year window analysis of strategy performance.
    Returns regime analysis data as formatted strings.
    """
    if trade_log.empty:
        return ["No trades available for regime analysis."]
    
    # Convert entry_time to datetime if needed
    trade_log = trade_log.copy()
    trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
    trade_log['year'] = trade_log['entry_time'].dt.year
    
    min_year = trade_log['year'].min()
    max_year = trade_log['year'].max()
    
    regime_results = []
    regime_results.append("\n" + "="*50)
    regime_results.append("REGIME ANALYSIS - Rolling 2-Year Windows")
    regime_results.append("="*50)
    
    # Generate non-overlapping 2-year windows
    for start_year in range(min_year, max_year + 1, 2):
        end_year = start_year + 1
        window_trades = trade_log[
            (trade_log['year'] >= start_year) & 
            (trade_log['year'] <= end_year)
        ]
        
        if len(window_trades) == 0:
            continue
            
        # Calculate metrics for this window
        total_trades = len(window_trades)
        winners = window_trades[window_trades['pnl_pct'] > 0]
        win_rate = (len(winners) / total_trades) * 100 if total_trades > 0 else 0
        avg_return = window_trades['pnl_pct'].mean()
        total_return = window_trades['pnl_pct'].sum()
        
        # Multi-period analysis for this window
        multiperiod_stats = {}
        for period in [1, 3, 5, 10]:
            col = f'return_{period}d'
            if col in window_trades.columns:
                valid_returns = window_trades[col].dropna()
                if len(valid_returns) > 0:
                    multiperiod_stats[f'{period}d'] = {
                        'avg_return': valid_returns.mean(),
                        'win_rate': (valid_returns > 0).mean() * 100,
                        'count': len(valid_returns)
                    }
        
        # Format results for this window
        regime_results.append(f"\n{start_year}-{end_year}:")
        regime_results.append(f"  Trades: {total_trades}")
        regime_results.append(f"  Win Rate: {win_rate:.1f}%")
        regime_results.append(f"  Avg Return: {avg_return:.2f}%")
        regime_results.append(f"  Total Return: {total_return:.2f}%")
        
        # Add multi-period stats if available
        if multiperiod_stats:
            regime_results.append("  Multi-Period Analysis:")
            for period, stats in multiperiod_stats.items():
                regime_results.append(f"    {period}: Avg {stats['avg_return']:.2f}%, Win Rate {stats['win_rate']:.1f}% ({stats['count']} trades)")
    
    return regime_results


def format_enhanced_results(results_df, trade_log, strategy_name, position_type):
    """
    Format results with traditional metrics + regime analysis + multi-period summary.
    """
    output_lines = []
    
    # Traditional results section (unchanged)
    output_lines.append("="*80)
    output_lines.append(f"FINAL BACKTEST RESULTS: {strategy_name}")
    output_lines.append(f"POSITION TYPE: {position_type.upper()}")
    output_lines.append("="*80)
    
    # Format traditional results
    formatted_df = results_df.copy()
    for col in formatted_df.columns:
        if formatted_df[col].dtype in ['float64', 'float32']:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and x != float('inf') else ('Inf' if x == float('inf') else 'N/A')
            )
    
    output_lines.append(formatted_df.to_string())
    
    # Multi-period analysis summary
    if not trade_log.empty:
        output_lines.append("\n" + "="*50)
        output_lines.append("MULTI-PERIOD ANALYSIS SUMMARY")
        output_lines.append("="*50)
        
        for period in [1, 3, 5, 10]:
            col = f'return_{period}d'
            if col in trade_log.columns:
                valid_returns = trade_log[col].dropna()
                if len(valid_returns) > 0:
                    avg_return = valid_returns.mean()
                    win_rate = (valid_returns > 0).mean() * 100
                    count = len(valid_returns)
                    output_lines.append(f"{period}-Day HPR: Avg {avg_return:.2f}%, Win Rate {win_rate:.1f}% ({count} trades)")
    
    # Regime analysis
    regime_analysis = calculate_regime_analysis(trade_log, results_df)
    output_lines.extend(regime_analysis)
    
    output_lines.append("="*80)
    return output_lines


def extract_required_columns_from_strategy(strategy_config):
    """
    Extract all columns that will be needed by a strategy.
    This includes columns from indicators, entry rules, and exit conditions.
    """
    required_columns = set()
    
    # 1. Extract columns from indicators
    for indicator in strategy_config.get('indicators', []):
        # Input column
        if 'on_column' in indicator and indicator['on_column']:
            required_columns.add(indicator['on_column'])
        
        # Output column (will be created, but we need to know dependencies)
        if 'output_col' in indicator:
            required_columns.add(indicator['output_col'])
        
        # Special indicator dependencies
        indicator_name = indicator['name']
        params = indicator.get('params', {})
        
        # Handle indicators that need multiple columns
        if indicator_name == 'ADX':
            instrument = indicator['on_column'].split('_')[0] if 'on_column' in indicator else 'ES'
            required_columns.update([f"{instrument}_high", f"{instrument}_low", f"{instrument}_close"])
        elif indicator_name == 'VIXSpike':
            if 'prev_close_col' in indicator:
                required_columns.add(indicator['prev_close_col'])
        elif indicator_name == 'DeclineFromPeak':
            if 'peak_column' in indicator:
                required_columns.add(indicator['peak_column'])
        elif indicator_name in ['IsUpDay', 'ConsecutiveGreenDays']:
            instrument = params.get('instrument', 'ES')
            required_columns.update([f"{instrument}_open", f"{instrument}_close"])
        elif indicator_name == 'OBV':
            instrument = indicator['on_column'].split('_')[0] if 'on_column' in indicator else 'ES'
            required_columns.update([f"{instrument}_close", f"{instrument}_volume"])
        elif indicator_name == 'OBVHighN':
            if 'obv_column' in params:
                required_columns.add(params['obv_column'])
        elif indicator_name == 'VolumeRatio':
            required_columns.add(indicator['on_column'])
        elif indicator_name == 'BreakoutRetest':
            instrument = params.get('instrument', 'ES')
            required_columns.update([f"{instrument}_high", f"{instrument}_low", f"{instrument}_close"])
        elif indicator_name == 'VWAP':
            instrument = params.get('instrument', 'ES')
            required_columns.update([f"{instrument}_close", f"{instrument}_volume"])
        elif indicator_name == 'MarketBreadthStrong':
            required_columns.update(['MARKET_BREADTH', 'TRIN_close'])
        elif indicator_name == 'VXDeclineWindow':
            required_columns.add('VX_close')
    
    # 2. Extract columns from entry rules
    all_rules = []
    
    # Base entry rules
    base_rules = strategy_config.get('base_entry_rules', strategy_config.get('entry_rules', []))
    all_rules.extend(base_rules)
    
    # Conditional entry rules
    for cond_rule in strategy_config.get('conditional_entry_rules', []):
        if 'rule' in cond_rule:
            if isinstance(cond_rule['rule'], str):
                all_rules.append(cond_rule['rule'])
            elif isinstance(cond_rule['rule'], dict) and 'conditions' in cond_rule['rule']:
                all_rules.extend(cond_rule['rule']['conditions'])
    
    # Parse rules for column names
    for rule in all_rules:
        if isinstance(rule, str):
            # Extract column names (identifiers that aren't keywords)
            columns = re.findall(r'\b(?<!@)([a-zA-Z_][a-zA-Z0-9_]*)\b', rule)
            keywords = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'}
            required_columns.update([col for col in columns if col not in keywords and not col.isnumeric()])
    
    # 3. Always include primary instrument OHLCV
    primary_instrument = strategy_config.get('primary_instrument', 'ES')
    required_columns.update([
        f"{primary_instrument}_open",
        f"{primary_instrument}_high", 
        f"{primary_instrument}_low",
        f"{primary_instrument}_close",
        f"{primary_instrument}_volume"
    ])
    
    # 4. Add EMA columns for exit logic (always needed)
    required_columns.update(['ES_EMA_9', 'ES_EMA_15'])
    
    # 5. Add VIX for liquidity exits
    if strategy_config.get('exit_type') == 'liquidity_exits':
        required_columns.update(['VIX_close'])
    
    # 6. Check for macro event requirements
    macro_keywords = ['cpi', 'fomc', 'fed', 'nfp', 'is_pre_', 'is_post_', '_inline', '_better', '_worse']
    needs_macro = any(keyword in col.lower() for col in required_columns for keyword in macro_keywords)
    
    if needs_macro:
        # Add base columns needed for macro calculations
        required_columns.update(['ES_close', 'VIX_close'])
    
    # Convert to sorted list for consistent ordering
    return sorted(list(required_columns))

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run a strategy backtest.")
    parser.add_argument('strategy_name', nargs='?', default=None, help="The name of the strategy config file (without .json extension).")
    args = parser.parse_args()

    if not args.strategy_name:
        print("Error: No strategy name provided.")
        available = list_available_strategies()
        if available:
            print("Available strategies:", ", ".join(available))
        else:
            print("No strategies found in the 'strategies' directory.")
        sys.exit(1)

    strategy_name = args.strategy_name
    logging.info(f"--- Starting Backtesting Engine for Strategy: {strategy_name} ---")
    
    # --- 1. Load Configurations ---
    try:
        with open('config.json', 'r') as f:
            global_config = json.load(f)
        logging.info("Loaded global configuration")
        
        strategy_path = os.path.join('strategies', f'{strategy_name}.json')
        with open(strategy_path, 'r') as f:
            strategy_config = json.load(f)
        logging.info(f"Loaded strategy configuration: {strategy_config['strategy_name']}")
        
        # Auto-add ATR indicator if ATR-based stop loss is configured
        exit_conditions = strategy_config.get('exit_conditions', {})
        if 'stop_loss_atr_multiplier' in exit_conditions and exit_conditions['stop_loss_atr_multiplier'] > 0:
            atr_period = exit_conditions.get('atr_period', 14)
            atr_indicator = {
                "name": "ATR",
                "params": {"window": atr_period},
                "on_column": "ES_close",
                "output_col": f"ES_ATR_{atr_period}"
            }
            
            # Add ATR indicator if not already present
            indicators = strategy_config.get('indicators', [])
            atr_exists = any(ind.get('name') == 'ATR' and ind.get('output_col') == f"ES_ATR_{atr_period}" for ind in indicators)
            if not atr_exists:
                indicators.append(atr_indicator)
                strategy_config['indicators'] = indicators
                logging.info(f"Auto-added ATR indicator with period {atr_period} for ATR-based stop loss")
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}. Make sure '{strategy_name}.json' exists in the 'strategies' folder.")
        return
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        traceback.print_exc()
        return

    # --- 2. Extract Required Columns ---
    try:
        logging.info("Extracting required columns from strategy...")
        required_columns = extract_required_columns_from_strategy(strategy_config)
        logging.info(f"Strategy requires {len(required_columns)} columns: {required_columns[:10]}..." if len(required_columns) > 10 else f"Strategy requires columns: {required_columns}")
    except Exception as e:
        logging.error(f"Error extracting required columns: {e}")
        traceback.print_exc()
        required_columns = None  # Fall back to loading all data

    # --- 3. Get Prepared Data ---
    try:
        logging.info("Loading and preparing data...")
        master_df = data_handler.get_merged_data(global_config, required_columns)
        logging.info(f"Data loaded successfully. Shape: {master_df.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        traceback.print_exc()
        return
    
    # --- 4. Generate Signals ---
    try:
        logging.info("Generating trading signals...")
        df_with_signals = signal_generator.generate_signals(master_df, strategy_config)
        logging.info(f"Signals generated. DataFrame shape: {df_with_signals.shape}")
        
        # Check if we have the required EMAs
        if 'ES_EMA_9' not in df_with_signals.columns or 'ES_EMA_15' not in df_with_signals.columns:
            logging.warning("ES_EMA_9 or ES_EMA_15 not found. Calculating now...")
            # Calculate 9 and 15 EMAs if not present
            daily_close = df_with_signals['ES_close'].resample('D').last().dropna()
            ema_9 = daily_close.ewm(span=9, adjust=False).mean()
            ema_15 = daily_close.ewm(span=15, adjust=False).mean()
            df_with_signals['ES_EMA_9'] = ema_9.reindex(df_with_signals.index, method='ffill').ffill()
            df_with_signals['ES_EMA_15'] = ema_15.reindex(df_with_signals.index, method='ffill').ffill()
            
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        traceback.print_exc()
        return
    
    # --- 5. Select the Correct Backtest Function ---
    exit_type = strategy_config.get('exit_type', 'simple_exits')
    position_type = strategy_config.get('position_type', 'long')
    
    # CRITICAL FIX: Check position type to select correct function
    if position_type == 'short':
        if exit_type == 'simple_exits_no_ma_cross':
            exit_type = 'simple_exits_short_no_ma_cross'
        elif exit_type == 'simple_exits':
            exit_type = 'simple_exits_short'
        logging.info(f"Using SHORT position logic for '{strategy_name}'")
    
    if exit_type not in backtest_engine.BACKTEST_FUNCTIONS:
        logging.error(f"Exit type '{exit_type}' specified in '{strategy_name}.json' is not a valid backtest function.")
        logging.error(f"Available exit types: {list(backtest_engine.BACKTEST_FUNCTIONS.keys())}")
        return
    
    run_backtest_func = backtest_engine.BACKTEST_FUNCTIONS[exit_type]
    logging.info(f"Using '{exit_type}' exit logic with position type: {position_type}")

    # --- 6. Run Backtests for each parameter set ---
    all_results = {}
    param_grid = strategy_config['param_grid']
    param_name = list(param_grid.keys())[0]
    param_values = param_grid[param_name]
    
    # Get the expected parameters from the function's signature
    func_signature = inspect.signature(run_backtest_func)
    expected_params = list(func_signature.parameters.keys())
    logging.info(f"Expected parameters for {exit_type}: {expected_params}")
    
    for value in param_values:
        logging.info(f"\n--- Running Backtest for {param_name} = {value} ---")
        
        signal_col = f"entry_signal_{value}"
        
        if signal_col not in df_with_signals.columns:
            # This can happen if the parameter is for exits, not entries.
            first_param_val = param_values[0]
            signal_col = f"entry_signal_{first_param_val}"
            if signal_col not in df_with_signals.columns:
                 logging.warning(f"No signal columns found. Skipping backtest.")
                 continue
            logging.info(f"Parameter '{param_name}' is not an entry rule. Using signals from '{signal_col}'.")

        # Log signal count
        signal_count = (df_with_signals[signal_col] != 0).sum()
        logging.info(f"Total signals found: {signal_count}")
        
        # CRITICAL CHECK: Verify we have short signals (-1) for short strategies
        if position_type == 'short':
            short_signals = (df_with_signals[signal_col] == -1).sum()
            long_signals = (df_with_signals[signal_col] == 1).sum()
            logging.info(f"Signal check - Short signals: {short_signals}, Long signals: {long_signals}")
            
            if short_signals == 0 and long_signals > 0:
                # Convert long signals to short signals
                df_with_signals[signal_col] = df_with_signals[signal_col] * -1

        try:
            # Prepare data for Numba
            data_for_numba = {
                'timestamps': df_with_signals.index.values.astype(np.int64),
                'entry_signals': df_with_signals[signal_col].values,
                'close_prices': df_with_signals['ES_close'].values,
                'high_prices': df_with_signals['ES_high'].values,
                'low_prices': df_with_signals['ES_low'].values,
                'ema_fast': df_with_signals['ES_EMA_9'].values,
                'ema_slow': df_with_signals['ES_EMA_15'].values,
            }
            
            # Add ATR data if strategy uses ATR-based stop losses
            exit_conditions = strategy_config.get('exit_conditions', {})
            if 'stop_loss_atr_multiplier' in exit_conditions and exit_conditions['stop_loss_atr_multiplier'] > 0:
                atr_col = 'ES_ATR_14'  # Default ATR column name
                if atr_col not in df_with_signals.columns:
                    logging.warning(f"ATR column {atr_col} not found, using zeros for ATR values")
                    data_for_numba['atr_values'] = np.zeros(len(df_with_signals))
                else:
                    data_for_numba['atr_values'] = df_with_signals[atr_col].values
                    logging.info(f"ATR data included: {atr_col}, sample values: {df_with_signals[atr_col].iloc[:5].values}")
            else:
                # Provide dummy ATR values for backward compatibility
                data_for_numba['atr_values'] = np.zeros(len(df_with_signals))
            
            # For liquidity_exits, we need additional data
            if exit_type == 'liquidity_exits':
                logging.info("Preparing VIX data for liquidity_exits...")
                # Add VIX closes
                data_for_numba['vix_closes'] = df_with_signals['VIX_close'].values
                
                # Create previous day VIX close array for spike detection
                vix_prev_day = df_with_signals['VIX_close'].resample('D').last().shift(1)
                data_for_numba['entry_vix_prices'] = vix_prev_day.reindex(df_with_signals.index, method='ffill').fillna(method='bfill').values
                
                logging.info(f"VIX data prepared. Sample VIX values: {df_with_signals['VIX_close'].iloc[:5].values}")
            
            # For time-based exits, we need time arrays
            if exit_type in ['time_exits', 'time_exits_short']:
                logging.info("Preparing time data for time-based exits...")
                data_for_numba['hours_array'] = df_with_signals.index.hour.values.astype(np.int32)
                data_for_numba['minutes_array'] = df_with_signals.index.minute.values.astype(np.int32)
                logging.info(f"Time data prepared. Sample hours: {data_for_numba['hours_array'][:5]}, Sample minutes: {data_for_numba['minutes_array'][:5]}")
                
            # Assemble all possible parameters
            backtest_params = strategy_config.get('exit_conditions', {}).copy()
            backtest_params[param_name] = value
            backtest_params['cost_pct'] = global_config['settings']['transaction_cost_pct']
            
            # Set default ATR parameters if not specified
            if 'stop_loss_atr_multiplier' not in backtest_params:
                backtest_params['stop_loss_atr_multiplier'] = 0.0
            
            # Filter the dictionary to only pass expected parameters
            final_backtest_params = {k: v for k, v in backtest_params.items() if k in expected_params}
            
            # Special handling for liquidity_exits
            if exit_type == 'liquidity_exits' and 'vix_spike_pct' in expected_params:
                final_backtest_params['vix_spike_pct'] = backtest_params.get('vix_spike_pct', 10.0)
                
            logging.info(f"Running backtest with parameters: {final_backtest_params}")
            
            # Run the backtest
            trades_list = run_backtest_func(
                **data_for_numba,
                **final_backtest_params
            )
            
            logging.info(f"Backtest complete. Number of trades: {len(trades_list)}")
            
        except Exception as e:
            logging.error(f"Error running backtest: {e}")
            traceback.print_exc()
            continue
        
        # Process results
        try:
            trade_log = pd.DataFrame(
                trades_list, 
                columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl_pct', 'exit_reason']
            )
            trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
            trade_log['exit_time'] = pd.to_datetime(trade_log['exit_time'])
            trade_log['pnl_pct'] = trade_log['pnl_pct'] * 100
            
            # Add descriptive exit reasons
            trade_log['exit_reason_desc'] = trade_log['exit_reason'].apply(backtest_engine.get_exit_reason_description)
            
            # Add position type to trade log
            trade_log['position_type'] = position_type
            
            # Calculate multi-period returns (1, 3, 5, 10 days) for each trade
            if len(trade_log) > 0:
                logging.info("Calculating multi-period holding returns...")
                trade_log = calculate_multiperiod_returns(trade_log, df_with_signals, position_type)
                logging.info(f"Enhanced trade log with multi-period returns. Shape: {trade_log.shape}")
            
            # Create Trade_Logs directory if it doesn't exist
            if not os.path.exists('Trade_Logs'):
                os.makedirs('Trade_Logs')
            
            if value == param_values[0] and len(trade_log) > 0:
                log_filename = f"Trade_Logs/trade_log_{strategy_config['strategy_name']}_{value}.csv"
                trade_log.to_csv(log_filename, index=False)
                logging.info(f"Saved enhanced trade log with multi-period returns to {log_filename}")
                
                # Verify a sample trade
                sample = trade_log.iloc[0]
                logging.info(f"Sample trade verification:")
                logging.info(f"  Position: {position_type}")
                logging.info(f"  Entry: ${sample['entry_price']:.2f}")
                logging.info(f"  Exit: ${sample['exit_price']:.2f}")
                logging.info(f"  P&L: {sample['pnl_pct']:.2f}%")
                
            metrics = backtest_engine.calculate_metrics(trade_log)
            all_results[value] = metrics
            
        except Exception as e:
            logging.error(f"Error processing results: {e}")
            traceback.print_exc()
            continue
    
    # --- 7. Display Final Comparison Report ---
    if not all_results:
        logging.error("No results to display. Something went wrong during backtesting.")
        return
        
    try:
        results_df = pd.DataFrame(all_results).T
        results_df.index.name = f"Parameter: {param_name}"
        
        # Get the first trade log for enhanced analysis (should have multi-period data)
        first_param_value = param_values[0]
        enhanced_trade_log = None
        
        # Try to get trade log from the first successful backtest
        for value in param_values:
            signal_col = f"entry_signal_{value}"
            if signal_col not in df_with_signals.columns:
                signal_col = f"entry_signal_{first_param_value}"
            
            if signal_col in df_with_signals.columns:
                # Reconstruct the trade log for analysis (this would be the same as what was processed)
                if value in all_results and 'Total Trades' in all_results[value] and all_results[value]['Total Trades'] > 0:
                    # We need to get the trade log that was used for this value
                    # For now, we'll use a basic approach - in practice, we might want to store this
                    break
        
        # Generate enhanced results format
        enhanced_output = format_enhanced_results(results_df, trade_log, strategy_config['strategy_name'], position_type)
        
        # Print the enhanced results
        for line in enhanced_output:
            print(line)
        
        # Create Results directory if it doesn't exist
        if not os.path.exists('Results'):
            os.makedirs('Results')
        
        # Save traditional CSV results (for backward compatibility)
        results_filename = f"Results/results_{strategy_config['strategy_name']}.csv"
        results_df.to_csv(results_filename)
        logging.info(f"Saved traditional results to {results_filename}")
        
        # Save enhanced results as text file
        enhanced_filename = f"Results/results_{strategy_config['strategy_name']}_enhanced.txt"
        with open(enhanced_filename, 'w') as f:
            for line in enhanced_output:
                f.write(line + '\n')
        logging.info(f"Saved enhanced results with regime analysis to {enhanced_filename}")
        
    except Exception as e:
        logging.error(f"Error displaying results: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()