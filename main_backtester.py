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
            
            # For liquidity_exits, we need additional data
            if exit_type == 'liquidity_exits':
                logging.info("Preparing VIX data for liquidity_exits...")
                # Add VIX closes
                data_for_numba['vix_closes'] = df_with_signals['VIX_close'].values
                
                # Create previous day VIX close array for spike detection
                vix_prev_day = df_with_signals['VIX_close'].resample('D').last().shift(1)
                data_for_numba['entry_vix_prices'] = vix_prev_day.reindex(df_with_signals.index, method='ffill').fillna(method='bfill').values
                
                logging.info(f"VIX data prepared. Sample VIX values: {df_with_signals['VIX_close'].iloc[:5].values}")
                
            # Assemble all possible parameters
            backtest_params = strategy_config.get('exit_conditions', {}).copy()
            backtest_params[param_name] = value
            backtest_params['cost_pct'] = global_config['settings']['transaction_cost_pct']
            
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
            
            # Add position type to trade log
            trade_log['position_type'] = position_type
            
            # Create Trade_Logs directory if it doesn't exist
            if not os.path.exists('Trade_Logs'):
                os.makedirs('Trade_Logs')
            
            if value == param_values[0] and len(trade_log) > 0:
                log_filename = f"Trade_Logs/trade_log_{strategy_config['strategy_name']}_{value}.csv"
                trade_log.to_csv(log_filename, index=False)
                logging.info(f"Saved detailed trade log to {log_filename}")
                
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
        
        def format_float(x):
            if isinstance(x, (int, float)):
                return f"{x:.2f}" if pd.notna(x) and x != float('inf') else ('Inf' if x == float('inf') else 'N/A')
            return str(x)
        
        print("\n" + "="*80)
        print(f"FINAL BACKTEST RESULTS: {strategy_config['strategy_name']}")
        print(f"POSITION TYPE: {position_type.upper()}")
        print("="*80)
        
        formatted_df = results_df.copy()
        for col in formatted_df.columns:
            if formatted_df[col].dtype in ['float64', 'float32']:
                formatted_df[col] = formatted_df[col].apply(format_float)
        
        print(formatted_df.to_string())
        print("="*80)
        
        # Create Results directory if it doesn't exist
        if not os.path.exists('Results'):
            os.makedirs('Results')
        
        results_filename = f"Results/results_{strategy_config['strategy_name']}.csv"
        results_df.to_csv(results_filename)
        logging.info(f"Saved results to {results_filename}")
        
    except Exception as e:
        logging.error(f"Error displaying results: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()