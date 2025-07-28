"""
main_backtester.py
The main script to run the backtesting engine. Updated to properly handle short positions.
"""
import json
import pandas as pd
import numpy as np
import logging
import argparse
import os
import sys
import inspect

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
        
        strategy_path = os.path.join('strategies', f'{strategy_name}.json')
        with open(strategy_path, 'r') as f:
            strategy_config = json.load(f)
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}. Make sure '{strategy_name}.json' exists in the 'strategies' folder.")
        return

    # --- 2. Get Prepared Data ---
    master_df = data_handler.get_merged_data(global_config)
    
    # --- 3. Generate Signals ---
    df_with_signals = signal_generator.generate_signals(master_df, strategy_config)
    
    # --- 4. Select the Correct Backtest Function ---
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
        return
    
    run_backtest_func = backtest_engine.BACKTEST_FUNCTIONS[exit_type]
    logging.info(f"Using '{exit_type}' exit logic with position type: {position_type}")

    # --- 5. Run Backtests for each parameter set ---
    all_results = {}
    param_grid = strategy_config['param_grid']
    param_name = list(param_grid.keys())[0]
    param_values = param_grid[param_name]
    
    # Get the expected parameters from the function's signature
    func_signature = inspect.signature(run_backtest_func)
    expected_params = list(func_signature.parameters.keys())
    
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

        # CRITICAL CHECK: Verify we have short signals (-1) for short strategies
        if position_type == 'short':
            short_signals = (df_with_signals[signal_col] == -1).sum()
            long_signals = (df_with_signals[signal_col] == 1).sum()
            logging.info(f"Signal check - Short signals: {short_signals}, Long signals: {long_signals}")
            
            if short_signals == 0 and long_signals > 0:
                logging.error("ERROR: Short strategy is generating long signals! Converting...")
                # Convert long signals to short signals
                df_with_signals[signal_col] = df_with_signals[signal_col] * -1

        data_for_numba = {
            'timestamps': df_with_signals.index.values.astype(np.int64),
            'entry_signals': df_with_signals[signal_col].values,
            'close_prices': df_with_signals['ES_close'].values,
            'high_prices': df_with_signals['ES_high'].values,
            'low_prices': df_with_signals['ES_low'].values,
            'ema_fast': df_with_signals['ES_EMA_9'].values,
            'ema_slow': df_with_signals['ES_EMA_15'].values,
        }
        
        # Assemble all possible parameters
        backtest_params = strategy_config.get('exit_conditions', {}).copy()
        backtest_params[param_name] = value
        backtest_params['cost_pct'] = global_config['settings']['transaction_cost_pct']
        
        # Filter the dictionary to only pass expected parameters
        final_backtest_params = {k: v for k, v in backtest_params.items() if k in expected_params}

        trades_list = run_backtest_func(
            **data_for_numba,
            **final_backtest_params
        )
        
        # Process results
        trade_log = pd.DataFrame(
            trades_list, 
            columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl_pct', 'exit_reason']
        )
        trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
        trade_log['exit_time'] = pd.to_datetime(trade_log['exit_time'])
        trade_log['pnl_pct'] = trade_log['pnl_pct'] * 100
        
        # Add position type to trade log
        trade_log['position_type'] = position_type
        
        if value == param_values[0]:
            log_filename = f"trade_log_{strategy_config['strategy_name']}_{value}.csv"
            trade_log.to_csv(log_filename, index=False)
            logging.info(f"Saved detailed trade log to {log_filename}")
            
            # Verify a sample trade
            if len(trade_log) > 0:
                sample = trade_log.iloc[0]
                logging.info(f"Sample trade verification:")
                logging.info(f"  Position: {position_type}")
                logging.info(f"  Entry: ${sample['entry_price']:.2f}")
                logging.info(f"  Exit: ${sample['exit_price']:.2f}")
                logging.info(f"  P&L: {sample['pnl_pct']:.2f}%")
                if position_type == 'short':
                    expected_pnl = ((sample['entry_price'] / sample['exit_price'] - 1) * 100) - (backtest_params['cost_pct'] * 100)
                    logging.info(f"  Expected P&L for short: {expected_pnl:.2f}%")
        
        metrics = backtest_engine.calculate_metrics(trade_log)
        all_results[value] = metrics
        
    # --- 6. Display Final Comparison Report ---
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
    
    results_filename = f"results_{strategy_config['strategy_name']}.csv"
    results_df.to_csv(results_filename)
    logging.info(f"Saved results to {results_filename}")

if __name__ == "__main__":
    main()