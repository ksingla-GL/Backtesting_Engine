"""
backtest_engine.py
Contains a library of high-performance, Numba-optimized backtesting functions,
each tailored to a specific type of exit logic. Also includes performance
metric calculation functions.
Enhanced to support short positions.
"""
import pandas as pd
import numpy as np
from numba import jit

# ##################################################################
# BACKTESTING FUNCTION LIBRARY
# ##################################################################

@jit(nopython=True)
def run_backtest_simple_exits(
    # Data arrays
    timestamps,
    entry_signals,
    close_prices,
    high_prices,
    low_prices,
    ema_fast,
    ema_slow,
    # Parameters
    stop_loss_pct,
    take_profit_pct,
    cost_pct,
    trailing_stop_loss_pct=0.0
):
    """
    Numba-optimized backtesting loop for strategies with a simple exit logic:
    1. Static Stop Loss
    2. Trailing Stop Loss
    3. Take Profit
    4. Moving Average Crossover
    """
    n = len(timestamps)
    trades = []
    
    in_position = False
    entry_time = 0
    entry_price = 0.0
    stop_loss_price = 0.0
    initial_stop_price = 0.0
    take_profit_price = 0.0
    
    EXIT_SL = 1
    EXIT_TP = 2
    EXIT_MA_CROSS = 3
    EXIT_TSL = 4

    for i in range(1, n):
        if in_position:
            if trailing_stop_loss_pct > 0:
                new_potential_stop = high_prices[i] * (1 - trailing_stop_loss_pct / 100)
                if new_potential_stop > stop_loss_price:
                    stop_loss_price = new_potential_stop

            exit_reason = 0
            exit_price = 0.0

            if low_prices[i] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = EXIT_TSL if stop_loss_price > initial_stop_price else EXIT_SL
            elif high_prices[i] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = EXIT_TP
            elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                exit_price = close_prices[i]
                exit_reason = EXIT_MA_CROSS

            if exit_reason > 0:
                pnl = (exit_price / entry_price - 1) - cost_pct
                trades.append((entry_time, timestamps[i], entry_price, exit_price, pnl, exit_reason))
                in_position = False

        if not in_position and entry_signals[i] == 1:
            in_position = True
            entry_time = timestamps[i]
            entry_price = close_prices[i]
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            initial_stop_price = stop_loss_price
            take_profit_price = entry_price * (1 + take_profit_pct / 100)
            
    return trades


@jit(nopython=True)
def run_backtest_simple_exits_no_ma_cross(
    # Data arrays
    timestamps,
    entry_signals,
    close_prices,
    high_prices,
    low_prices,
    ema_fast, # Included for function signature compatibility, but not used
    ema_slow, # Included for function signature compatibility, but not used
    # Parameters
    stop_loss_pct,
    take_profit_pct,
    cost_pct,
    trailing_stop_loss_pct=0.0
):
    """
    Numba-optimized backtesting loop for reversal strategies. Exits ONLY on:
    1. Static Stop Loss
    2. Trailing Stop Loss
    3. Take Profit
    (MA Crossover exit is REMOVED)
    """
    n = len(timestamps)
    trades = []
    
    in_position = False
    entry_time = 0
    entry_price = 0.0
    stop_loss_price = 0.0
    initial_stop_price = 0.0
    take_profit_price = 0.0
    
    EXIT_SL = 1
    EXIT_TP = 2
    # MA_CROSS (3) is intentionally omitted
    EXIT_TSL = 4

    for i in range(1, n):
        if in_position:
            if trailing_stop_loss_pct > 0:
                new_potential_stop = high_prices[i] * (1 - trailing_stop_loss_pct / 100)
                if new_potential_stop > stop_loss_price:
                    stop_loss_price = new_potential_stop

            exit_reason = 0
            exit_price = 0.0

            if low_prices[i] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = EXIT_TSL if stop_loss_price > initial_stop_price else EXIT_SL
            elif high_prices[i] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = EXIT_TP
            
            # MA CROSSOVER LOGIC IS REMOVED

            if exit_reason > 0:
                pnl = (exit_price / entry_price - 1) - cost_pct
                trades.append((entry_time, timestamps[i], entry_price, exit_price, pnl, exit_reason))
                in_position = False

        if not in_position and entry_signals[i] == 1:
            in_position = True
            entry_time = timestamps[i]
            entry_price = close_prices[i]
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            initial_stop_price = stop_loss_price
            take_profit_price = entry_price * (1 + take_profit_pct / 100)
            
    return trades


@jit(nopython=True)
def run_backtest_simple_exits_short(
    # Data arrays
    timestamps,
    entry_signals,
    close_prices,
    high_prices,
    low_prices,
    ema_fast,
    ema_slow,
    # Parameters
    stop_loss_pct,
    take_profit_pct,
    cost_pct,
    trailing_stop_loss_pct=0.0
):
    """
    Numba-optimized backtesting loop for SHORT strategies with simple exit logic:
    1. Static Stop Loss (above entry)
    2. Trailing Stop Loss (moving down)
    3. Take Profit (below entry)
    4. Moving Average Crossover (9 EMA crossing above 15 EMA for shorts)
    """
    n = len(timestamps)
    trades = []
    
    in_position = False
    entry_time = 0
    entry_price = 0.0
    stop_loss_price = 0.0
    initial_stop_price = 0.0
    take_profit_price = 0.0
    
    EXIT_SL = 1
    EXIT_TP = 2
    EXIT_MA_CROSS = 3
    EXIT_TSL = 4

    for i in range(1, n):
        if in_position:
            # For shorts, trailing stop moves DOWN (tracks lower lows)
            if trailing_stop_loss_pct > 0:
                new_potential_stop = low_prices[i] * (1 + trailing_stop_loss_pct / 100)
                if new_potential_stop < stop_loss_price:
                    stop_loss_price = new_potential_stop

            exit_reason = 0
            exit_price = 0.0

            # For shorts: stop loss is ABOVE entry, take profit is BELOW
            if high_prices[i] >= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = EXIT_TSL if stop_loss_price < initial_stop_price else EXIT_SL
            elif low_prices[i] <= take_profit_price:
                exit_price = take_profit_price
                exit_reason = EXIT_TP
            # For shorts: exit when 9 EMA crosses ABOVE 15 EMA
            elif ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                exit_price = close_prices[i]
                exit_reason = EXIT_MA_CROSS

            if exit_reason > 0:
                # For shorts: profit when price goes DOWN
                pnl = (entry_price / exit_price - 1) - cost_pct
                trades.append((entry_time, timestamps[i], entry_price, exit_price, pnl, exit_reason))
                in_position = False

        # Check for short entry signal (-1)
        if not in_position and entry_signals[i] == -1:
            in_position = True
            entry_time = timestamps[i]
            entry_price = close_prices[i]
            # For shorts: stop loss is ABOVE entry price
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
            initial_stop_price = stop_loss_price
            # For shorts: take profit is BELOW entry price
            take_profit_price = entry_price * (1 - take_profit_pct / 100)
            
    return trades


@jit(nopython=True)
def run_backtest_simple_exits_short_no_ma_cross(
    # Data arrays
    timestamps,
    entry_signals,
    close_prices,
    high_prices,
    low_prices,
    ema_fast,
    ema_slow,
    # Parameters
    stop_loss_pct,
    take_profit_pct,
    cost_pct,
    trailing_stop_loss_pct=0.0
):
    """
    Numba-optimized backtesting loop for SHORT strategies without MA crossover exit.
    """
    n = len(timestamps)
    trades = []
    
    in_position = False
    entry_time = 0
    entry_price = 0.0
    stop_loss_price = 0.0
    initial_stop_price = 0.0
    take_profit_price = 0.0
    
    EXIT_SL = 1
    EXIT_TP = 2
    EXIT_TSL = 4

    for i in range(1, n):
        if in_position:
            # For shorts, trailing stop moves DOWN (tracks lower lows)
            if trailing_stop_loss_pct > 0:
                new_potential_stop = low_prices[i] * (1 + trailing_stop_loss_pct / 100)
                if new_potential_stop < stop_loss_price:
                    stop_loss_price = new_potential_stop

            exit_reason = 0
            exit_price = 0.0

            # For shorts: stop loss is ABOVE entry, take profit is BELOW
            if high_prices[i] >= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = EXIT_TSL if stop_loss_price < initial_stop_price else EXIT_SL
            elif low_prices[i] <= take_profit_price:
                exit_price = take_profit_price
                exit_reason = EXIT_TP

            if exit_reason > 0:
                # For shorts: profit when price goes DOWN
                pnl = (entry_price / exit_price - 1) - cost_pct
                trades.append((entry_time, timestamps[i], entry_price, exit_price, pnl, exit_reason))
                in_position = False

        # Check for short entry signal (-1)
        if not in_position and entry_signals[i] == -1:
            in_position = True
            entry_time = timestamps[i]
            entry_price = close_prices[i]
            # For shorts: stop loss is ABOVE entry price
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
            initial_stop_price = stop_loss_price
            # For shorts: take profit is BELOW entry price
            take_profit_price = entry_price * (1 - take_profit_pct / 100)
            
    return trades


# Dictionary to map exit types from config to functions
BACKTEST_FUNCTIONS = {
    'simple_exits': run_backtest_simple_exits,
    'simple_exits_no_ma_cross': run_backtest_simple_exits_no_ma_cross,
    'simple_exits_short': run_backtest_simple_exits_short,
    'simple_exits_short_no_ma_cross': run_backtest_simple_exits_short_no_ma_cross
}


# ##################################################################
# PERFORMANCE METRICS
# ##################################################################

def calculate_metrics(trade_log):
    """Calculates performance metrics from a trade log DataFrame."""
    if trade_log.empty:
        return {
            'Total Trades': 0, 'Win Rate (%)': 0, 'Profit Factor': 'N/A',
            'Total Return (%)': 0, 'Max Drawdown (%)': 0, 'Avg Duration (hrs)': 0,
            'Stop Loss Exits': 0, 'Trailing SL Exits': 0, 'Take Profit Exits': 0, 'MA Cross Exits': 0
        }
    
    total_trades = len(trade_log)
    winners = trade_log[trade_log['pnl_pct'] > 0]
    losers = trade_log[trade_log['pnl_pct'] < 0]
    
    win_rate = (len(winners) / total_trades) * 100 if total_trades > 0 else 0
    
    gross_profit = winners['pnl_pct'].sum()
    gross_loss = abs(losers['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    trade_log['cumulative_return'] = (1 + trade_log['pnl_pct'] / 100).cumprod()
    total_return = (trade_log['cumulative_return'].iloc[-1] - 1) * 100
    
    running_max = trade_log['cumulative_return'].cummax()
    drawdown = (trade_log['cumulative_return'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    trade_log['duration'] = (trade_log['exit_time'] - trade_log['entry_time']).dt.total_seconds() / 3600
    avg_duration = trade_log['duration'].mean()
    
    exit_reason_map = {1: 'Stop Loss', 2: 'Take Profit', 3: 'MA Cross', 4: 'Trailing SL'}
    exit_counts = trade_log['exit_reason'].map(exit_reason_map).value_counts()
    
    return {
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Avg Duration (hrs)': avg_duration,
        'Stop Loss Exits': exit_counts.get('Stop Loss', 0),
        'Trailing SL Exits': exit_counts.get('Trailing SL', 0),
        'Take Profit Exits': exit_counts.get('Take Profit', 0),
        'MA Cross Exits': exit_counts.get('MA Cross', 0)
    }