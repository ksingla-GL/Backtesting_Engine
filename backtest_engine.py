import numpy as np
import pandas as pd
from numba import jit

# ##################################################################
# HELPER FUNCTIONS FOR EXIT CONDITIONS
# ##################################################################

@jit(nopython=True)
def check_percentage_stop_loss(current_price, entry_price, stop_loss_pct, position_type):
    """Check if percentage-based stop loss is hit."""
    if position_type == 1:  # Long position
        stop_price = entry_price * (1 - stop_loss_pct / 100)
        return current_price <= stop_price, stop_price
    else:  # Short position
        stop_price = entry_price * (1 + stop_loss_pct / 100)
        return current_price >= stop_price, stop_price

@jit(nopython=True)
def check_atr_stop_loss(current_price, entry_price, atr_value, atr_multiplier, position_type):
    """Check if ATR-based stop loss is hit."""
    if position_type == 1:  # Long position
        stop_price = entry_price - (atr_value * atr_multiplier)
        return current_price <= stop_price, stop_price
    else:  # Short position
        stop_price = entry_price + (atr_value * atr_multiplier)
        return current_price >= stop_price, stop_price

@jit(nopython=True)
def check_take_profit(current_price, entry_price, take_profit_pct, position_type):
    """Check if take profit target is hit."""
    if position_type == 1:  # Long position
        target_price = entry_price * (1 + take_profit_pct / 100)
        return current_price >= target_price, target_price
    else:  # Short position
        target_price = entry_price * (1 - take_profit_pct / 100)
        return current_price <= target_price, target_price

@jit(nopython=True)
def update_trailing_stop(high_price, low_price, current_stop, trailing_pct, position_type):
    """Update trailing stop loss price."""
    if position_type == 1:  # Long position
        new_stop = high_price * (1 - trailing_pct / 100)
        return max(current_stop, new_stop)
    else:  # Short position
        new_stop = low_price * (1 + trailing_pct / 100)
        return min(current_stop, new_stop)

@jit(nopython=True)
def check_ma_crossover(ema_fast_curr, ema_slow_curr, ema_fast_prev, ema_slow_prev, position_type):
    """Check if MA crossover exit condition is met."""
    if position_type == 1:  # Long position - exit when fast MA crosses below slow MA
        return ema_fast_curr < ema_slow_curr and ema_fast_prev >= ema_slow_prev
    else:  # Short position - exit when fast MA crosses above slow MA
        return ema_fast_curr > ema_slow_curr and ema_fast_prev <= ema_slow_prev

@jit(nopython=True)
def check_price_below_ma(current_price, ma_price, position_type):
    """Check if price is below (for longs) or above (for shorts) the moving average."""
    if position_type == 1:  # Long position - exit when price below MA
        return current_price < ma_price
    else:  # Short position - exit when price above MA
        return current_price > ma_price

@jit(nopython=True)
def check_vix_spike(current_vix, entry_vix, spike_pct):
    """Check if VIX has spiked by specified percentage."""
    if entry_vix == 0.0 or current_vix == 0.0:
        return False  # No valid VIX data, no spike detected
    vix_change = (current_vix / entry_vix - 1) * 100
    return vix_change >= spike_pct

@jit(nopython=True)
def check_time_limit(current_time, entry_time, max_days):
    """Check if position has exceeded maximum holding period."""
    hours_held = (current_time - entry_time) / (1000 * 1000 * 1000 * 60 * 60)  # Convert nanoseconds to hours
    max_hours = max_days * 24
    return hours_held >= max_hours

# Exit reason codes
EXIT_SL_PERCENT = 1
EXIT_SL_ATR = 2
EXIT_TP = 3
EXIT_MA_CROSS = 4
EXIT_TSL = 5
EXIT_PRICE_BELOW_MA = 6
EXIT_VIX_SPIKE = 7
EXIT_TIME_LIMIT = 8

# ##################################################################
# UNIFIED MODULAR BACKTEST ENGINE
# ##################################################################

@jit(nopython=True)
def run_backtest_modular(
    # Data arrays
    timestamps,
    entry_signals,
    close_prices,
    high_prices,
    low_prices,
    ema_fast,
    ema_slow,
    atr_values,
    vix_closes,
    entry_vix_prices,
    # Exit condition flags
    use_pct_stop,
    use_atr_stop,
    use_take_profit,
    use_trailing_stop,
    use_ma_cross,
    use_price_below_ma,
    use_vix_spike,
    use_time_limit,
    # Position management flags
    use_pyramiding,
    # Parameters
    stop_loss_pct,
    stop_loss_atr_multiplier,
    take_profit_pct,
    trailing_stop_loss_pct,
    vix_spike_pct,
    max_days,
    cost_pct,
    position_type=1,  # 1 for long, -1 for short
    # Pyramiding parameters
    max_pyramid_levels=4,
    pyramid_scale_factor=2.0
):
    """
    Unified modular backtesting engine that can enable/disable any combination of exit conditions.
    Now supports pyramiding position management with any exit condition combination.

    Exit priority order (first hit wins):
      1) Percentage SL, 2) ATR SL, 3) Trailing SL,
      4) Take Profit, 5) MA Cross, 6) Price vs MA,
      7) VIX Spike, 8) Time Limit
    """
    n = len(timestamps)
    trades = []
    
    # Position tracking variables
    if use_pyramiding:
        # Pyramiding mode - track position size and weighted averages
        position_size = 0.0  # 0, 1, 2, 4, 8...
        pyramid_levels = 0
        weighted_entry_price = 0.0
        in_position = False  # For compatibility with non-pyramiding logic
    else:
        # Regular mode - simple position tracking
        in_position = False
        entry_price = 0.0
    
    entry_time = 0
    entry_index = 0  # Track data index when entry occurred
    
    # Common variables
    entry_vix = 0.0
    stop_loss_price = 0.0
    # VIX baseline captured on entry and reused for duration of position
    
    for i in range(1, n):
        # Handle position management (pyramiding vs regular)
        current_position = position_size > 0 if use_pyramiding else in_position
        
        if current_position:
            # Get current entry price (weighted for pyramiding, regular for single position)
            current_entry_price = weighted_entry_price if use_pyramiding else entry_price
            
            # Initialize trailing stop price before updating if it's zero
            if use_trailing_stop and trailing_stop_loss_pct > 0 and stop_loss_price == 0.0:
                if position_type == 1:
                    stop_loss_price = current_entry_price * (1 - trailing_stop_loss_pct / 100)
                else:
                    stop_loss_price = current_entry_price * (1 + trailing_stop_loss_pct / 100)
            
            # Update trailing stop
            if use_trailing_stop and trailing_stop_loss_pct > 0:
                stop_loss_price = update_trailing_stop(
                    high_prices[i], low_prices[i], stop_loss_price, 
                    trailing_stop_loss_pct, position_type
                )
            
            current_price = close_prices[i]
            
            # Check all enabled exit conditions - FIXED: Only check conditions that are actually enabled
            exit_reason = 0
            exit_price = 0.0
            
            # Check percentage stop loss (only if enabled)
            if exit_reason == 0 and use_pct_stop:
                pct_hit, pct_price = check_percentage_stop_loss(
                    low_prices[i] if position_type == 1 else high_prices[i],
                    current_entry_price, stop_loss_pct, position_type
                )
                if pct_hit:
                    exit_price = pct_price
                    exit_reason = EXIT_SL_PERCENT
            
            # Check ATR stop loss (only if enabled and no previous exit)
            if exit_reason == 0 and use_atr_stop and i < len(atr_values):
                atr_hit, atr_price = check_atr_stop_loss(
                    low_prices[i] if position_type == 1 else high_prices[i],
                    current_entry_price, atr_values[i], stop_loss_atr_multiplier, position_type
                )
                if atr_hit:
                    exit_price = atr_price
                    exit_reason = EXIT_SL_ATR
            
            # Check trailing stop (only if enabled and trailing is active)
            if exit_reason == 0 and use_trailing_stop and stop_loss_price > 0:
                if position_type == 1 and low_prices[i] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = EXIT_TSL
                elif position_type == -1 and high_prices[i] >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = EXIT_TSL
            
            if exit_reason == 0 and use_take_profit:
                tp_hit, tp_price = check_take_profit(
                    high_prices[i] if position_type == 1 else low_prices[i],
                    current_entry_price, take_profit_pct, position_type
                )
                if tp_hit:
                    exit_price = tp_price
                    exit_reason = EXIT_TP
            
            if exit_reason == 0 and use_ma_cross and i < len(ema_fast) and i < len(ema_slow) and i > 0:
                if check_ma_crossover(ema_fast[i], ema_slow[i], ema_fast[i-1], ema_slow[i-1], position_type):
                    exit_price = current_price
                    exit_reason = EXIT_MA_CROSS
            
            if exit_reason == 0 and use_price_below_ma and i < len(ema_fast):
                if check_price_below_ma(current_price, ema_fast[i], position_type):
                    exit_price = current_price
                    exit_reason = EXIT_PRICE_BELOW_MA
            
            if exit_reason == 0 and use_vix_spike and i < len(vix_closes):
                # Use fixed VIX baseline from entry
                if entry_vix > 0.0 and check_vix_spike(vix_closes[i], entry_vix, vix_spike_pct):
                    exit_price = current_price
                    exit_reason = EXIT_VIX_SPIKE
            
            if exit_reason == 0 and use_time_limit:
                if check_time_limit(timestamps[i], entry_time, max_days):
                    exit_price = current_price
                    exit_reason = EXIT_TIME_LIMIT
            
            # Process exit if any condition is met
            if exit_reason > 0:
                if use_pyramiding:
                    # Pyramiding exit - close entire pyramid
                    # Scale costs by number of fills (entries + 1 exit)
                    fills_count = pyramid_levels + 1
                    effective_cost = cost_pct * fills_count
                    if position_type == 1:
                        pnl = (exit_price / weighted_entry_price - 1) - effective_cost
                    else:  # Short position
                        pnl = (weighted_entry_price / exit_price - 1) - effective_cost
                    
                    # Create trade record with position size and entry index
                    trades.append((entry_time, timestamps[i], weighted_entry_price, exit_price, pnl, exit_reason, position_size, entry_index))
                    
                    # Reset pyramiding state
                    position_size = 0.0
                    pyramid_levels = 0
                    weighted_entry_price = 0.0
                    in_position = False
                    entry_vix = 0.0
                else:
                    # Regular exit - add position size 1.0 for consistency
                    if position_type == 1:
                        pnl = (exit_price / entry_price - 1) - cost_pct
                    else:  # Short position
                        pnl = (entry_price / exit_price - 1) - cost_pct
                    
                    trades.append((entry_time, timestamps[i], entry_price, exit_price, pnl, exit_reason, 1.0, entry_index))
                    in_position = False
                    entry_vix = 0.0
        
        # Check for entry signals
        if use_pyramiding:
            entry_condition = (position_size == 0.0 or (pyramid_levels < max_pyramid_levels))
        else:
            entry_condition = not current_position
            
        # Check for appropriate entry signal based on position type
        signal_match = False
        if position_type == 1 and entry_signals[i] == 1:  # Long signal for long strategy
            signal_match = True
        elif position_type == -1 and entry_signals[i] == -1:  # Short signal for short strategy  
            signal_match = True
        elif entry_signals[i] == 1:  # Default to long signal for backward compatibility
            signal_match = True
        
        if entry_condition and signal_match:
            current_price = close_prices[i]
            
            if use_pyramiding:
                if position_size == 0.0:
                    # First pyramid entry
                    position_size = 1.0
                    pyramid_levels = 1
                    weighted_entry_price = current_price
                    entry_time = timestamps[i]
                    entry_index = i  # Capture entry data index
                    in_position = True
                    # Capture VIX baseline at entry (prefer prior day's close if available)
                    if i < len(entry_vix_prices) and entry_vix_prices[i] > 0.0:
                        entry_vix = entry_vix_prices[i]
                    elif i < len(vix_closes):
                        entry_vix = vix_closes[i]
                else:
                    # Additional pyramid entry - multiplicative scaling
                    # Each new level size = previous total * scale_factor
                    new_total_size = position_size * pyramid_scale_factor
                    new_level_size = new_total_size - position_size  # Additional units being added
                    
                    # Update weighted average entry price
                    weighted_entry_price = ((weighted_entry_price * position_size) + (current_price * new_level_size)) / new_total_size
                    
                    position_size = new_total_size
                    pyramid_levels += 1
            else:
                # Regular single position entry
                in_position = True
                entry_time = timestamps[i]
                entry_index = i  # Capture entry data index
                entry_price = current_price
                # Capture VIX baseline at entry (prefer prior day's close if available)
                if i < len(entry_vix_prices) and entry_vix_prices[i] > 0.0:
                    entry_vix = entry_vix_prices[i]
                elif i < len(vix_closes):
                    entry_vix = vix_closes[i]
            
            # Initialize stop loss price using current entry price
            current_entry_ref = weighted_entry_price if use_pyramiding else entry_price
            
            if use_pct_stop and use_atr_stop and i < len(atr_values):
                # Use stricter of the two
                _, pct_stop = check_percentage_stop_loss(0, current_entry_ref, stop_loss_pct, position_type)
                _, atr_stop = check_atr_stop_loss(0, current_entry_ref, atr_values[i], stop_loss_atr_multiplier, position_type)
                if position_type == 1:
                    stop_loss_price = max(pct_stop, atr_stop)  # Stricter (higher) for longs
                else:
                    stop_loss_price = min(pct_stop, atr_stop)  # Stricter (lower) for shorts
            elif use_pct_stop:
                _, stop_loss_price = check_percentage_stop_loss(0, current_entry_ref, stop_loss_pct, position_type)
            elif use_atr_stop and i < len(atr_values):
                _, stop_loss_price = check_atr_stop_loss(0, current_entry_ref, atr_values[i], stop_loss_atr_multiplier, position_type)
            elif use_trailing_stop and trailing_stop_loss_pct > 0:
                # Initialize trailing stop at current price for activation
                if position_type == 1:
                    stop_loss_price = current_entry_ref * (1 - trailing_stop_loss_pct / 100)
                else:
                    stop_loss_price = current_entry_ref * (1 + trailing_stop_loss_pct / 100)
            else:
                stop_loss_price = 0.0
            
            # No separate initial TP/SL tracking needed; TP/SL evaluated each bar
    
    return trades

# ##################################################################
# MODULAR SYSTEM CONFIGURATION
# ##################################################################

# Exit condition mapping for modular system
EXIT_CONDITIONS_MAP = {
    'percentage_stop': 'use_pct_stop',
    'atr_stop': 'use_atr_stop', 
    'take_profit': 'use_take_profit',
    'trailing_stop': 'use_trailing_stop',
    'ma_crossover': 'use_ma_cross',
    'price_below_ma': 'use_price_below_ma',
    'vix_spike': 'use_vix_spike',
    'time_limit': 'use_time_limit'
}

# Dictionary to map exit types from config to functions
BACKTEST_FUNCTIONS = {
    'modular': run_backtest_modular
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
    
    exit_reason_map = {
        1: 'Stop Loss', 
        2: 'ATR Stop Loss',
        3: 'Take Profit', 
        4: 'MA Cross', 
        5: 'Trailing SL',
        6: 'Price Below MA',
        7: 'VIX Spike',
        8: 'Time Exit (Days)'
    }
    exit_counts = trade_log['exit_reason'].map(exit_reason_map).value_counts()
    
    return {
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Avg Duration (hrs)': avg_duration,
        'Stop Loss Exits': exit_counts.get('Stop Loss', 0),
        'ATR Stop Loss Exits': exit_counts.get('ATR Stop Loss', 0),
        'Trailing SL Exits': exit_counts.get('Trailing SL', 0),
        'Take Profit Exits': exit_counts.get('Take Profit', 0),
        'MA Cross Exits': exit_counts.get('MA Cross', 0),
        'Price Below MA Exits': exit_counts.get('Price Below MA', 0),
        'VIX Spike Exits': exit_counts.get('VIX Spike', 0),
        'Time Exit (Days) Exits': exit_counts.get('Time Exit (Days)', 0)
    }

def get_exit_reason_description(exit_reason_code):
    """Convert exit reason code to description"""
    exit_reason_map = {
        1: 'Stop Loss',
        2: 'ATR Stop Loss', 
        3: 'Take Profit',
        4: 'MA Cross',
        5: 'Trailing SL',
        6: 'Price Below MA',
        7: 'VIX Spike',
        8: 'Time Exit (Days)'
    }
    return exit_reason_map.get(exit_reason_code, f'Unknown ({exit_reason_code})')
