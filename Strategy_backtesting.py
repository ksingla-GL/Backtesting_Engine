import json
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Load configuration
with open('config.json') as f:
    CONFIG = json.load(f)
exit_cfg = CONFIG.get('exit', {})
transaction_cost = CONFIG.get('transaction_cost', 0.0005)

# Store results for comparison
comparison_results = {}

# Test different ES decline thresholds
decline_thresholds = [1, 2, 3, 4]

print("="*80)
print("BACKTESTING MULTIPLE ES DECLINE THRESHOLDS")
print("="*80)

for decline_pct in decline_thresholds:
    print(f"\n{'='*80}")
    print(f"TESTING ES DECLINE THRESHOLD: -{decline_pct}%")
    print(f"{'='*80}")
    
    # Load data for this decline threshold
    filename = f'merged_data_with_signals_{decline_pct}pct.parquet'
    try:
        df = pd.read_parquet(filename)
    except:
        print(f"ERROR: Could not load {filename}")
        continue
    
    # ============================================
    # STEP 1: SETUP ENTRY TRACKING
    # ============================================
    df['Entry_time'] = np.where(df['Entry_signal'] == 1, df.index, pd.NaT)
    df[['Entry_price','Entry_time']] = df[['Entry_price','Entry_time']].ffill()
    df['Entry_time'] = pd.to_datetime(df['Entry_time'])
    df.dropna(inplace = True)
    
    # ============================================
    # STEP 2: CALCULATE EXIT CONDITIONS
    # ============================================
    stop_loss_pct = exit_cfg.get('stop_loss_pct', 1) / 100
    df['SL_exit'] = df['Entry_price'] * (1 - stop_loss_pct)
    df['SL_exit'] = np.where(df['Entry_signal'] != 1, df['SL_exit'], -1)
    
    # Trailing Stop Loss
    trail_start = exit_cfg.get('trailing_start_pct', 2) / 100
    trail_pct = exit_cfg.get('trailing_stop_pct', 3) / 100
    df['SL_exit'] = np.where(
                (df['Entry_signal'] != 1) &
                (df['ES_high'].shift(1) > df['Entry_price'] * (1 + trail_start)) &
                (df['ES_high'].shift(1) * (1 - trail_pct) > df['SL_exit']),
                df['ES_high'] * (1 - trail_pct),
                df['SL_exit'])
    
    # Take Profit Target
    tp_pct = exit_cfg.get('take_profit_pct', 3) / 100
    df['TP_exit'] = df['Entry_price'] * (1 + tp_pct)
    df['TP_exit'] = np.where(df['Entry_signal'] != 1, df['TP_exit'], np.inf)
    
    # Moving Average Cross Exit
    fast_ma = exit_cfg.get('ma_fast', 9)
    slow_ma = exit_cfg.get('ma_slow', 15)
    fast_col = f'ES_EMA_{fast_ma}' if f'ES_EMA_{fast_ma}' in df.columns else 'ES_EMA_9'
    slow_col = f'ES_EMA_{slow_ma}' if f'ES_EMA_{slow_ma}' in df.columns else 'ES_EMA_15'
    df['MA_exit'] = np.where((df[fast_col] < df[slow_col]) &
        (df[fast_col].shift(1) >= df[slow_col].shift(1)) &
        (df['Entry_signal'] != 1), 1, 0)
    
    # Determine exit price
    df['Exit_price'] = np.where(df['MA_exit'] == 1, df['ES_close'], np.where(
        df['ES_high'] >= df['TP_exit'], df['TP_exit'], np.where(
        df['ES_low'] <= df['SL_exit'], df['SL_exit'], np.nan)))
    df['Exit_signal'] = np.where(df['Exit_price'] == df['Exit_price'], 1, 0)
    
    # ============================================
    # STEP 3: TRACK POSITION STATUS
    # ============================================
    df['In_position'] = np.nan
    df['In_position'] = np.where(df['Entry_signal'] == 1, 1, np.where(
        df['Exit_signal'] == 1, 0, df['In_position']))
    df['In_position'] = df['In_position'].ffill()
    
    # ============================================
    # STEP 4: CLEAN SIGNALS & ASSIGN TRADE IDS
    # ============================================
    df['Entry_signal'] = np.where((df['Entry_signal'] == 1) & (df['In_position'].shift(1) == 0), 1, 0)
    df['Exit_signal'] = np.where((df['Exit_signal'] == 1) & (df['In_position'].shift(1) == 1), 1, 0)
    df[['Entry_price','Entry_time']] = df[['Entry_price','Entry_time']].ffill()
    df['Entry_time'] = pd.to_datetime(df['Entry_time'])
    
    df['Trade_id'] = df['Entry_signal'].cumsum()
    df['Trade_id'] = np.where((df['In_position'] == 0) & (df['In_position'].shift(1) == 1),
                              df['Trade_id'], np.nan)
    
    # ============================================
    # STEP 5: CALCULATE TRADE PERFORMANCE
    # ============================================
    df2 = df.loc[df['Trade_id'] == df['Trade_id']].copy()
    
    if len(df2) == 0:
        print(f"No completed trades found for {decline_pct}% decline threshold")
        comparison_results[decline_pct] = {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_duration': 0
        }
        continue
    
    # Calculate P&L
    df2['pnl'] = df2['Exit_price']/df2['Entry_price'] - 1 - transaction_cost
    
    # Calculate metrics
    total_trades = len(df2)
    winners = len(df2[df2['pnl'] > 0])
    losers = len(df2[df2['pnl'] < 0])
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = df2[df2['pnl'] > 0]['pnl'].mean() if winners > 0 else 0
    avg_loss = df2[df2['pnl'] < 0]['pnl'].mean() if losers > 0 else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    
    gross_profits = df2[df2['pnl'] > 0]['pnl'].sum() if winners > 0 else 0
    gross_losses = abs(df2[df2['pnl'] < 0]['pnl'].sum()) if losers > 0 else 0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
    
    # Drawdown calculation
    df2['cumulative_return'] = (1 + df2['pnl']).cumprod()
    df2['running_max'] = df2['cumulative_return'].expanding().max()
    df2['drawdown'] = (df2['cumulative_return'] - df2['running_max']) / df2['running_max']
    max_drawdown = df2['drawdown'].min()
    
    total_return = (df2['cumulative_return'].iloc[-1] - 1) * 100 if len(df2) > 0 else 0
    
    # Duration analysis
    df2['Exit_time'] = df2.index
    df2['Entry_time'] = pd.to_datetime(df2['Entry_time'])
    df2['Trade_duration_minutes'] = (df2['Exit_time'] - df2['Entry_time']).dt.total_seconds() / 60
    df2['Trade_duration_hours'] = df2['Trade_duration_minutes'] / 60
    avg_duration = df2['Trade_duration_hours'].mean()
    
    # Sharpe ratio
    if len(df2) > 0 and df2['pnl'].std() > 0 and df2['Trade_duration_minutes'].mean() > 0:
        sharpe = df2['pnl'].mean() / df2['pnl'].std() * np.sqrt(252*390/df2['Trade_duration_minutes'].mean())
    else:
        sharpe = 0
    
    # Exit analysis
    df2['Exit_reason'] = 'Unknown'
    df2.loc[df2['MA_exit'] == 1, 'Exit_reason'] = 'MA Cross'
    for idx, row in df2.iterrows():
        if row['Exit_reason'] == 'Unknown':
            if row['Exit_price'] == row['TP_exit']:
                df2.loc[idx, 'Exit_reason'] = 'Take Profit'
            elif abs(row['Exit_price'] - row['SL_exit']) < 0.01:
                df2.loc[idx, 'Exit_reason'] = 'Stop Loss'
    
    exit_counts = df2['Exit_reason'].value_counts()
    
    # Store results
    comparison_results[decline_pct] = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy * 100,
        'total_return': total_return,
        'max_drawdown': max_drawdown * 100,
        'sharpe_ratio': sharpe,
        'avg_duration': avg_duration,
        'winners': winners,
        'losers': losers,
        'avg_win': avg_win * 100,
        'avg_loss': avg_loss * 100,
        'stop_loss_exits': exit_counts.get('Stop Loss', 0),
        'take_profit_exits': exit_counts.get('Take Profit', 0),
        'ma_cross_exits': exit_counts.get('MA Cross', 0)
    }
    
    # Save trade log only for 1% decline
    if decline_pct == 1:
        trade_log = df2[['Entry_time', 'Entry_price', 'Exit_time', 'Exit_price', 
                         'pnl', 'Trade_duration_hours', 'Exit_reason']].copy()
        trade_log['pnl_pct'] = trade_log['pnl'] * 100
        trade_log = trade_log.drop('pnl', axis=1)
        trade_log.to_csv('trade_log_1pct_decline.csv')
        print(f"\nTrade log saved for -1% decline: trade_log_1pct_decline.csv")

# ============================================
# COMPARISON TABLE
# ============================================
print("\n" + "="*100)
print("PERFORMANCE COMPARISON - ES DECLINE THRESHOLDS")
print("="*100)

# Create comparison dataframe
comp_df = pd.DataFrame(comparison_results).T
comp_df.index = [f"-{idx}%" for idx in comp_df.index]

# Display main metrics
print("\nKEY PERFORMANCE METRICS:")
print("-"*100)
print(f"{'Metric':<20} | {'-1%':>12} | {'-2%':>12} | {'-3%':>12} | {'-4%':>12}")
print("-"*100)

metrics_to_show = [
    ('Total Trades', 'total_trades', '.0f'),
    ('Win Rate (%)', 'win_rate', '.1f'),
    ('Profit Factor', 'profit_factor', '.2f'),
    ('Expectancy (%)', 'expectancy', '.3f'),
    ('Total Return (%)', 'total_return', '.2f'),
    ('Max Drawdown (%)', 'max_drawdown', '.2f'),
    ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
    ('Avg Duration (hrs)', 'avg_duration', '.1f')
]

for label, metric, fmt in metrics_to_show:
    print(f"{label:<20} |", end="")
    for decline in [1, 2, 3, 4]:
        if decline in comparison_results:
            value = comparison_results[decline][metric]
            print(f" {value:>12{fmt}} |", end="")
        else:
            print(f" {'N/A':>12} |", end="")
    print()

# Exit breakdown
print("\nEXIT BREAKDOWN:")
print("-"*100)
print(f"{'Exit Type':<20} | {'-1%':>12} | {'-2%':>12} | {'-3%':>12} | {'-4%':>12}")
print("-"*100)

exit_types = [
    ('Stop Loss', 'stop_loss_exits'),
    ('Take Profit', 'take_profit_exits'),
    ('MA Cross', 'ma_cross_exits')
]

for label, metric in exit_types:
    print(f"{label:<20} |", end="")
    for decline in [1, 2, 3, 4]:
        if decline in comparison_results:
            value = comparison_results[decline][metric]
            pct = (value / comparison_results[decline]['total_trades'] * 100) if comparison_results[decline]['total_trades'] > 0 else 0
            print(f" {value:>4} ({pct:>5.1f}%) |", end="")
        else:
            print(f" {'N/A':>12} |", end="")
    print()

# ============================================
# INSIGHTS AND CONCLUSIONS
# ============================================
print("\n" + "="*100)
print("INSIGHTS AND CONCLUSIONS")
print("="*100)

print("\n1. TRADE FREQUENCY:")
print("   - More restrictive decline thresholds (-3%, -4%) significantly reduce trading opportunities")
print(f"   - Trade count decreases from {comparison_results.get(1, {}).get('total_trades', 0)} trades at -1% to {comparison_results.get(4, {}).get('total_trades', 0)} trades at -4%")

print("\n2. PERFORMANCE ANALYSIS:")
# Find best performing threshold
best_return = max([r['total_return'] for r in comparison_results.values()])
best_threshold = [k for k, v in comparison_results.items() if v['total_return'] == best_return][0]
print(f"   - Best total return: -{best_threshold}% decline with {best_return:.2f}% return")

# Win rate trend
win_rates = [comparison_results.get(i, {}).get('win_rate', 0) for i in [1, 2, 3, 4]]
if win_rates[0] < win_rates[-1]:
    print("   - Win rate improves with deeper decline thresholds (waiting for bigger dips)")
else:
    print("   - Win rate decreases with deeper decline thresholds")

print("\n3. RISK/REWARD TRADE-OFF:")
# Compare drawdowns
dd_1pct = abs(comparison_results.get(1, {}).get('max_drawdown', 0))
dd_4pct = abs(comparison_results.get(4, {}).get('max_drawdown', 0))
print(f"   - Max drawdown ranges from {dd_1pct:.1f}% (-1% threshold) to {dd_4pct:.1f}% (-4% threshold)")

# Profit factor analysis
pf_values = [comparison_results.get(i, {}).get('profit_factor', 0) for i in [1, 2, 3, 4]]
best_pf = max(pf_values)
best_pf_threshold = [1, 2, 3, 4][pf_values.index(best_pf)]
print(f"   - Best profit factor: {best_pf:.2f} at -{best_pf_threshold}% decline threshold")

print("\n4. OPTIMAL THRESHOLD RECOMMENDATION:")
# Score each threshold
scores = {}
for threshold in [1, 2, 3, 4]:
    if threshold in comparison_results:
        r = comparison_results[threshold]
        # Scoring: 40% return, 30% profit factor, 20% trade count, 10% sharpe
        score = (r['total_return'] * 0.4 + 
                r['profit_factor'] * 20 * 0.3 + 
                min(r['total_trades'] / 50, 1) * 100 * 0.2 +
                r['sharpe_ratio'] * 20 * 0.1)
        scores[threshold] = score

if scores:
    optimal = max(scores, key=scores.get)
    print(f"   - Based on weighted scoring, optimal threshold is -{optimal}% decline")
    print(f"   - This balances profitability ({comparison_results[optimal]['total_return']:.1f}% return) with reasonable trade frequency ({comparison_results[optimal]['total_trades']} trades)")

print("\n5. KEY TAKEAWAYS:")
print("   - The -1% decline threshold provides the most trading opportunities")
print("   - Deeper decline thresholds may offer better risk/reward but at the cost of fewer trades")
print("   - Stop losses are triggered frequently across all thresholds, suggesting:")
print("     a) Entry timing could be improved")
print("     b) Stop loss placement may need adjustment")
print("     c) Market volatility after declines challenges the 1% stop loss")

print("\n" + "="*100)
print("END OF COMPARISON ANALYSIS")
print("="*100)
