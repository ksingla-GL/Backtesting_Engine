import json
import pandas as pd
import numpy as np

# Load configuration
with open('config.json') as f:
    CONFIG = json.load(f)
entry_cfg = CONFIG.get('entry', {})

# Load the merged data
df = pd.read_parquet('merged_data_1min.parquet')

# Check what columns we have
print("Checking for required columns:")
required_cols = ['ES_close', 'ES_SMA_50', 'VIX_close', 'VX_close', 'VX_Prev_day_close', 
                 'ES_High_10D', 'ES_RSI_2', 'TRIN_close', 'VIX_VXV_RATIO']
for col in required_cols:
    if col in df.columns:
        print(f"✓ {col}")
    else:
        print(f"✗ {col} - MISSING")

# Compute VX spike using the pre-calculated previous day close
df['VX_spike'] = (df['VX_close'] / df['VX_Prev_day_close'] - 1) * 100

# Compute ES decline from recent peak using configured rolling window
rolling_days = entry_cfg.get('rolling_high_days', 10)
high_col = f'ES_High_{rolling_days}D'
if high_col in df.columns:
    df['ES_decline_from_peak'] = (df['ES_close'] / df[high_col] - 1) * 100
else:
    df['ES_decline_from_peak'] = (df['ES_close'] / df['ES_High_10D'] - 1) * 100

# Test different ES decline thresholds
decline_thresholds = [-1, -2, -3, -4]

print("\n" + "="*60)
print("GENERATING SIGNALS FOR MULTIPLE ES DECLINE THRESHOLDS")
print("="*60)

for decline_pct in decline_thresholds:
    print(f"\n--- ES Decline Threshold: {decline_pct}% ---")
    
    # Create signal conditions with current decline threshold
    signal_conditions = (
        (df['ES_close'] > df['ES_SMA_50']) &
        (df['VIX_close'] < entry_cfg.get('vix_max', 25)) &
        (df['ES_decline_from_peak'] <= decline_pct) &  # Use current threshold
        (df['VX_spike'] < entry_cfg.get('vx_spike_max', 25)) &
        (df['ES_RSI_2'] > entry_cfg.get('rsi2_min', 10))
    )
    
    # Add other conditions if available
    signal_conditions &= (df['TRIN_close'] < entry_cfg.get('trin_max', 1.5))
    signal_conditions &= (df['VIX_VXV_RATIO'] < entry_cfg.get('vix_vxv_ratio_max', 1.0))
    signal_conditions &= (df['FED_STANCE'] <= entry_cfg.get('fed_stance_max', 2))
    signal_conditions &= (df['CNN_FEAR_GREED'] > entry_cfg.get('cnn_index_min', 35))
    signal_conditions &= (df['NAAIM'] > entry_cfg.get('naaim_min', 40))
    signal_conditions &= (df['BUFFETT_INDICATOR'] < entry_cfg.get('buffett_ratio_max', 200))
    signal_conditions &= (df['MARKET_BREADTH'] > entry_cfg.get('market_breadth_min', 40))
    
    # Convert to int
    df[f'quick_panic_signal_{abs(decline_pct)}pct'] = signal_conditions.astype(int)
    
    # Display initial results
    print(f"Total signals before filtering: {df[f'quick_panic_signal_{abs(decline_pct)}pct'].sum()}")
    
    signal_mask = df[f'quick_panic_signal_{abs(decline_pct)}pct'] == 1
    if signal_mask.any():
        # Group by date and keep only first signal
        first_signals = df[signal_mask].groupby(df[signal_mask].index.date).first()
        
        # Reset all signals to 0
        df[f'quick_panic_signal_{abs(decline_pct)}pct'] = 0
        
        # Set only the first signal of each day to 1
        for date, row in first_signals.iterrows():
            # Find the first minute of this day with a signal
            day_mask = (df.index.date == date) & signal_mask
            first_signal_time = df[day_mask].index[0]
            df.loc[first_signal_time, f'quick_panic_signal_{abs(decline_pct)}pct'] = 1
    
    # Calculate statistics
    total_trading_days = df.index.normalize().nunique()
    days_with_signals = df[df[f'quick_panic_signal_{abs(decline_pct)}pct'] == 1].index.normalize().nunique()
    signal_percentage = (days_with_signals / total_trading_days) * 100
    
    print(f"Total trading days: {total_trading_days}")
    print(f"Days with signals: {days_with_signals}")
    print(f"Percentage of days with signals: {signal_percentage:.2f}%")
    
    # Create entry price column
    df[f'Entry_price_{abs(decline_pct)}pct'] = np.where(
        df[f'quick_panic_signal_{abs(decline_pct)}pct'] == 1,
        df['ES_close'],
        np.nan
    )
    
    # Prepare dataframe for this decline threshold
    needed_cols = ['ES_close', 'ES_high', 'ES_low', 'ES_EMA_9', 'ES_EMA_15', 
                  f'quick_panic_signal_{abs(decline_pct)}pct', f'Entry_price_{abs(decline_pct)}pct']
    df_output = df[needed_cols].copy()
    
    # Rename columns to standard format
    df_output.columns = ['ES_close', 'ES_high', 'ES_low', 'ES_EMA_9', 'ES_EMA_15', 
                        'Entry_signal', 'Entry_price']
    
    # Save with signals
    filename = f'merged_data_with_signals_{abs(decline_pct)}pct.parquet'
    df_output.to_parquet(filename)
    print(f"Saved: {filename}")

print("\n" + "="*60)
print("ALL SIGNAL FILES GENERATED SUCCESSFULLY")
print("="*60)

# Summary comparison
print("\nSUMMARY COMPARISON:")
print("ES Decline | Days with Signals | Percentage")
print("-"*45)
for decline_pct in decline_thresholds:
    col_name = f'quick_panic_signal_{abs(decline_pct)}pct'
    days_with_signals = df[df[col_name] == 1].index.normalize().nunique()
    signal_percentage = (days_with_signals / total_trading_days) * 100
    print(f"   {decline_pct:>4}%   |       {days_with_signals:>3}        | {signal_percentage:>6.2f}%")
