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
    
    if name not in ['IsUpDay', 'MACD', 'ConsecutiveGreenDays', 'BreakoutRetest', 'OBVHighN', 'PreEventWeakness', 'PostEventVIXDecline', 'IntraydayESDecline', 'MarketBreadthStrong', 'VWAPReclaim', 'FedDovishSurprise', 'PostEventRally', 'PostEventTime'] and on_col and on_col not in df.columns:
        logging.error(f"Input column '{on_col}' not found for indicator {name}")
        df[out_col] = np.nan
        return df
    
    if name not in ['IsUpDay', 'MACD', 'ConsecutiveGreenDays', 'BreakoutRetest', 'OBVHighN', 'PreEventWeakness', 'PostEventVIXDecline', 'IntraydayESDecline', 'MarketBreadthStrong', 'VWAPReclaim', 'FedDovishSurprise', 'PostEventRally', 'PostEventTime'] and on_col and df[on_col].isna().all():
        logging.error(f"Input column '{on_col}' is all NaN for indicator {name}")
        df[out_col] = np.nan
        return df
    
    try:
        if name == 'SMA':
            window = params.get('window', 50)
            daily_open = df[on_col].resample('D').first().dropna()
            daily_sma = daily_open.rolling(window=window, min_periods=window).mean()
            df[out_col] = daily_sma.reindex(df.index, method='ffill').ffill()
                
        elif name == 'EMA':
            span = params.get('span', params.get('window'))
            daily_open = df[on_col].resample('D').first().dropna()
            daily_ema = daily_open.ewm(span=span, adjust=False).mean()
            df[out_col] = daily_ema.reindex(df.index, method='ffill').ffill()
                
        elif name == 'RSI':
            window = params.get('window', 2)
            daily_open = df[on_col].resample('D').first().dropna()
            daily_rsi = ta.momentum.rsi(daily_open, window=window)
            df[out_col] = daily_rsi.reindex(df.index, method='ffill').ffill()
        
        elif name == 'ATR':
            window = params.get('window', 14)
            # ATR requires high, low, close prices
            high_col = params.get('high_col', 'ES_high')
            low_col = params.get('low_col', 'ES_low')
            close_col = params.get('close_col', 'ES_close')
            
            if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
                logging.error(f"ATR requires {high_col}, {low_col}, {close_col} columns")
                df[out_col] = np.nan
                return df
            
            # Calculate on daily data
            daily_high = df[high_col].resample('D').max().shift(1).dropna()
            daily_low = df[low_col].resample('D').min().shift(1).dropna()
            daily_close = df[close_col].resample('D').last().shift(1).dropna()
            
            # Calculate True Range
            prev_close = daily_close
            tr1 = daily_high - daily_low
            tr2 = abs(daily_high - prev_close)
            tr3 = abs(daily_low - prev_close)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as rolling mean of True Range
            daily_atr = true_range.rolling(window=window, min_periods=window).mean().shift(1)
            df[out_col] = daily_atr.reindex(df.index, method='ffill').ffill()
            
            logging.info(f"    ATR calculated with window {window}, values: {df[out_col].notna().sum()}")
        
        elif name == 'RollingHigh':
            daily_data = df[on_col].resample('D').max()
            window_size = params.get('window', 10)
            rolling_data = daily_data.rolling(window=window_size, min_periods=1).max().shift(1).dropna()
            df[out_col] = rolling_data.reindex(df.index, method='ffill').ffill()
        
        elif name == 'RollingLow':
            daily_data = df[on_col].resample('D').min()
            window_size = params.get('window', 10)
            rolling_data = daily_data.rolling(window=window_size, min_periods=1).min().shift(1).dropna()
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
            
            # Calculate on daily data using opens
            daily_open = df[on_col].resample('D').first().dropna()
            
            # Calculate MACD line
            ema_fast = daily_open.ewm(span=fast, adjust=False).mean()
            ema_slow = daily_open.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculate histogram
            macd_histogram = macd_line - signal_line
            
            # Reindex to 1-minute
            df[out_col] = macd_histogram.reindex(df.index, method='ffill').ffill()

        # NEW INDICATORS START HERE
        elif name == 'PriceAtTime':
            """Capture price at specific time"""
            target_time = params.get('time', '08:25:00')
            target_hour = int(target_time.split(':')[0])
            target_minute = int(target_time.split(':')[1])
            
            # Get prices at target time
            target_mask = (df.index.hour == target_hour) & (df.index.minute == target_minute)
            
            if target_mask.sum() > 0:
                # Create a Series with date as index and price as value
                daily_prices = df[target_mask][on_col].copy()
                daily_prices.index = daily_prices.index.date
                
                # Map prices to all timestamps
                df['_temp_date'] = df.index.date
                df[out_col] = df['_temp_date'].map(daily_prices)
                
                # Only keep values after target time
                before_target = (df.index.hour < target_hour) | \
                              ((df.index.hour == target_hour) & (df.index.minute < target_minute))
                df.loc[before_target, out_col] = np.nan
                
                # Cleanup temp column
                df.drop('_temp_date', axis=1, inplace=True)
            else:
                df[out_col] = np.nan
                
            logging.info(f"    PriceAtTime {target_time}: Set {df[out_col].notna().sum()} values")
            
        elif name == 'ESDeclineFromTime':
            """ES decline from specific time price - VECTORIZED"""
            base_price_col = params.get('base_price_col')
            threshold = params.get('threshold', -1.0)
            
            if base_price_col not in df.columns:
                df[out_col] = False
                return df
            
            # Vectorized decline calculation
            valid_mask = df[base_price_col].notna()
            decline_pct = np.full(len(df), np.nan)
            decline_pct[valid_mask] = (df.loc[valid_mask, 'ES_close'] / df.loc[valid_mask, base_price_col] - 1) * 100
            
            df[out_col] = decline_pct <= threshold
            df[out_col] = df[out_col].fillna(False)
            
            logging.info(f"    ESDeclineFromTime threshold {threshold}%: {df[out_col].sum()} times met")
            
        elif name == 'VXDeclineWindow':
            """Check if VX declined from 8:30 to 8:45 on CPI days - VECTORIZED"""
            if 'VX_close' not in df.columns:
                df[out_col] = False
                return df
            
            df[out_col] = False
            
            # Get all CPI days
            cpi_days = df[df.get('is_cpi_day', False)].index.date
            unique_cpi_days = pd.unique(cpi_days)
            
            for date in unique_cpi_days:
                # Get VX at 8:30 and 8:45
                vx_830 = df[(df.index.date == date) & (df.index.hour == 8) & (df.index.minute == 30)]['VX_close']
                vx_845 = df[(df.index.date == date) & (df.index.hour == 8) & (df.index.minute == 45)]['VX_close']
                
                if len(vx_830) > 0 and len(vx_845) > 0:
                    if vx_845.iloc[0] <= vx_830.iloc[0]:
                        # Set True for CPI window on this day
                        mask = (df.index.date == date) & df.get('is_cpi_window', False)
                        df.loc[mask, out_col] = True
            
            logging.info(f"    VXDeclineWindow: {df[out_col].sum()} times True")
            
        elif name == 'VIXRollingDecline':
            """VIX decline in rolling window - VECTORIZED"""
            window = params.get('window', 10)
            
            if 'VIX_close' not in df.columns:
                df[out_col] = 0
                return df
            
            # Vectorized rolling max and decline calculation
            vix_rolling_max = df['VIX_close'].rolling(window=window, min_periods=1).max()
            df[out_col] = (df['VIX_close'] / vix_rolling_max - 1) * 100
            
            logging.info(f"    VIXRollingDecline: {(df[out_col] < 0).sum()} periods show decline")
            
        elif name == 'VIXDeclineFromTime':
            """VIX change from specific time - VECTORIZED"""
            base_time = params.get('time', '15:00:00')
            base_hour = int(base_time.split(':')[0])
            base_minute = int(base_time.split(':')[1])
            
            if 'VIX_close' not in df.columns:
                df[out_col] = 0
                return df
            
            # Get VIX at base time for each day
            time_mask = (df.index.hour == base_hour) & (df.index.minute == base_minute)
            base_vix = df[time_mask]['VIX_close'].groupby(df[time_mask].index.date).first()
            
            # Reindex to full dataframe
            daily_base_vix = base_vix.reindex(df.index.date).values
            
            # Calculate change, but only for times after base time
            df[out_col] = 0
            after_time = (df.index.hour > base_hour) | \
                        ((df.index.hour == base_hour) & (df.index.minute >= base_minute))
            valid = after_time & ~pd.isna(daily_base_vix)
            df.loc[valid, out_col] = (df.loc[valid, 'VIX_close'] / daily_base_vix[valid] - 1) * 100
            
            logging.info(f"    VIXDeclineFromTime: Calculated for {(df[out_col] != 0).sum()} periods")
            
        elif name == 'ESNotDownFromTime':
            """Check if ES is not down from specific time - VECTORIZED"""
            base_price_col = params.get('base_price_col')
            
            if base_price_col not in df.columns or 'ES_close' not in df.columns:
                df[out_col] = False
                return df
            
            valid_mask = df[base_price_col].notna()
            df[out_col] = False
            df.loc[valid_mask, out_col] = df.loc[valid_mask, 'ES_close'] >= df.loc[valid_mask, base_price_col]
            
            logging.info(f"    ESNotDownFromTime: {df[out_col].sum()} times True")
            
        elif name == 'FedDayQualified':
            """Check if Fed day met all conditions for next day entry"""
            if 'ES_close' not in df.columns or 'VIX_close' not in df.columns:
                df[out_col] = False
                return df
            
            df[out_col] = False
            
            # Get unique FOMC days
            fomc_days = df[df.get('is_fomc_day', False)].index.date
            unique_fomc_days = pd.unique(fomc_days)
            
            logging.info(f"    Found {len(unique_fomc_days)} FOMC days")
            
            for date in unique_fomc_days:
                day_data = df[df.index.date == date]
                
                if len(day_data) == 0:
                    continue
                
                # Check breadth
                breadth_check = False
                if 'TRIN_close' in df.columns:
                    avg_trin = day_data['TRIN_close'].mean()
                    if not pd.isna(avg_trin):
                        breadth_proxy = 100 * (2 - np.clip(avg_trin, 0.5, 2.0)) / 1.5
                        breadth_check = breadth_proxy > 60
                else:
                    breadth_check = True  # Don't block if no data
                
                # Check ES gain
                es_gain = (day_data['ES_close'].iloc[-1] / day_data['ES_open'].iloc[0] - 1) * 100
                es_check = es_gain > 0.5
                
                # Check VIX decline vs previous day
                vix_check = False
                prev_day_data = df[df.index.date < date]
                if len(prev_day_data) > 0:
                    prev_vix = prev_day_data['VIX_close'].iloc[-1]
                    curr_vix = day_data['VIX_close'].iloc[-1]
                    vix_check = curr_vix < prev_vix
                
                # Mark next trading day if all conditions met
                if breadth_check and es_check and vix_check:
                    next_days = df[df.index.date > date].index.date
                    if len(next_days) > 0:
                        next_date = pd.unique(next_days)[0]
                        df.loc[df.index.date == next_date, out_col] = True
                        logging.info(f"      {date}: Qualified, marking {next_date}")
            
            logging.info(f"    FedDayQualified: {df[out_col].sum()} rows marked")
            
        elif name == 'NextDaySignal':
            """Delay signal to next trading day"""
            signal_col = params.get('signal_col')
            
            if signal_col not in df.columns:
                df[out_col] = False
                return df
            
            # Get days where signal was true
            daily_signals = df[df[signal_col]].groupby(df[df[signal_col]].index.date).first()
            
            # Shift to next trading day
            df[out_col] = False
            for date in daily_signals.index:
                # Find next trading day
                next_day_mask = df.index.date > date
                if next_day_mask.any():
                    next_date = df[next_day_mask].index.date[0]
                    next_day_full_mask = df.index.date == next_date
                    df.loc[next_day_full_mask, out_col] = True
        
        elif name == 'ADX':
            window = params.get('window', 14)
            # Need high, low, open for ADX calculation
            instrument = on_col.split('_')[0]
            daily_high = df[f"{instrument}_high"].resample('D').max().shift(1).dropna()
            daily_low = df[f"{instrument}_low"].resample('D').min().shift(1).dropna()
            daily_close = df[f"{instrument}_close"].resample('D').last().shift(1).dropna()
            daily_adx = ta.trend.adx(daily_high, daily_low, daily_close, window=window)
            df[out_col] = daily_adx.reindex(df.index, method='ffill').ffill()

        elif name == 'EMASlope':
            span = params.get('span')
            lookback = params.get('lookback', 1)
            daily_open = df[on_col].resample('D').first().dropna()
            daily_ema = daily_open.ewm(span=span, adjust=False).mean()
            daily_slope = ((daily_ema - daily_ema.shift(lookback)) / daily_ema.shift(lookback))
            df[out_col] = daily_slope.reindex(df.index, method='ffill').ffill()

        elif name == 'VIXChange':
            lookback = params.get('lookback', 20)
            daily_vix = df[on_col].resample('D').first().dropna()
            vix_sma = daily_vix.rolling(window=lookback).mean()
            vix_declining = (daily_vix < vix_sma)
            df[out_col] = vix_declining.reindex(df.index, method='ffill').ffill()

        elif name == 'SMASlope':
            window = params.get('window', 50)
            lookback = params.get('lookback', 1)
            daily_open = df[on_col].resample('D').first().dropna()
            daily_sma = daily_open.rolling(window=window, min_periods=window).mean()
            sma_slope = (daily_sma - daily_sma.shift(lookback) > 0)
            df[out_col] = sma_slope.reindex(df.index, method='ffill').ffill()

        elif name == 'RealizedVolatility':
            window = params.get('window', 10)
            daily_open = df[on_col].resample('D').first().dropna()
            daily_returns = daily_open.pct_change()
            realized_vol = daily_returns.rolling(window=window).std() * np.sqrt(252) * 100
            df[out_col] = realized_vol.reindex(df.index, method='ffill').ffill()

        elif name == 'ConsecutiveGreenDays':
            instrument = params.get('instrument', 'SPX')
            close_col = f"{instrument}_close"
            open_col = f"{instrument}_open"
            
            # Calculate daily green/red on trading days only (drop non-trading days)
            daily_open = df[open_col].resample('D').first()
            daily_close = df[close_col].resample('D').last()
            daily = pd.DataFrame({'open': daily_open, 'close': daily_close}).dropna()
            is_green = daily['close'] > daily['open']
            
            # Count consecutive green days (trading days only)
            consecutive = is_green.astype(int).groupby((~is_green).cumsum()).cumsum()
            
            # FIXED: Lag by 1 day to avoid look-ahead bias - use previous day's consecutive count
            # This ensures we only count completed consecutive days, not including current day
            consecutive_lagged = consecutive.shift(1).fillna(0)
            df[out_col] = consecutive_lagged.reindex(df.index, method='ffill').ffill()
            
            # Original implementation (kept for reference):
            # consecutive = is_green.astype(int).groupby((~is_green).cumsum()).cumsum()
            # df[out_col] = consecutive.reindex(df.index, method='ffill').ffill()

        elif name == 'VolumeRatio':
            window = params.get('window', 20)
            daily_volume = df[on_col].resample('D').sum().shift(1)
            avg_volume = daily_volume.rolling(window=window).mean()
            volume_ratio = (daily_volume / avg_volume)
            df[out_col] = volume_ratio.reindex(df.index, method='ffill').ffill()

        elif name == 'OBV':
            # On Balance Volume
            instrument = on_col.split('_')[0]
            open_col = f"{instrument}_close"
            
            daily_open = df[open_col].resample('D').last().shift(1).dropna()
            daily_volume = df[on_col].resample('D').sum().shift(1).dropna()
            
            # Calculate OBV using opens instead of closes
            open_diff = daily_open.diff()
            obv = daily_volume.copy()
            obv[open_diff < 0] *= -1
            obv[open_diff == 0] = 0
            obv = obv.cumsum()
            
            df[out_col] = obv.reindex(df.index, method='ffill').ffill()

        elif name == 'VolumeSMA':
            # Volume Simple Moving Average
            window = params.get('window', 20)
            daily_volume = df[on_col].resample('D').sum().shift(1).dropna()
            daily_volume_sma = daily_volume.rolling(window=window, min_periods=window).mean()
            df[out_col] = daily_volume_sma.reindex(df.index, method='ffill').ffill()
            logging.info(f"    VolumeSMA calculated with window {window}")

        elif name == 'VolumeEMA':
            # Volume Exponential Moving Average
            span = params.get('span', params.get('window', 20))
            daily_volume = df[on_col].resample('D').sum().shift(1).dropna()
            daily_volume_ema = daily_volume.ewm(span=span, adjust=False).mean()
            df[out_col] = daily_volume_ema.reindex(df.index, method='ffill').ffill()
            logging.info(f"    VolumeEMA calculated with span {span}")

        elif name == 'VWAP':
            # Volume Weighted Average Price
            instrument = on_col.split('_')[0] if '_' in on_col else on_col
            close_col = f"{instrument}_close"
            high_col = f"{instrument}_high"
            low_col = f"{instrument}_low" 
            volume_col = f"{instrument}_volume"
            
            # Check if required columns exist
            required_cols = [close_col, high_col, low_col, volume_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.error(f"VWAP missing required columns: {missing_cols}")
                df[out_col] = np.nan
                return df
            
            # Calculate typical price
            typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
            
            # Calculate VWAP for each day separately
            df['date'] = df.index.date
            
            def calc_daily_vwap(group):
                cum_volume = group[volume_col].cumsum()
                cum_price_volume = (typical_price.loc[group.index] * group[volume_col]).cumsum()
                # Avoid division by zero
                vwap = np.where(cum_volume != 0, cum_price_volume / cum_volume, typical_price.loc[group.index])
                return pd.Series(vwap, index=group.index)
            
            df[out_col] = df.groupby('date').apply(calc_daily_vwap).values
            
            # Cleanup temp column
            df.drop('date', axis=1, inplace=True)
            
            logging.info(f"    VWAP calculated for instrument {instrument}")

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

        # MACRO INDICATORS
        elif name == 'PostEventTime':
            """Track time since event announcement"""
            event_time = params.get('event_time', '08:30:00')
            event_hour = int(event_time.split(':')[0])
            event_minute = int(event_time.split(':')[1])
            
            # Minutes since event time
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df[out_col] = (df['hour'] - event_hour) * 60 + (df['minute'] - event_minute)
            
            # Only positive values (after event)
            df[out_col] = df[out_col].clip(lower=0)
            
            # Cleanup temp columns
            df.drop(['hour', 'minute'], axis=1, inplace=True)
            
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
    
    # Check which columns are missing or all NaN
    missing_columns = []
    nan_columns = []
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
        elif df[col].isna().all():
            nan_columns.append(col)
    
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
    if nan_columns:
        logging.error(f"All-NaN columns: {nan_columns}")
    
    # Filter required columns to only those that exist and have data
    valid_required_columns = [col for col in required_columns if col in df.columns and not df[col].isna().all()]
    
    if not valid_required_columns:
        logging.error("No valid columns for signal generation")
        return df
    
    # Drop NaNs after all calculations are done
    df = df.ffill().dropna(subset=valid_required_columns)
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
        
        logging.info(f"Evaluating rule set for {param_name}={value}:")
        logging.info(f"  Base rules: {len(base_rules)}")
        logging.info(f"  Conditional rules: {len(conditional_rules_list)}")
        
        all_conditions = []
        
        # ALWAYS evaluate base rules first
        for i, rule in enumerate(base_rules):
            try:
                if isinstance(rule, str):
                    condition = df.eval(rule, local_dict=local_vars)
                    logging.info(f"  Base Rule {i+1:02d} '{rule}': Met {condition.sum():,} times.")
                    all_conditions.append(condition)
            except Exception as e:
                logging.error(f"  Base Rule {i+1:02d} '{rule}': FAILED TO EVALUATE. Error: {e}")
                all_conditions.append(pd.Series(False, index=df.index))
        
        # Then evaluate conditional rules
        if conditional_rules_list:
            for j, cond_rule in enumerate(conditional_rules_list):
                condition_str = cond_rule['condition'].replace(f"@{param_name}", str(value))
                if eval(condition_str):
                    rule = cond_rule['rule']
                    try:
                        if isinstance(rule, str):
                            condition = df.eval(rule, local_dict=local_vars)
                            logging.info(f"  Conditional Rule {j+1:02d} '{rule}': Met {condition.sum():,} times.")
                            all_conditions.append(condition)
                        elif isinstance(rule, dict) and rule.get('type') == 'sum_of_conditions':
                            sub_conditions = []
                            for sub_rule in rule['conditions']:
                                sub_cond = df.eval(sub_rule, local_dict=local_vars)
                                sub_conditions.append(sub_cond)
                            
                            summed_conds = pd.concat(sub_conditions, axis=1).sum(axis=1)
                            
                            threshold_str = str(rule['threshold']).replace(f"@{param_name}", str(value))
                            threshold = int(threshold_str)
                            
                            condition = (summed_conds >= threshold)
                            logging.info(f"  Conditional Rule {j+1:02d} (Cluster): {len(rule['conditions'])} sub-rules, threshold >= {threshold}: Met {condition.sum():,} times.")
                            all_conditions.append(condition)
                    except Exception as e:
                        logging.error(f"  Conditional Rule {j+1:02d}: FAILED TO EVALUATE. Error: {e}")
                        all_conditions.append(pd.Series(False, index=df.index))
        
        if not all_conditions:
            logging.warning(f"No conditions to evaluate for {param_name}={value}")
            df[f"entry_signal_{value}"] = 0
            continue
            
        final_entry_condition = pd.concat(all_conditions, axis=1).all(axis=1)
        
        signal_col_name = f"entry_signal_{value}"
        df[signal_col_name] = np.where(final_entry_condition, signal_value, 0)
        
        # One-signal-per-day logic - FIXED VERSION
        if position_type == 'short':
            # For shorts, group by date and keep only first -1 signal
            signal_df = df[df[signal_col_name] == -1]
            if len(signal_df) > 0:
                daily_first = signal_df.groupby(signal_df.index.date).head(1)
                df[signal_col_name] = 0  # Reset all to 0
                df.loc[daily_first.index, signal_col_name] = -1  # Set only first signal of each day
        else:
            # For longs, group by date and keep only first 1 signal
            signal_df = df[df[signal_col_name] == 1]
            if len(signal_df) > 0:
                daily_first = signal_df.groupby(signal_df.index.date).head(1)
                df[signal_col_name] = 0  # Reset all to 0
                df.loc[daily_first.index, signal_col_name] = 1  # Set only first signal of each day
        
        signal_count = abs(df[signal_col_name]).sum()
        unique_days = len(np.unique(df[df[signal_col_name] != 0].index.date)) if signal_count > 0 else 0
        logging.info(f"Generated {signal_count} final signals for {signal_col_name} on {unique_days} unique days")
        logging.info(f"Position type: {position_type} (signal value: {signal_value})")
        
    # Special handling for LiquidityDrift strategy - delay signals by 1 day
    if strategy_config['strategy_name'] == 'LiquidityDrift':
        for value in param_values:
            signal_col_name = f"entry_signal_{value}"
            if signal_col_name in df.columns:
                daily_signals = df[signal_col_name].resample('D').max()
                df[signal_col_name] = daily_signals.reindex(df.index, method='ffill').fillna(0).astype(int)
                
                # Re-apply one signal per day logic
                df[signal_col_name] = df[signal_col_name].groupby(df.index.date).transform(
                    lambda x: (x.cumsum() == 1) & (x == 1)
                ).astype(int)
                
                logging.info(f"Signals for {signal_col_name} - now {df[signal_col_name].sum()} signals")

    return df
