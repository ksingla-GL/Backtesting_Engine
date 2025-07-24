# Custom Backtesting Engine
A high-performance, configuration-driven framework for backtesting equity index trading strategies. Built with a modular architecture and Numba optimization for processing millions of data points efficiently.

## Overview
This project provides a complete backtesting solution that:
- Processes and stores millions of historical price and market indicator records
- Dynamically generates trading signals from configurable strategies
- Executes high-speed backtests with realistic position management
- Produces comprehensive performance analytics and detailed trade logs
- Supports multiple strategies through simple JSON configuration files

## Project Structure
```
Custom_Backtesting_Engine/
├── main_backtester.py       # Main orchestrator script
├── data_handler.py          # Database connection and data loading
├── signal_generator.py      # Dynamic indicator calculation and signal generation
├── backtest_engine.py       # Numba-optimized backtesting functions
├── config.json              # Global settings (DB path, dates, costs)
├── strategies/              # Strategy configuration files
│   ├── quick_panic.json     # Quick Panic mean reversion strategy
│   ├── bottom_fishing.json  # Bottom Fishing strategy
│   └── [your_strategy].json # Add new strategies here
├── SQL_Db_Migration.py      # Initial database population script
├── backtesting_v2.db        # SQLite database (created by migration)
└── README.md
```

## Prerequisites
- Python 3.8+
- Required packages:
```bash
pip install pandas numpy numba ta sqlite3 logging
```

## Setup Instructions

### Step 1: Prepare Your Data
Create the following directory structure:
```
Backtesting_data/
├── 1 min/           # 1-minute price data files
├── 5 min/           # 5-minute price data files
├── 15 min/          # 15-minute price data files
├── 30 min/          # 30-minute price data files
├── 1 hour/          # 1-hour price data files
├── Daily/           # Daily price data files
├── NAAIM.xlsx       # NAAIM sentiment data
├── FedReserve stance.xlsx
├── CNN Fear and Greed Index.xlsx
├── Buffett Indicator.xlsx
└── Market_breadth($S5FI)_2011-2025.parquet
```

Expected file formats:
- **Price data**: .txt, .csv, .xlsx, or .parquet with Date, Time (if intraday), OHLC, and Volume
- **Indicator files**: .xlsx or .parquet with Date and Value columns

### Step 2: Configure Database Path
Edit `config.json` to match your environment:
```json
{
  "settings": {
    "db_path": "../backtesting_v2.db",
    "primary_instrument": "ES",
    "start_date": "2013-01-01",
    "end_date": "2024-12-31",
    "transaction_cost_pct": 0.0005
  }
}
```

### Step 3: Run Data Migration
Populate the SQLite database with historical data:
```bash
python SQL_Db_Migration.py
```

This will:
- Create database schema (symbol, price_data, market_indicator tables)
- Load all price data (ES, SPX, VIX, VX, TRIN) across multiple timeframes
- Import market indicators (NAAIM, Fed stance, CNN Fear & Greed, etc.)
- Display comprehensive summary statistics

Expected output:
```
MIGRATION COMPLETE
Files processed: XX
Price records loaded: 17,800,000+
Indicator records loaded: X,XXX,XXX
```

### Step 4: Run a Backtest
Execute a backtest by specifying the strategy name:
```bash
python main_backtester.py quick_panic
```

This single command will:
1. Load and merge all data at 1-minute intervals
2. Calculate technical indicators specified in the strategy config
3. Generate entry signals based on the strategy rules
4. Run the backtest with position management
5. Output comprehensive performance metrics

## Strategy Configuration

### Quick Panic ES Example
The `strategies/quick_panic.json` file defines:
```json
{
  "strategy_name": "QuickPanicMeanReversion",
  "exit_type": "simple_exits",
  "indicators": [
    { "name": "SMA", "params": { "window": 50 }, "on_column": "ES_close", "output_col": "ES_SMA_50" },
    { "name": "RSI", "params": { "window": 2 }, "on_column": "ES_close", "output_col": "ES_RSI_2" },
    // ... more indicators
  ],
  "param_grid": {
    "es_decline_pct": [-1, -2, -3, -4]
  },
  "entry_rules": [
    "ES_close > ES_SMA_50",
    "VIX_close < 25",
    "ES_decline_from_peak <= @es_decline_pct",
    // ... more rules
  ],
  "exit_conditions": {
    "stop_loss_pct": 1.0,
    "take_profit_pct": 5.0,
    "trailing_stop_loss_pct": 3.0
  }
}
```

### Creating New Strategies
1. Copy an existing strategy JSON file
2. Modify the indicators, entry rules, and exit conditions
3. Save with a descriptive name in the `strategies/` folder
4. Run: `python main_backtester.py your_strategy_name`

## Output Files

### Performance Report (Console Output)
```
FINAL BACKTEST RESULTS: QuickPanicMeanReversion
================================================================================
                          Total Trades  Win Rate (%)  Profit Factor  Total Return (%)  Max Drawdown (%)  Avg Duration (hrs)
Parameter: es_decline_pct                                                                                                    
-1                                  90         42.22           2.04             64.72             -6.32              322.35
-2                                  43         39.53           2.51             40.07             -4.13              233.38
-3                                   5         40.00           5.12              8.03             -1.05              233.00
-4                                   0          0.00            N/A              0.00              0.00                0.00
================================================================================
```

### Generated Files
- **results_[strategy_name].csv** - Summary performance metrics for all parameters
- **trade_log_[strategy_name]_[param].csv** - Detailed log of every trade (for first parameter only)

## Key Features

### Modular Architecture
- **data_handler.py** - Handles all database operations and data alignment
- **signal_generator.py** - Dynamically calculates indicators and generates signals
- **backtest_engine.py** - High-performance trade simulation with Numba
- **main_backtester.py** - Orchestrates the entire process

### Configuration-Driven
- Global settings in `config.json`
- Individual strategies in `strategies/` folder
- No code changes needed to test new ideas

### Performance Optimized
- Numba JIT compilation for 100x faster backtesting
- Efficient data structures for millions of records
- Minimal memory footprint

### Comprehensive Analytics
- Win rate, profit factor, Sharpe ratio
- Maximum drawdown analysis
- Trade duration statistics
- Exit reason breakdown

## Advanced Usage

### Parameter Optimization
Test multiple parameter values simultaneously:
```json
"param_grid": {
  "es_decline_pct": [-0.5, -1, -1.5, -2, -2.5, -3]
}
```

### Complex Entry Rules
Use conditional logic and sum-of-conditions:
```json
"conditional_entry_rules": [
  {
    "condition": "@es_decline_pct >= -1",
    "rule": {
      "type": "sum_of_conditions",
      "conditions": ["TRIN_close < 1.5", "CNN_FEAR_GREED > 35", "NAAIM > 40"],
      "threshold": "2"
    }
  }
]
```

### Available Exit Types
- `simple_exits` - Stop loss, take profit, trailing stop, MA crossover
- `simple_exits_no_ma_cross` - Same as above but without MA crossover

## Troubleshooting

**No trades generated**
- Check if entry conditions are too restrictive
- Review signal generation logs for how many times each rule is met
- Try relaxing thresholds in the strategy config

**Data quality issues**
- Check logs during data loading for NaN warnings
- Verify date ranges in `config.json` match your data
- Ensure all required indicators are present in the database

**Performance issues**
- For initial testing, use a smaller date range
- Check available system memory for large datasets
- Consider processing one strategy at a time

## Next Steps
1. **Add More Strategies** - Create new JSON configs for trend following or volatility strategies
2. **Optimize Parameters** - Use the param_grid to find optimal thresholds
3. **Enhance Exit Logic** - Implement time-based exits or volatility-adjusted stops
4. **Position Sizing** - Add Kelly criterion or risk parity sizing (Phase 3)
5. **Live Trading** - Connect to broker API for real-time execution