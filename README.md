# Custom Backtesting Engine

A comprehensive framework for backtesting equity index trading strategies. It focuses on mean reversion setups for ES (E-mini S&P 500 futures) and other major indices.

## Overview

This project allows you to:

- Process and store millions of historical price and market indicator records
- Generate trading signals from configurable technical and sentiment indicators
- Execute backtests with realistic position management and risk controls
- Produce detailed performance analytics and trade logs

The architecture is modular so we can add new strategies or tweak parameters through configuration files.

## Project Structure

```text
Custom_Backtesting_engine/
├── SQL_Schema/              # Database schema documentation
├── SQL_Db_Migration.py      # Load historical data into SQLite
├── fast_merge_1min.py       # Merge price and indicator data at 1‑minute intervals
├── Strategy_entry_signals.py # Signal generation with scoring system
├── Strategy_backtesting.py  # Backtesting engine with performance metrics
├── config.json              # Strategy parameters and thresholds
├── trade_log.csv            # Output: detailed log of all trades
└── README.md
```

## Prerequisites

- Python 3.8+
- Required packages:

```bash
pip install -r requirements.txt
```

## Setup Instructions

### Step 1: Prepare Your Data

Create the following directory structure:

```text
Backtesting_data/
├── 1 min/           # 1‑minute price data files
├── 5 min/           # 5‑minute price data files
├── 15 min/          # 15‑minute price data files
├── 30 min/          # 30‑minute price data files
├── 1 hour/          # 1‑hour price data files
├── Daily/           # Daily price data files
├── NAAIM.xlsx       # NAAIM sentiment data
├── FedReserve stance.xlsx
├── CNN Fear and Greed Index.xlsx
├── Buffett Indicator.xlsx
└── Market_breadth($S5FI)_2011-2025.parquet
```

Expected file formats:

- Price data: `.txt`, `.csv` or `.parquet` with columns for Date, Time (if intraday), OHLC and Volume
- Indicator files: `.xlsx` or `.parquet` with Date and Value columns

### Step 2: Configure Paths

Edit the paths in each script to match your environment:

```python
# In SQL_Db_Migration.py
DB_PATH = r'C:\Your\Path\backtesting.db'
DATA_ROOT = r'C:\Your\Path\Backtesting_data'

# In fast_merge_1min.py
base_path = r'C:\Your\Path\Backtesting_data'
db_path = r'C:\Your\Path\backtesting.db'
```

### Step 3: Run Data Migration

Populate the SQLite database with historical data:

```bash
python SQL_Db_Migration.py
```

This will:

- Create database schema (symbols, price_data, market_indicators tables)
- Load all price data (ES, SPX, VIX, VX) across multiple timeframes
- Import market indicators (NAAIM, Fed stance, CNN Fear & Greed, etc.)
- Display summary statistics of the loaded data

Expected output:

```
MIGRATION COMPLETE
Files processed: XX
Price records loaded: 17,800,000+
Indicator records loaded: X,XXX,XXX
```

### Step 4: Merge Data at 1‑Minute Intervals

Combine all data sources into a single analysis-ready file:

```bash
python fast_merge_1min.py
```

This creates `merged_data_1min.parquet` containing:

- ES price data and technical indicators (SMA, EMA, RSI)
- VIX and VX calculations (spikes, ratios)
- Market indicators aligned to 1‑minute timestamps
- Pre‑calculated fields like 10‑day highs and previous closes

### Step 5: Generate Entry Signals

Run the signal generation with the Quick Panic strategy:

```bash
python Strategy_entry_signals.py
```

This will:

- Apply the scoring system to identify entry opportunities
- Filter to one signal per day (highest score)
- Save enhanced data to `merged_data_with_signals.parquet`
- Display signal statistics and parameter sensitivity analysis

### Step 6: Run Backtest

Execute the backtest with position management:

```bash
python Strategy_backtesting.py
```

Output includes:

- Comprehensive performance metrics (Win rate, Profit factor, Sharpe ratio, etc.)
- Trade duration analysis
- Exit reason breakdown
- Monthly performance summary
- A detailed trade log saved to `trade_log.csv`

## Configuration

Adjust strategy parameters in `config.json`:

```json
{
  "entry_conditions": {
    "es_decline_threshold": -1.0,
    "vix_threshold": 25,
    "rsi_threshold": 10,
    "min_score_for_signal": 12
  },
  "exit_conditions": {
    "stop_loss_pct": 0.99,
    "take_profit_pct": 1.03,
    "trailing_stop_activation": 1.02,
    "trailing_stop_distance": 0.97
  }
}
```

## Key Features

1. **Modular Design** – data loading, signal generation and backtesting are separate and reusable components
2. **Config-Driven** – strategy parameters live in external config files for easy experimentation
3. **Comprehensive Analytics** – automatically calculates a full suite of performance metrics
4. **Realistic Execution** – handles stops, targets and trailing stops in correct order
5. **Extensible** – simple to add new strategies and indicators

## Example Results

Quick Panic ES Strategy (3% profit target):

- Total Trades: 90
- Win Rate: 30%
- Profit Factor: 1.23
- Total Return: 13.78%
- Max Drawdown: -11.52%

## Next Steps

1. **Parameter Optimization** – experiment with different thresholds using the config file
2. **Add Strategies** – extend the framework for trend following or macro event strategies
3. **Position Sizing** – implement dynamic sizing based on signal strength
4. **Market Regime Filters** – trade only in favorable environments

## Troubleshooting

1. **No trades generated** – check if entry conditions are too restrictive
2. **Data not loading** – verify file paths and formats match the expected structure
3. **Memory issues** – process data in chunks or use date filters for large datasets

