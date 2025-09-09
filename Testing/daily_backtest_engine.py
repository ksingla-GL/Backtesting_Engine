"""
Daily Backtest Engine for Testing Environment
Runs backtests on daily data with detailed logging and Excel output
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

class DailyBacktestEngine:
    def __init__(self):
        self.logger = self._setup_logger()
        self.trades = []
        self.daily_positions = []
        self.performance_metrics = {}
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def run_backtest(self, data: pd.DataFrame, strategy_config: Dict, initial_capital: float = 100000) -> Dict:
        """Run backtest on daily data"""
        self.logger.info("Starting daily backtest...")
        
        # Initialize backtest variables
        self.trades = []
        self.daily_positions = []
        current_position = None
        portfolio_value = initial_capital
        
        # Get strategy parameters
        position_type = strategy_config.get('position_type', 'long')
        exit_params = strategy_config.get('exit_parameters', {})
        
        stop_loss_pct = exit_params.get('stop_loss_pct', 2.0) / 100.0
        take_profit_pct = exit_params.get('take_profit_pct', 4.0) / 100.0
        trailing_stop_pct = exit_params.get('trailing_stop_loss_pct', 1.5) / 100.0
        
        self.logger.info(f"Position type: {position_type}")
        self.logger.info(f"Stop loss: {stop_loss_pct*100:.1f}%")
        self.logger.info(f"Take profit: {take_profit_pct*100:.1f}%")
        self.logger.info(f"Trailing stop: {trailing_stop_pct*100:.1f}%")
        
        # Process each day
        for idx, row in data.iterrows():
            date = row['date']
            
            # Get prices (use ES_close as the trading price)
            if pd.isna(row['ES_close']):
                continue
                
            current_price = row['ES_close']
            
            # Record daily position info
            daily_info = {
                'date': date,
                'price': current_price,
                'portfolio_value': portfolio_value,
                'position': 'long' if current_position else 'none',
                'position_size': current_position['size'] if current_position else 0,
                'position_value': current_position['size'] * current_price if current_position else 0,
                'unrealized_pnl': 0,
                'entry_signal': row.get('entry_signal_final', False)
            }
            
            # Check for exit conditions if in position
            if current_position:
                entry_price = current_position['entry_price']
                size = current_position['size']
                highest_price = current_position.get('highest_price', entry_price)
                
                # Update highest price for trailing stop
                if position_type == 'long' and current_price > highest_price:
                    current_position['highest_price'] = current_price
                    highest_price = current_price
                
                # Calculate unrealized PnL
                if position_type == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:
                    unrealized_pnl = (entry_price - current_price) * size
                
                daily_info['unrealized_pnl'] = unrealized_pnl
                
                # Check exit conditions
                exit_reason = None
                
                if position_type == 'long':
                    # Stop loss
                    if current_price <= entry_price * (1 - stop_loss_pct):
                        exit_reason = 'stop_loss'
                    # Take profit
                    elif current_price >= entry_price * (1 + take_profit_pct):
                        exit_reason = 'take_profit'
                    # Trailing stop
                    elif current_price <= highest_price * (1 - trailing_stop_pct):
                        exit_reason = 'trailing_stop'
                
                else:  # short position
                    # Stop loss
                    if current_price >= entry_price * (1 + stop_loss_pct):
                        exit_reason = 'stop_loss'
                    # Take profit
                    elif current_price <= entry_price * (1 - take_profit_pct):
                        exit_reason = 'take_profit'
                    # Trailing stop (for short, lowest price)
                    elif current_price >= highest_price * (1 + trailing_stop_pct):
                        exit_reason = 'trailing_stop'
                
                # Execute exit if triggered
                if exit_reason:
                    exit_price = current_price
                    pnl = (exit_price - entry_price) * size if position_type == 'long' else (entry_price - exit_price) * size
                    
                    trade_record = {
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': size,
                        'position_type': position_type,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * abs(size))) * 100,
                        'exit_reason': exit_reason,
                        'days_held': (date - current_position['entry_date']).days,
                        'max_favorable': highest_price if position_type == 'long' else current_position.get('lowest_price', entry_price),
                        'entry_indicators': current_position['entry_indicators']
                    }
                    
                    self.trades.append(trade_record)
                    portfolio_value += pnl
                    current_position = None
                    
                    self.logger.info(f"Exit trade on {date.date()}: {exit_reason}, PnL: ${pnl:.2f}")
            
            # Check for new entry signal
            elif row.get('entry_signal_final', False):
                # Calculate position size (risk-based sizing)
                risk_per_trade = portfolio_value * 0.02  # Risk 2% per trade
                stop_distance = current_price * stop_loss_pct
                position_size = risk_per_trade / stop_distance
                
                # Don't let position size exceed 20% of portfolio
                max_position_value = portfolio_value * 0.2
                max_size = max_position_value / current_price
                position_size = min(position_size, max_size)
                
                # Record entry indicators
                entry_indicators = {}
                for col in data.columns:
                    if any(indicator in col for indicator in ['EMA', 'RSI', 'SMA', 'VIX', 'TRIN', 'CNN', 'rule', 'signal']):
                        entry_indicators[col] = row[col]
                
                current_position = {
                    'entry_date': date,
                    'entry_price': current_price,
                    'size': position_size,
                    'highest_price': current_price,
                    'entry_indicators': entry_indicators
                }
                
                self.logger.info(f"Enter {position_type} position on {date.date()}: ${current_price:.2f}, size: {position_size:.2f}")
            
            self.daily_positions.append(daily_info)
        
        # Close any remaining position at end
        if current_position:
            final_price = data.iloc[-1]['ES_close']
            size = current_position['size']
            entry_price = current_position['entry_price']
            pnl = (final_price - entry_price) * size if position_type == 'long' else (entry_price - final_price) * size
            
            trade_record = {
                'entry_date': current_position['entry_date'],
                'exit_date': data.iloc[-1]['date'],
                'entry_price': entry_price,
                'exit_price': final_price,
                'size': size,
                'position_type': position_type,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * abs(size))) * 100,
                'exit_reason': 'end_of_data',
                'days_held': (data.iloc[-1]['date'] - current_position['entry_date']).days,
                'max_favorable': current_position.get('highest_price', entry_price),
                'entry_indicators': current_position['entry_indicators']
            }
            
            self.trades.append(trade_record)
            portfolio_value += pnl
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(initial_capital, portfolio_value)
        
        self.logger.info(f"Backtest completed: {len(self.trades)} trades")
        self.logger.info(f"Final portfolio value: ${portfolio_value:.2f}")
        
        return {
            'trades': self.trades,
            'daily_positions': self.daily_positions,
            'performance_metrics': self.performance_metrics,
            'initial_capital': initial_capital,
            'final_value': portfolio_value
        }
    
    def _calculate_performance_metrics(self, initial_capital: float, final_value: float) -> Dict:
        """Calculate performance metrics from trades"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_return = (final_value - initial_capital) / initial_capital * 100
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # PnL statistics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')
        
        # Drawdown calculation
        daily_df = pd.DataFrame(self.daily_positions)
        if not daily_df.empty:
            daily_df['cumulative_pnl'] = (daily_df['portfolio_value'] - initial_capital).cumsum()
            daily_df['peak'] = daily_df['cumulative_pnl'].cummax()
            daily_df['drawdown'] = daily_df['cumulative_pnl'] - daily_df['peak']
            max_drawdown = daily_df['drawdown'].min()
            max_drawdown_pct = (max_drawdown / initial_capital) * 100 if initial_capital > 0 else 0
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Trade duration
        avg_days_held = trades_df['days_held'].mean() if total_trades > 0 else 0
        
        metrics = {
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_days_held': avg_days_held,
            'best_trade': trades_df['pnl'].max() if total_trades > 0 else 0,
            'worst_trade': trades_df['pnl'].min() if total_trades > 0 else 0,
            'total_pnl': trades_df['pnl'].sum() if total_trades > 0 else 0
        }
        
        return metrics
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
    
    def get_daily_positions_dataframe(self) -> pd.DataFrame:
        """Get daily positions as DataFrame"""
        return pd.DataFrame(self.daily_positions) if self.daily_positions else pd.DataFrame()
    
    def print_performance_summary(self):
        """Print performance summary"""
        if not self.performance_metrics:
            print("No performance metrics available")
            return
        
        print("\n" + "="*50)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*50)
        
        metrics = self.performance_metrics
        
        print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
        print(f"Winning Trades: {metrics.get('winning_trades', 0)}")
        print(f"Losing Trades: {metrics.get('losing_trades', 0)}")
        print(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
        print(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Average Days Held: {metrics.get('avg_days_held', 0):.1f}")
        print(f"Best Trade: ${metrics.get('best_trade', 0):.2f}")
        print(f"Worst Trade: ${metrics.get('worst_trade', 0):.2f}")
        print("="*50)

if __name__ == "__main__":
    # Test the backtest engine
    from daily_data_handler import DailyDataHandler
    from daily_signal_generator import DailySignalGenerator
    
    # Load data
    handler = DailyDataHandler()
    data = handler.load_data()
    
    if data is not None:
        # Test strategy config
        test_config = {
            "position_type": "long",
            "indicators": [
                {"name": "EMA", "params": {"span": 9}, "on_column": "ES_close", "output_col": "ES_EMA_9"},
                {"name": "RSI", "params": {"window": 2}, "on_column": "ES_close", "output_col": "ES_RSI_2"}
            ],
            "param_grid": {"fear_threshold": [30]},
            "base_entry_rules": ["CNN_FEAR_GREED < @fear_threshold", "ES_RSI_2 > 50"],
            "conditional_entry_rules": [{
                "condition": "True",
                "rule": {
                    "type": "sum_of_conditions",
                    "conditions": ["VIX_close > 20", "ES_RSI_2 > 60", "TRIN_close < 0.9"],
                    "threshold": "1"
                }
            }],
            "exit_parameters": {
                "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0,
                "trailing_stop_loss_pct": 1.5
            }
        }
        
        # Generate signals
        generator = DailySignalGenerator()
        data_with_indicators = generator.calculate_indicators(data, test_config)
        data_with_signals = generator.generate_signals(data_with_indicators, test_config)
        
        # Run backtest
        engine = DailyBacktestEngine()
        results = engine.run_backtest(data_with_signals, test_config)
        
        # Print results
        engine.print_performance_summary()
        
        print(f"\nGenerated {len(results['trades'])} trades")
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            print("\nSample trades:")
            print(trades_df[['entry_date', 'exit_date', 'pnl', 'exit_reason']].head())
    
    else:
        print("Failed to load test data")