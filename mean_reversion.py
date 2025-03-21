import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import requests
import ccxt
import warnings
warnings.filterwarnings('ignore')

# Define the strategy class - rewritten without depending on backtesting library
class MeanReversionStrategy:
    def __init__(self, data, n_sma=20, n_std=1.5, initial_capital=1000, commission=0.001):
        self.data = data.copy()
        self.n_sma = n_sma
        self.n_std = n_std
        self.initial_capital = initial_capital
        self.commission = commission
        self.prepare_indicators()
        self.trades = []
        self.positions = []
        self.equity_curve = []
        
    def prepare_indicators(self):
        # Calculate SMA
        self.data['sma'] = self.data['Close'].rolling(window=self.n_sma).mean()
        
        # Calculate standard deviation
        self.data['std'] = self.data['Close'].rolling(window=self.n_sma).std()
        
        # Calculate upper and lower bands
        self.data['upper'] = self.data['sma'] + (self.n_std * self.data['std'])
        self.data['lower'] = self.data['sma'] - (self.n_std * self.data['std'])
        
        # Drop NaN values from the warmup period
        self.data = self.data.dropna()
    
    def run(self):
        # Initialize variables
        current_position = None  # None, 'long', 'short'
        entry_price = 0
        position_size = 0
        cash = self.initial_capital
        equity = self.initial_capital
        
        # Iterate through each data point
        for i in range(1, len(self.data)):
            date = self.data.index[i]
            current_price = self.data['Close'].iloc[i]
            prev_price = self.data['Close'].iloc[i-1]
            sma = self.data['sma'].iloc[i]
            upper = self.data['upper'].iloc[i]
            lower = self.data['lower'].iloc[i]
            
            # Update equity if we have a position
            if current_position == 'long':
                equity = cash + (position_size * current_price)
            elif current_position == 'short':
                equity = cash + (position_size * (2 * entry_price - current_price))
            else:
                equity = cash
            
            self.equity_curve.append((date, equity))
            
            # Check for entry/exit signals
            
            # Entry: If we don't have a position and price crosses below lower band
            if current_position is None and prev_price >= lower and current_price < lower:
                position_size = 0.2 * equity / current_price  # Use 20% of equity
                entry_price = current_price
                current_position = 'long'
                # Account for commission
                cash -= position_size * current_price * (1 + self.commission)
                self.trades.append({
                    'date': date,
                    'type': 'buy',
                    'price': current_price,
                    'size': position_size,
                    'value': position_size * current_price
                })
                print(f"BUY at {date}: Price={current_price:.2f}")
                
            # Entry: If we don't have a position and price crosses above upper band
            elif current_position is None and prev_price <= upper and current_price > upper:
                position_size = 0.2 * equity / current_price  # Use 20% of equity
                entry_price = current_price
                current_position = 'short'
                # No immediate cash impact for short, but record the entry
                self.trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': current_price,
                    'size': position_size,
                    'value': position_size * current_price
                })
                print(f"SELL at {date}: Price={current_price:.2f}")
                
            # Exit long position: If we're long and price crosses above SMA
            elif current_position == 'long' and prev_price <= sma and current_price > sma:
                # Calculate profit/loss
                exit_value = position_size * current_price * (1 - self.commission)
                entry_value = position_size * entry_price
                pnl = exit_value - entry_value
                
                # Update cash
                cash += exit_value
                
                # Record the trade
                self.trades.append({
                    'date': date,
                    'type': 'close_long',
                    'price': current_price,
                    'size': position_size,
                    'value': position_size * current_price,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_value) * 100
                })
                
                print(f"CLOSE LONG at {date}: Price={current_price:.2f}, PnL={pnl:.2f}")
                
                # Reset position
                current_position = None
                position_size = 0
                
            # Exit short position: If we're short and price crosses below SMA
            elif current_position == 'short' and prev_price >= sma and current_price < sma:
                # Calculate profit/loss for short
                exit_value = position_size * current_price
                entry_value = position_size * entry_price
                pnl = entry_value - exit_value - (self.commission * (entry_value + exit_value))
                
                # Update cash
                cash += entry_value + pnl
                
                # Record the trade
                self.trades.append({
                    'date': date,
                    'type': 'close_short',
                    'price': current_price,
                    'size': position_size,
                    'value': position_size * current_price,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_value) * 100
                })
                
                print(f"CLOSE SHORT at {date}: Price={current_price:.2f}, PnL={pnl:.2f}")
                
                # Reset position
                current_position = None
                position_size = 0
        
        # Close any open position at the end
        if current_position is not None:
            last_date = self.data.index[-1]
            last_price = self.data['Close'].iloc[-1]
            
            if current_position == 'long':
                exit_value = position_size * last_price * (1 - self.commission)
                entry_value = position_size * entry_price
                pnl = exit_value - entry_value
                cash += exit_value
            else:  # short
                exit_value = position_size * last_price
                entry_value = position_size * entry_price
                pnl = entry_value - exit_value - (self.commission * (entry_value + exit_value))
                cash += entry_value + pnl
                
            self.trades.append({
                'date': last_date,
                'type': f'close_{current_position}',
                'price': last_price,
                'size': position_size,
                'value': position_size * last_price,
                'pnl': pnl,
                'pnl_pct': (pnl / entry_value) * 100
            })
            
            print(f"CLOSE {current_position.upper()} at END: Price={last_price:.2f}, PnL={pnl:.2f}")
        
        # Calculate final equity
        final_equity = cash
        
        # Calculate strategy statistics
        return self.calculate_stats(final_equity)
    
    def calculate_stats(self, final_equity):
        # Extract dates and equity values
        dates = [item[0] for item in self.equity_curve]
        equity_values = [item[1] for item in self.equity_curve]
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({'equity': equity_values}, index=dates)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        # Filter completed trades (with PnL)
        completed_trades = [t for t in self.trades if 'pnl' in t]
        
        # Calculate trade statistics
        if completed_trades:
            trade_returns = [t['pnl_pct'] for t in completed_trades]
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            
            # Calculate Sharpe ratio (annualized)
            if len(equity_df['returns'].dropna()) > 0:
                sharpe = np.sqrt(252) * (equity_df['returns'].mean() / equity_df['returns'].std()) if equity_df['returns'].std() != 0 else 0
            else:
                sharpe = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            avg_trade_return = 0
            sharpe = 0
        
        # Compile stats
        stats = {
            'Initial Capital': self.initial_capital,
            'Final Equity': final_equity,
            'Return [%]': ((final_equity / self.initial_capital) - 1) * 100,
            'Max. Drawdown [%]': equity_df['drawdown'].min() if not equity_df['drawdown'].empty else 0,
            '# Trades': len(completed_trades),
            'Win Rate [%]': win_rate,
            'Avg. Trade [%]': avg_trade_return,
            'Avg. Win': avg_win,
            'Avg. Loss': avg_loss,
            'Sharpe Ratio': sharpe,
            '_trades': completed_trades,
            '_equity_curve': equity_df
        }
        
        return stats
    
    def plot(self, filename=None):
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot price and indicators on the first subplot
        ax1.plot(self.data.index, self.data['Close'], label='Price', alpha=0.7)
        ax1.plot(self.data.index, self.data['sma'], label=f'SMA({self.n_sma})', alpha=0.7)
        ax1.plot(self.data.index, self.data['upper'], label='Upper Band', linestyle='--', alpha=0.5)
        ax1.plot(self.data.index, self.data['lower'], label='Lower Band', linestyle='--', alpha=0.5)
        
        # Plot buy/sell signals
        for trade in self.trades:
            if trade['type'] == 'buy':
                ax1.scatter(trade['date'], trade['price'], marker='^', color='green', s=100, label='_nolegend_')
            elif trade['type'] == 'sell':
                ax1.scatter(trade['date'], trade['price'], marker='v', color='red', s=100, label='_nolegend_')
            elif trade['type'] == 'close_long':
                ax1.scatter(trade['date'], trade['price'], marker='o', color='blue', s=60, label='_nolegend_')
            elif trade['type'] == 'close_short':
                ax1.scatter(trade['date'], trade['price'], marker='o', color='purple', s=60, label='_nolegend_')
        
        ax1.set_title('ETH/USDT Mean Reversion Strategy')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot equity curve on the second subplot
        dates = [item[0] for item in self.equity_curve]
        equity_values = [item[1] for item in self.equity_curve]
        
        if dates and equity_values:
            ax2.plot(dates, equity_values, label='Equity', color='blue')
            ax2.set_title('Equity Curve')
            ax2.set_ylabel('Equity (USDT)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"Plot saved to {filename}")
        else:
            plt.show()
            
        return fig

# Function to download ETH/USDT historical data
def download_eth_data(start_date, end_date, timeframe='1h'):
    filename = f"ETH_USDT_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    # Check if file already exists
    if os.path.exists(filename):
        print(f"Loading data from existing file: {filename}")
        return pd.read_csv(filename, index_col=0, parse_dates=True)
    
    print(f"Downloading ETH/USDT {timeframe} data from {start_date} to {end_date}...")
    
    try:
        # Initialize exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        # Convert dates to milliseconds timestamp
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        # Download data
        all_candles = []
        current = since
        
        while current < until:
            print(f"Fetching data from {datetime.fromtimestamp(current/1000)}")
            candles = exchange.fetch_ohlcv('ETH/USDT', timeframe, current)
            if not candles:
                break
            
            all_candles.extend(candles)
            current = candles[-1][0] + 1  # Start from the next timestamp
            
        # Convert to dataframe
        df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save to CSV
        df.to_csv(filename)
        print(f"Data saved to {filename}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        
        # Fallback to alternative data source if ccxt fails
        try:
            print("Trying alternative data source...")
            # You can modify this URL to get different timeframes or date ranges
            url = f"https://api.cryptowat.ch/markets/binance/ethusdt/ohlc?periods=3600"  # 3600 seconds = 1 hour
            response = requests.get(url)
            data = response.json()
            
            if 'result' in data and '3600' in data['result']:
                candles = data['result']['3600']
                df = pd.DataFrame(candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                # Save to CSV
                df.to_csv(filename)
                print(f"Data saved to {filename}")
                
                return df
            else:
                raise Exception("No data returned from alternative source")
                
        except Exception as alt_e:
            print(f"Error with alternative source: {alt_e}")
            
            # Last resort: generate sample data
            print("Using sample data as fallback...")
            return generate_sample_data(start_date, end_date)

# Generate sample data as a fallback
def generate_sample_data(start_date, end_date):
    print("Generating sample ETH/USDT data...")
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Generate prices with pronounced mean reversion
    np.random.seed(42)
    
    # Parameters
    initial_price = 3000
    volatility = 15
    mean_reversion_strength = 0.05
    cycle_length = 200  # hours
    
    # Generate price series
    prices = [initial_price]
    
    for i in range(1, len(dates)):
        # Random component
        noise = np.random.normal(0, volatility)
        
        # Mean reversion component
        if i >= 20:
            sma_20 = sum(prices[-20:]) / 20
            reversion = (sma_20 - prices[-1]) * mean_reversion_strength
        else:
            reversion = 0
            
        # Cyclical component (to create more trading opportunities)
        cycle = 20 * np.sin(i * 2 * np.pi / cycle_length)
        
        # Combine components
        change = noise + reversion + cycle
        
        # Calculate new price
        new_price = max(prices[-1] + change, 500)  # Ensure price stays above 500
        prices.append(new_price)
    
    # Create DataFrame
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = [prices[max(0, i-1)] for i in range(len(prices))]
    df['High'] = [max(df['Open'].iloc[i], df['Close'].iloc[i]) * (1 + np.random.uniform(0, 0.005)) 
                 for i in range(len(df))]
    df['Low'] = [min(df['Open'].iloc[i], df['Close'].iloc[i]) * (1 - np.random.uniform(0, 0.005)) 
                for i in range(len(df))]
    df['Volume'] = [np.random.uniform(1000, 5000) for _ in range(len(df))]
    
    print(f"Generated {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    return df

# Add a function for parameter optimization
def optimize_strategy(data):
    print("Optimizing strategy parameters...")
    
    # Define parameter ranges to test
    sma_values = [10, 15, 20, 25, 30]
    std_values = [1.0, 1.5, 2.0, 2.5]
    
    # Track the best parameters and performance
    best_return = -np.inf
    best_params = None
    best_stats = None
    
    # Try all combinations
    total_combinations = len(sma_values) * len(std_values)
    current_combination = 0
    
    for n_sma in sma_values:
        for n_std in std_values:
            current_combination += 1
            print(f"Testing combination {current_combination}/{total_combinations}: SMA={n_sma}, STD={n_std}")
            
            # Run strategy with these parameters
            strategy = MeanReversionStrategy(data, n_sma=n_sma, n_std=n_std)
            stats = strategy.run()
            
            # Update best parameters if this combination has a better return
            if stats['Return [%]'] > best_return:
                best_return = stats['Return [%]']
                best_params = {'n_sma': n_sma, 'n_std': n_std}
                best_stats = stats
    
    print("\nOptimization Results:")
    print(f"Best Parameters: SMA={best_params['n_sma']}, STD Multiplier={best_params['n_std']}")
    print(f"Return: {best_stats['Return [%]']:.2f}%")
    
    # Run the strategy with the best parameters and plot
    best_strategy = MeanReversionStrategy(data, n_sma=best_params['n_sma'], n_std=best_params['n_std'])
    best_strategy.run()
    best_strategy.plot(filename='optimized_mean_reversion_results.png')
    
    return best_stats, best_params

# Main function
def main():
    try:
        # Date range for the backtest
        end_date = datetime(2025, 3, 20)
        start_date = end_date - timedelta(days=90)  # 3 months of data
        
        # Download or load real ETH/USDT data
        data = download_eth_data(start_date, end_date)
        
        # Print sample of the data
        print("\nSample data:")
        print(data.head())
        
        # Ensure data is properly formatted
        # Convert column names to title case if needed
        if 'close' in data.columns and 'Close' not in data.columns:
            data.columns = [col.title() for col in data.columns]
        
        # Run the backtest
        strategy = MeanReversionStrategy(data)
        stats = strategy.run()
        
        # Plot the results
        strategy.plot(filename='mean_reversion_results.png')
        
        # Print the results
        print("\nBacktest Results:")
        print(f"Initial Capital: ${stats['Initial Capital']:.2f}")
        print(f"Final Equity: ${stats['Final Equity']:.2f}")
        print(f"Total Return: {stats['Return [%]']:.2f}%")
        
        if 'Sharpe Ratio' in stats and not np.isnan(stats['Sharpe Ratio']):
            print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        
        if 'Max. Drawdown [%]' in stats:
            print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        
        if '# Trades' in stats:
            print(f"Number of Trades: {stats['# Trades']}")
        
        if 'Win Rate [%]' in stats and not np.isnan(stats['Win Rate [%]']):
            print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        
        if 'Avg. Trade [%]' in stats and not np.isnan(stats['Avg. Trade [%]']):
            print(f"Average Trade: {stats['Avg. Trade [%]']:.2f}%")
            
        # Print detailed trade information
        if '# Trades' in stats and stats['# Trades'] > 0:
            print("\nTrade Details:")
            trades = stats.get('_trades')
            if trades is not None and len(trades) > 0:
                for i, trade in enumerate(trades[:min(5, len(trades))]):
                    print(f"Trade {i+1}: Size={trade['size']:.4f}, "
                          f"Entry={trade['price']:.2f}, "
                          f"PnL={trade.get('pnl', 'N/A')}, "
                          f"Return={trade.get('pnl_pct', 'N/A')}%")
                if len(trades) > 5:
                    print(f"... and {len(trades) - 5} more trades")
            
            # Ask if user would like to run optimization
            run_opt = input("\nWould you like to run parameter optimization? (y/n): ")
            if run_opt.lower() == 'y':
                optimize_strategy(data)
        else:
            print("\nNo trades were executed. Consider running parameter optimization.")
            run_opt = input("Would you like to run parameter optimization? (y/n): ")
            if run_opt.lower() == 'y':
                optimize_strategy(data)
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()