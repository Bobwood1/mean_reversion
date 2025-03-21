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
    def __init__(self, data, n_sma=20, n_std=1.5, initial_capital=1000, 
                 commission=0.001, slippage=0.001, max_position_size_pct=0.05,
                 stop_loss_pct=0.03, max_drawdown_pct=0.15, execution_delay=1):
        self.data = data.copy()
        self.n_sma = n_sma
        self.n_std = n_std
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% commission rate
        self.slippage = slippage  # 0.1% base slippage
        self.max_position_size_pct = max_position_size_pct  # Maximum position size as % of equity
        self.stop_loss_pct = stop_loss_pct  # Stop loss as % of entry price
        self.max_drawdown_pct = max_drawdown_pct  # Maximum allowed drawdown
        self.execution_delay = execution_delay  # Delay in bars between signal and execution
        self.prepare_indicators()
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.signals = []  # Store signals for delayed execution
        
        # Add absolute position size cap ($100K per position regardless of equity)
        self.absolute_position_cap = 100000
        
        # Add volume threshold based on average daily volume
        self.min_volume_factor = 10  # Require 10x our trade value in volume
        
    def prepare_indicators(self):
        # Calculate SMA
        self.data['sma'] = self.data['Close'].rolling(window=self.n_sma).mean()
        
        # Calculate standard deviation
        self.data['std'] = self.data['Close'].rolling(window=self.n_sma).std()
        
        # Calculate upper and lower bands
        self.data['upper'] = self.data['sma'] + (self.n_std * self.data['std'])
        self.data['lower'] = self.data['sma'] - (self.n_std * self.data['std'])
        
        # Add volatility indicator for dynamic position sizing
        self.data['volatility'] = self.data['std'] / self.data['sma']
        
        # Add volume moving average for liquidity assessment
        self.data['volume_ma'] = self.data['Volume'].rolling(window=self.n_sma).mean()
        
        # Drop NaN values from the warmup period
        self.data = self.data.dropna()
    
    def apply_slippage(self, price, is_buy, position_size, volume):
        """
        Apply realistic slippage to price based on:
        - Direction (buy/sell)
        - Position size relative to volume
        - Base slippage rate
        
        Returns adjusted price with slippage
        """
        # Calculate market impact based on position size relative to volume
        if volume > 0:
            market_impact = min(0.05, (position_size * price / volume) * 0.1)
        else:
            market_impact = 0.01  # Default to 1% if volume data is missing
            
        # Total slippage is base slippage plus market impact
        total_slippage = self.slippage + market_impact
        
        # Apply slippage - buys pay more, sells receive less
        if is_buy:
            return price * (1 + total_slippage)  # Higher price for buys
        else:
            return price * (1 - total_slippage)  # Lower price for sells
    
    def check_liquidity(self, i, position_size, price):
        """
        Check if there's enough volume for our trade
        Scales with position size
        """
        position_value = position_size * price
        required_volume = position_value * self.min_volume_factor
        
        if i < len(self.data):
            return self.data['Volume'].iloc[i] > required_volume
        return False
    
    def calculate_position_size(self, equity, price, volatility, volume):
        """
        Calculate position size based on multiple factors:
        - Current equity
        - Price volatility
        - Available liquidity (volume)
        - Scaling down as equity grows
        
        Returns realistic position size in base units
        """
        # Reduce position size in higher volatility
        vol_factor = max(0.2, 1 - volatility * 10)  # Scale down with volatility
        
        # Apply diminishing returns to position sizing as equity grows
        # This prevents exponential equity growth that's unrealistic
        equity_scale = min(1.0, pow(self.initial_capital / equity, 0.5))
        
        # Calculate base position percentage
        adjusted_max_position_pct = self.max_position_size_pct * equity_scale * vol_factor
        
        # Calculate position value based on equity
        position_value = equity * adjusted_max_position_pct
        
        # Limit position based on volume (don't take more than 1% of volume)
        volume_limit = volume * 0.01
        position_value = min(position_value, volume_limit)
        
        # Apply absolute cap (no position larger than self.absolute_position_cap)
        position_value = min(position_value, self.absolute_position_cap)
        
        # Convert value to units
        position_size = position_value / price
        
        return position_size
        
    def run(self):
        # Initialize variables
        current_position = None  # None, 'long', 'short'
        entry_price = 0
        position_size = 0
        cash = self.initial_capital
        equity = self.initial_capital
        peak_equity = self.initial_capital
        max_drawdown = 0
        pending_signals = []  # For delayed execution
        
        # Iterate through each data point
        for i in range(1, len(self.data)):
            date = self.data.index[i]
            current_price = self.data['Close'].iloc[i]
            prev_price = self.data['Close'].iloc[i-1]
            sma = self.data['sma'].iloc[i]
            upper = self.data['upper'].iloc[i]
            lower = self.data['lower'].iloc[i]
            volatility = self.data['volatility'].iloc[i]
            volume = self.data['Volume'].iloc[i]
            
            # Process pending signals (delayed execution)
            new_pending_signals = []
            for signal in pending_signals:
                signal['delay'] -= 1
                if signal['delay'] <= 0:
                    # Execute the signal
                    signal_type = signal['type']
                    
                    if signal_type == 'buy' and current_position is None:
                        # Calculate adaptive position size
                        proposed_position_size = self.calculate_position_size(
                            equity, current_price, volatility, volume
                        )
                        
                        # Check if we have enough liquidity for this position
                        if self.check_liquidity(i, proposed_position_size, current_price):
                            # Apply slippage for entry
                            execution_price = self.apply_slippage(
                                current_price, True, proposed_position_size, volume
                            )
                            
                            # Execute trade
                            position_size = proposed_position_size
                            entry_price = execution_price
                            current_position = 'long'
                            
                            # Account for commission
                            trade_value = position_size * execution_price
                            commission_amount = trade_value * self.commission
                            cash -= (trade_value + commission_amount)
                            
                            self.trades.append({
                                'date': date,
                                'type': 'buy',
                                'price': execution_price,
                                'size': position_size,
                                'value': trade_value,
                                'commission': commission_amount
                            })
                            
                            print(f"BUY at {date}: Price={execution_price:.2f}, Size={position_size:.6f}, Value={trade_value:.2f}")
                    
                    elif signal_type == 'sell' and current_position is None:
                        # Calculate adaptive position size
                        proposed_position_size = self.calculate_position_size(
                            equity, current_price, volatility, volume
                        )
                        
                        # Check if we have enough liquidity for this position
                        if self.check_liquidity(i, proposed_position_size, current_price):
                            # Apply slippage for entry
                            execution_price = self.apply_slippage(
                                current_price, False, proposed_position_size, volume
                            )
                            
                            # Execute trade
                            position_size = proposed_position_size
                            entry_price = execution_price
                            current_position = 'short'
                            
                            # Reserve cash for potential short position losses
                            # For shorts, we reserve a portion of the position value
                            trade_value = position_size * execution_price
                            reserve_amount = trade_value * 0.5  # Reserve 50% of position value
                            commission_amount = trade_value * self.commission
                            cash -= (reserve_amount + commission_amount)
                            
                            self.trades.append({
                                'date': date,
                                'type': 'sell',
                                'price': execution_price,
                                'size': position_size,
                                'value': trade_value,
                                'commission': commission_amount,
                                'reserve': reserve_amount
                            })
                            
                            print(f"SELL at {date}: Price={execution_price:.2f}, Size={position_size:.6f}, Value={trade_value:.2f}")
                else:
                    # Keep in pending list
                    new_pending_signals.append(signal)
            
            # Update pending signals
            pending_signals = new_pending_signals
            
            # Update equity calculation based on position
            if current_position == 'long':
                # For long positions: equity = cash + position value
                position_value = position_size * current_price
                equity = cash + position_value
                
                # Check stop loss for long positions
                stop_price = entry_price * (1 - self.stop_loss_pct)
                if current_price <= stop_price:
                    # Execute stop loss - apply slippage for exit (worse for panic selling)
                    exit_price = self.apply_slippage(
                        current_price, False, position_size, volume * 0.8  # Less liquidity in stop-out
                    )
                    
                    # Calculate trade value and costs
                    trade_value = position_size * exit_price
                    commission_amount = trade_value * self.commission
                    exit_value = trade_value - commission_amount
                    
                    # Calculate PnL
                    entry_value = position_size * entry_price
                    pnl = exit_value - entry_value
                    
                    # Update cash
                    cash += exit_value
                    
                    # Record the trade
                    self.trades.append({
                        'date': date,
                        'type': 'stop_loss_long',
                        'price': exit_price,
                        'size': position_size,
                        'value': trade_value,
                        'commission': commission_amount,
                        'pnl': pnl,
                        'pnl_pct': (pnl / entry_value) * 100
                    })
                    
                    print(f"STOP LOSS LONG at {date}: Price={exit_price:.2f}, PnL={pnl:.2f}")
                    
                    # Reset position
                    current_position = None
                    position_size = 0
                
            elif current_position == 'short':
                # For short positions: equity = cash + (entry_price - current_price) * position_size
                # Short positions make money when price falls from entry_price
                position_value = position_size * (entry_price - current_price)
                equity = cash + position_value
                
                # Check stop loss for short positions
                stop_price = entry_price * (1 + self.stop_loss_pct)
                if current_price >= stop_price:
                    # Execute stop loss - apply slippage for exit (worse for panic buying)
                    exit_price = self.apply_slippage(
                        current_price, True, position_size, volume * 0.8  # Less liquidity in stop-out
                    )
                    
                    # Calculate trade value and costs
                    trade_value = position_size * exit_price
                    commission_amount = trade_value * self.commission
                    
                    # For shorts: we make money if exit_price < entry_price
                    entry_value = position_size * entry_price
                    pnl = entry_value - trade_value - commission_amount
                    
                    # Return the reserved cash plus PnL
                    reserved_amount = entry_value * 0.5  # The amount we reserved at entry
                    cash += (reserved_amount + pnl)
                    
                    # Record the trade
                    self.trades.append({
                        'date': date,
                        'type': 'stop_loss_short',
                        'price': exit_price,
                        'size': position_size,
                        'value': trade_value,
                        'commission': commission_amount,
                        'pnl': pnl,
                        'pnl_pct': (pnl / entry_value) * 100
                    })
                    
                    print(f"STOP LOSS SHORT at {date}: Price={exit_price:.2f}, PnL={pnl:.2f}")
                    
                    # Reset position
                    current_position = None
                    position_size = 0
            else:
                equity = cash
            
            # Record equity for this timestamp
            self.equity_curve.append((date, equity))
            
            # Update peak equity and calculate drawdown
            peak_equity = max(peak_equity, equity)
            current_drawdown = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Check max drawdown threshold - stop trading if exceeded
            if current_drawdown >= self.max_drawdown_pct:
                print(f"Max drawdown threshold ({self.max_drawdown_pct*100}%) reached at {date}. Stopping trading.")
                break
            
            # Check for entry/exit signals
            
            # Entry: If we don't have a position and price crosses below lower band
            if (current_position is None and 
                prev_price >= lower and 
                current_price < lower):
                
                # Add signal with delay
                pending_signals.append({
                    'type': 'buy',
                    'delay': self.execution_delay
                })
                
            # Entry: If we don't have a position and price crosses above upper band
            elif (current_position is None and 
                  prev_price <= upper and 
                  current_price > upper):
                
                # Add signal with delay
                pending_signals.append({
                    'type': 'sell',
                    'delay': self.execution_delay
                })
                
            # Exit long position: If we're long and price crosses above SMA
            elif current_position == 'long' and prev_price <= sma and current_price > sma:
                # Apply slippage for exit
                exit_price = self.apply_slippage(
                    current_price, False, position_size, volume
                )
                
                # Calculate trade value and costs
                trade_value = position_size * exit_price
                commission_amount = trade_value * self.commission
                exit_value = trade_value - commission_amount
                
                # Calculate PnL
                entry_value = position_size * entry_price
                pnl = exit_value - entry_value
                
                # Update cash
                cash += exit_value
                
                # Record the trade
                self.trades.append({
                    'date': date,
                    'type': 'close_long',
                    'price': exit_price,
                    'size': position_size,
                    'value': trade_value,
                    'commission': commission_amount,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_value) * 100
                })
                
                print(f"CLOSE LONG at {date}: Price={exit_price:.2f}, PnL={pnl:.2f}")
                
                # Reset position
                current_position = None
                position_size = 0
                
            # Exit short position: If we're short and price crosses below SMA
            elif current_position == 'short' and prev_price >= sma and current_price < sma:
                # Apply slippage for exit
                exit_price = self.apply_slippage(
                    current_price, True, position_size, volume
                )
                
                # Calculate trade value and costs
                trade_value = position_size * exit_price
                commission_amount = trade_value * self.commission
                
                # For shorts: we make money if exit_price < entry_price
                entry_value = position_size * entry_price
                pnl = entry_value - trade_value - commission_amount
                
                # Return the reserved cash plus PnL
                reserved_amount = entry_value * 0.5  # The amount we reserved at entry
                cash += (reserved_amount + pnl)
                
                # Record the trade
                self.trades.append({
                    'date': date,
                    'type': 'close_short',
                    'price': exit_price,
                    'size': position_size,
                    'value': trade_value,
                    'commission': commission_amount,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_value) * 100
                })
                
                print(f"CLOSE SHORT at {date}: Price={exit_price:.2f}, PnL={pnl:.2f}")
                
                # Reset position
                current_position = None
                position_size = 0
        
        # Close any open position at the end
        if current_position is not None:
            last_date = self.data.index[-1]
            last_price = self.data['Close'].iloc[-1]
            last_volume = self.data['Volume'].iloc[-1]
            
            # Apply slippage for exit
            execution_price = self.apply_slippage(
                last_price, current_position == 'short', position_size, last_volume
            )
            
            if current_position == 'long':
                # Calculate trade value and costs
                trade_value = position_size * execution_price
                commission_amount = trade_value * self.commission
                exit_value = trade_value - commission_amount
                
                # Calculate PnL
                entry_value = position_size * entry_price
                pnl = exit_value - entry_value
                
                # Update cash
                cash += exit_value
            else:  # short
                # Calculate trade value and costs
                trade_value = position_size * execution_price
                commission_amount = trade_value * self.commission
                
                # For shorts: we make money if exit_price < entry_price
                entry_value = position_size * entry_price
                pnl = entry_value - trade_value - commission_amount
                
                # Return the reserved cash plus PnL
                reserved_amount = entry_value * 0.5  # The amount we reserved at entry
                cash += (reserved_amount + pnl)
                
            self.trades.append({
                'date': last_date,
                'type': f'close_{current_position}_end',
                'price': execution_price,
                'size': position_size,
                'value': trade_value,
                'commission': commission_amount,
                'pnl': pnl,
                'pnl_pct': (pnl / entry_value) * 100
            })
            
            print(f"CLOSE {current_position.upper()} at END: Price={execution_price:.2f}, PnL={pnl:.2f}")
        
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
            print("Cannot proceed without real market data. Exiting...")
            exit(1)  # Exit instead of generating sample data

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
    
    # Define parameter ranges to test (expanded range)
    sma_values = [10, 15, 20, 25, 30, 35, 40]
    std_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    stop_loss_values = [0.02, 0.03, 0.05, 0.07]
    max_position_size_values = [0.05, 0.10, 0.15]
    
    # Track the best parameters and performance
    best_return = -np.inf
    best_sharpe = -np.inf
    best_params = None
    best_stats = None
    
    # Try key combinations first (focusing on the most impactful parameters)
    total_combinations = len(sma_values) * len(std_values)
    current_combination = 0
    
    results = []
    
    for n_sma in sma_values:
        for n_std in std_values:
            current_combination += 1
            print(f"Testing combination {current_combination}/{total_combinations}: SMA={n_sma}, STD={n_std}")
            
            # Run strategy with these parameters
            strategy = MeanReversionStrategy(
                data, 
                n_sma=n_sma, 
                n_std=n_std,
                commission=0.001,  # 0.1% commission rate
                slippage=0.0005    # 0.05% slippage
            )
            stats = strategy.run()
            
            # Store all results for analysis
            results.append({
                'n_sma': n_sma,
                'n_std': n_std,
                'return': stats['Return [%]'],
                'sharpe': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'win_rate': stats['Win Rate [%]'],
                'trades': stats['# Trades']
            })
            
            # Update best parameters if this combination has a better return
            if stats['Return [%]'] > best_return:
                best_return = stats['Return [%]']
                best_params = {'n_sma': n_sma, 'n_std': n_std}
                best_stats = stats
            
            # Also track best Sharpe ratio (risk-adjusted return)
            if stats['Sharpe Ratio'] > best_sharpe and stats['Return [%]'] > 0:
                best_sharpe = stats['Sharpe Ratio']
    
    # Now test risk parameters with the best SMA and STD values
    print("\nFine-tuning risk parameters with best SMA and STD values...")
    
    for stop_loss in stop_loss_values:
        for position_size in max_position_size_values:
            print(f"Testing: SMA={best_params['n_sma']}, STD={best_params['n_std']}, Stop Loss={stop_loss*100}%, Position Size={position_size*100}%")
            
            strategy = MeanReversionStrategy(
                data, 
                n_sma=best_params['n_sma'], 
                n_std=best_params['n_std'],
                stop_loss_pct=stop_loss,
                max_position_size_pct=position_size,
                commission=0.001,
                slippage=0.0005
            )
            stats = strategy.run()
            
            # Update best parameters if this combination has a better return
            if stats['Return [%]'] > best_return:
                best_return = stats['Return [%]']
                best_params = {
                    'n_sma': best_params['n_sma'], 
                    'n_std': best_params['n_std'],
                    'stop_loss_pct': stop_loss,
                    'max_position_size_pct': position_size
                }
                best_stats = stats
    
    # Sort and analyze results
    sorted_by_return = sorted(results, key=lambda x: x['return'], reverse=True)
    sorted_by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    
    print("\nOptimization Results:")
    print(f"Best Parameters for Maximum Return:")
    print(f"SMA = {best_params['n_sma']}")
    print(f"STD Multiplier = {best_params['n_std']}")
    if 'stop_loss_pct' in best_params:
        print(f"Stop Loss = {best_params['stop_loss_pct']*100:.1f}%")
    if 'max_position_size_pct' in best_params:
        print(f"Max Position Size = {best_params['max_position_size_pct']*100:.1f}%")
    print(f"Return: {best_stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {best_stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {best_stats['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {best_stats['Win Rate [%]']:.2f}%")
    print(f"Number of Trades: {best_stats['# Trades']}")
    
    # Also show top 5 configurations by return
    print("\nTop 5 Configurations by Return:")
    for i, result in enumerate(sorted_by_return[:5]):
        print(f"{i+1}. SMA={result['n_sma']}, STD={result['n_std']}, Return={result['return']:.2f}%, Sharpe={result['sharpe']:.2f}")
    
    # Also show top 5 configurations by Sharpe ratio
    print("\nTop 5 Configurations by Sharpe Ratio:")
    for i, result in enumerate(sorted_by_sharpe[:5]):
        print(f"{i+1}. SMA={result['n_sma']}, STD={result['n_std']}, Sharpe={result['sharpe']:.2f}, Return={result['return']:.2f}%")
    
    # Run the strategy with the best parameters and plot
    final_strategy = MeanReversionStrategy(
        data, 
        n_sma=best_params['n_sma'], 
        n_std=best_params['n_std'],
        stop_loss_pct=best_params.get('stop_loss_pct', 0.05),
        max_position_size_pct=best_params.get('max_position_size_pct', 0.1),
        commission=0.001,
        slippage=0.0005
    )
    final_stats = final_strategy.run()
    final_strategy.plot(filename='optimized_mean_reversion_results.png')
    
    # Create config code snippet to add to the main function
    config_code = f"""
    # Optimized configuration
    strategy = MeanReversionStrategy(
        data,
        n_sma={best_params['n_sma']},
        n_std={best_params['n_std']},
        stop_loss_pct={best_params.get('stop_loss_pct', 0.05)},
        max_position_size_pct={best_params.get('max_position_size_pct', 0.1)},
        commission=0.001,  # 0.1% commission rate
        slippage=0.0005,   # 0.05% slippage
        max_drawdown_pct=0.25  # Stop trading if drawdown exceeds 25%
    )
    """
    
    print("\nConfiguration code to add to your script:")
    print(config_code)
    
    return best_stats, best_params, config_code



# Main function# Add this function after the optimize_strategy function
def run_multiple_period_tests(data, best_params=None):
    """
    Run the strategy across different market regimes and validation periods
    
    Parameters:
    - data: Full dataset
    - best_params: Optional parameters from optimization
    
    Returns:
    - Dictionary with results from each period
    """
    print("\nRunning tests across multiple market regimes...")
    
    # Define periods (you can adjust these based on known market regimes)
    periods = []
    
    # Calculate total length and split into segments
    total_days = (data.index[-1] - data.index[0]).days
    segment_size = total_days // 4  # Split into quarters
    
    # Create periods with overlap for better validation
    for i in range(4):
        start_idx = max(0, len(data) // 4 * i - len(data) // 8)  # Allow for overlap
        end_idx = min(len(data), start_idx + len(data) // 3)
        
        period_data = data.iloc[start_idx:end_idx].copy()
        period_start = period_data.index[0]
        period_end = period_data.index[-1]
        
        # Identify market regime based on trend and volatility
        returns = period_data['Close'].pct_change()
        total_return = (period_data['Close'].iloc[-1] / period_data['Close'].iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Classify the regime
        if total_return > 20:
            regime = "Bull Market"
        elif total_return < -20:
            regime = "Bear Market"
        else:
            regime = "Sideways Market"
            
        if volatility > 0.5:  # 50% annualized volatility
            regime += " (High Volatility)"
        else:
            regime += " (Low Volatility)"
            
        periods.append({
            'data': period_data,
            'start': period_start,
            'end': period_end,
            'regime': regime
        })
    
    # Add a validation period from the most recent data (25% of dataset)
    validation_size = len(data) // 4
    validation_data = data.iloc[-validation_size:].copy()
    
    # Calculate regime for validation data
    val_returns = validation_data['Close'].pct_change()
    val_total_return = (validation_data['Close'].iloc[-1] / validation_data['Close'].iloc[0] - 1) * 100
    val_volatility = val_returns.std() * np.sqrt(252)
    
    # Classify validation regime
    if val_total_return > 20:
        val_regime = "Bull Market"
    elif val_total_return < -20:
        val_regime = "Bear Market"
    else:
        val_regime = "Sideways Market"
        
    if val_volatility > 0.5:
        val_regime += " (High Volatility)"
    else:
        val_regime += " (Low Volatility)"
    
    periods.append({
        'data': validation_data,
        'start': validation_data.index[0],
        'end': validation_data.index[-1],
        'regime': f"Out-of-Sample Validation: {val_regime}"
    })
    
    # Run tests for each period
    results = {}
    for i, period in enumerate(periods):
        print(f"\nTesting Period {i+1}: {period['start']} to {period['end']}")
        print(f"Market Regime: {period['regime']}")
        
        # Use optimal parameters if provided, otherwise use defaults
        if best_params:
            strategy = MeanReversionStrategy(
                period['data'], 
                n_sma=best_params['n_sma'], 
                n_std=best_params['n_std']
            )
        else:
            strategy = MeanReversionStrategy(period['data'])
        
        stats = strategy.run()
        
        # Store results
        results[f"Period {i+1} ({period['regime']})"] = {
            'stats': stats,
            'equity_curve': strategy.equity_curve
        }
        
        # Print key metrics
        print(f"Return: {stats['Return [%]']:.2f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Number of Trades: {stats['# Trades']}")
        
        # Save period chart
        strategy.plot(filename=f"period_{i+1}_results.png")
    
    # Calculate aggregate statistics
    agg_return = np.mean([r['stats']['Return [%]'] for r in results.values()])
    agg_sharpe = np.mean([r['stats']['Sharpe Ratio'] for r in results.values()])
    agg_drawdown = np.mean([r['stats']['Max. Drawdown [%]'] for r in results.values()])
    
    print("\nAggregate Results Across All Periods:")
    print(f"Average Return: {agg_return:.2f}%")
    print(f"Average Sharpe Ratio: {agg_sharpe:.2f}")
    print(f"Average Max Drawdown: {agg_drawdown:.2f}%")
    
    return results
def main():
    try:
        # Date range for the backtest
        end_date = datetime(2025, 3, 20)
        start_date = end_date - timedelta(days=180)  # 6 months of data for more robust testing
        
        # Download real ETH/USDT data
        data = download_eth_data(start_date, end_date)
        
        # Print sample of the data
        print("\nSample data:")
        print(data.head())
        
        # Ensure data is properly formatted
        # Convert column names to title case if needed
        if 'close' in data.columns and 'Close' not in data.columns:
            data.columns = [col.title() for col in data.columns]
        
        # Run the backtest with realistic settings
        strategy = MeanReversionStrategy(
            data,
            commission=0.001,  # 0.1% commission rate
            slippage=0.0005,   # 0.05% slippage
            max_position_size_pct=0.1,  # More conservative position sizing
            stop_loss_pct=0.05,  # 5% stop loss
            max_drawdown_pct=0.25  # Stop trading if drawdown exceeds 25%
        )
        
        # Run basic backtest
        print("\nRunning initial backtest with realistic settings...")
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
                best_stats, best_params = optimize_strategy(data)
                
                # Run cross-validation with optimized parameters
                print("\nRunning cross-validation with optimized parameters...")
                period_results = run_multiple_period_tests(data, best_params)
            else:
                # Run cross-validation with default parameters
                print("\nRunning cross-validation with default parameters...")
                period_results = run_multiple_period_tests(data)
        else:
            print("\nNo trades were executed. Consider running parameter optimization.")
            run_opt = input("Would you like to run parameter optimization? (y/n): ")
            if run_opt.lower() == 'y':
                best_stats, best_params = optimize_strategy(data)
                
                # Run cross-validation with optimized parameters
                print("\nRunning cross-validation with optimized parameters...")
                period_results = run_multiple_period_tests(data, best_params)
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
