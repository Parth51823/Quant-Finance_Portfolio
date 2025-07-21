import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST
import datetime as dt


# 1. get SPY data from Alpaca 
api = REST('PKOEY8NK7JXD8WBD8J82','74b9K2VsGyY68yx7w5Ijov5hKdRXoGCFT0h4EjkT', base_url="https://paper-api.alpaca.markets/v2")
spy = api.get_bars("SPY","1D", start="2019-01-01", end="2024-12-31").df


# Renaming open as spy_price
df = spy[['open']].rename(columns={'open':'spy_price'})

# removing timezone info if present
df.index = df.index.tz_localize(None)
# normalizing date so all the time stamps are set to 00:00:00
df['date'] = df.index.normalize()
# setting index to date
df.set_index('date', inplace=True)
print(df.index)

df.to_csv('/Users/parthaggarwal/Desktop/Quant_Learning/Projects/Equity_crypto_analysis/datasets/SPY_5year')

# 2. creating moving averages
df['sma_30_spy'] = df['spy_price'].rolling(30).mean()
df['sma_200_spy'] = df['spy_price'].rolling(200).mean()


# 3. Generating signals
df['signal'] = 0
df.loc[df.index[30:], 'signal'] = np.where(df['sma_30_spy'][30:]> df['sma_200_spy'][30:], 1, 0)
# lag to avoid the lookahead bias
df['position'] = df['signal'].shift(1)


# 4. Computing strategy returns
df['returns'] = df['spy_price'].pct_change()
df['strategy_returns'] = df['position'] * df['returns']

# 5. Cumulative returns
df[['returns', 'strategy_returns']] = df[['returns', 'strategy_returns']].fillna(0)
df['cumulative_strategy'] = (1+df['strategy_returns']).cumprod()
df['buy_and_hold_returns'] = (1+df['returns']).cumprod()


# 6. plot 
plt.figure(figsize=(12,6))
plt.plot(df['cumulative_strategy'], label='Strategy')
plt.plot(df['buy_and_hold_returns'], label='Buy and Hold')
plt.title('Moving average strategy vs Buy and hold')
plt.legend()
plt.grid(True)
plt.show()

# performance metrics
def performance_metrics(returns, risk_free=0.00):
    sharpe = (returns.mean() - returns.std()) * np.sqrt(252)
    cumulative = (1+returns).cumprod()[-1] - 1
    max_dd = (1 - (df['cumulative_strategy']/df['cumulative_strategy'].cummax())).max()
    return round(sharpe, 2), round(cumulative * 100, 2), round(max_dd * 100, 2)

sharpe, cum_return, max_draw = performance_metrics(df['strategy_returns'])
print(f"Sharpe ratio is {sharpe}, Cumulative return is {cum_return}%, Maximum DrawDown is {max_draw}%")

sharpe, cum_return, max_draw = performance_metrics(df['returns'])
print(f"Sharpe ratio is {sharpe}, Cumulative return is {cum_return}%, Maximum DrawDown is {max_draw}%")
df