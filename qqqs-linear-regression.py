import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# Plan: When in bear market (def: decline of 20% from peak), implement strategy
# (buy QQQS when price 1 sd below reg line and sell when 1 sd above reg line)
# until 388 days (average duration of a bear market) from peak

qqq_data = yf.Ticker("QQQ").history(period='2y')
qqq_prices = pd.DataFrame(qqq_data['Close'].round(decimals=2))
qqq_peak_price = max(qqq_prices['Close'])
qqq_peak_date = qqq_prices.index[qqq_prices['Close'] == qqq_peak_price][0].date()
end_of_strategy_date = qqq_peak_date + timedelta(days=388)
dates_after_peak = qqq_prices.loc[str(qqq_peak_date):]
qqq_strategy_start_price = [dates_after_peak['Close'][i] for i in range(len(dates_after_peak)) if
                            dates_after_peak['Close'][i] <= 0.8 * qqq_peak_price][0]
qqq_strategy_start_date = dates_after_peak.index[dates_after_peak['Close'] == qqq_strategy_start_price][0].date()

qqqs_data = yf.Ticker("QQQS.L").history(start=f'{qqq_peak_date}', end=f'{end_of_strategy_date}')
qqqs_prices = pd.DataFrame(data=qqqs_data['Close'].round(decimals=2))

qqqs_prices['x'] = np.arange(len(qqqs_prices))
qqqs_prices['y'] = qqqs_prices['Close']

X = qqqs_prices['x'].values[:, np.newaxis]
y = qqqs_prices['y'].values[:, np.newaxis]

lin_reg = LinearRegression()
lin_reg.fit(X, y)

qqqs_prices['y_pred'] = lin_reg.predict(qqqs_prices['x'].values[:, np.newaxis])
y_pred = qqqs_prices['y_pred'].values

qqqs_prices['dev'] = (y_pred - qqqs_prices['Close'].values) ** 2
dev = qqqs_prices['dev'].values

vol = []

for i in range(1, len(qqqs_prices['Close']) + 1):
    if i >= 90:
        vol.append(sum(dev[i - 90:i]) / len(dev[i - 90:i]))
    else:
        vol.append(sum(dev[:i]) / len(dev[:i]))

vol = [(vol[i]) ** .5 for i in range(len(vol))]

qqqs_prices['above'] = y_pred + vol
qqqs_prices['below'] = y_pred - vol

new = qqqs_prices.drop(['x', 'Close', 'dev'], axis=1)

plt.style.use('dark_background')
plt.plot(new)
plt.title('QQQS')
plt.ylabel('Price (USD)')
plt.xlabel('Date')
plt.legend(list(new.columns))
plt.show()
