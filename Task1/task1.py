import os
import joblib

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

OUT = './output1'

# Import data
df = pd.read_csv('medical_time_series .csv', index_col='Day')

# Preview data
print(df.head().to_string(), '\n')

# Examine the data structure
df.info()
print()

# Check for duplicates
print('Duplicates:', df.duplicated().sum(), '\n')

# Check for missing values
print('Missing values:')
print(df.isnull().sum().to_string(), '\n')

# Convert DataFrame to timeseries object
# https://stackoverflow.com/a/62098606
df.index = pd.to_datetime(df.index, unit='D',
                          origin=pd.Timestamp('2019-12-31')).to_period('D')
print('Datetime index:\n', df.head().to_string(), '\n', df.tail().to_string(), '\n')

# Summarize the data using descriptive statistics
print('Data summary:\n', df.describe(), '\n')

# Plot preliminary data for visual understanding
_, ax1 = plt.subplots(1, figsize=(10, 3))
df.plot(grid=True, ax=ax1, title='Realized Revenue by Day')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'realized_revenue_by_day.png'))
plt.show(block=False)
plt.close()

# ADFuller to test for stationarity
adf = adfuller(df, autolag='AIC')
adf_df = pd.Series(adf[:4], index=['Test Statistic', 'p-value',
                                   'Lags', 'Observations'])
for k, v in adf[4].items():
    adf_df[f'Critical Value {k}'] = v
print(f'Augmented Dickey-Fuller test:\n{adf_df.to_string()}\n')

# Take the difference to coerce stationarity
print('Differencing the data to coerce stationarity...')
df_diff = df.diff().dropna()

# Verify stationarity
adf = adfuller(df_diff, autolag='AIC')
adf_df = pd.Series(adf[:4], index=['Test Statistic', 'p-value',
                                   'Lags', 'Observations'])
for k, v in adf[4].items():
    adf_df[f'Critical Value {k}'] = v
print(f'Augmented Dickey-Fuller test:\n{adf_df.to_string()}\n')

# Split the pre-differenced data saving last month for test
df_train = df.loc[:'2021-11-30']
df_test = df.loc['2021-12-01':]
df_train.to_csv(os.path.join(OUT, 'training_data.csv'))
df_test.to_csv(os.path.join(OUT, 'testing_data.csv'))

# Check for seasonality
detrend = df_train - df_train.rolling(180).mean()
detrend.dropna(inplace=True)
fig, ax = plt.subplots(figsize=(10, 3))
fig.suptitle('Detrended Daily Revenue')
plot_acf(detrend, ax=ax, lags=180, zero=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'detrended_acf.png'))
plt.show(block=False)
plt.close()

# Auto-ARIMA to get p, d, q and check for seasonality
stepwise_fit = auto_arima(df_train.Revenue, trace=True,
                          suppress_warnings=True)

# Inspect trends
data = {}
trends = [30, 90, 180, 365]
_, ax = plt.subplots(figsize=(10, 3))
for t in trends:
    data[t] = df_train.rolling(t).mean()
    data[t].columns = [f'Rolling {t} Day Mean']
    data[t].plot(ax=ax, grid=True)
ax.set_title('Trends')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'trends.png'))
plt.show(block=False)
plt.close()

# Perform acf/pacf
df_train_diff = df_train.diff().dropna()
_, (ax1, ax2) = plt.subplots(2, figsize=(10, 6), sharey=True)
plot_acf(df_train_diff, zero=False, ax=ax1)
plot_pacf(df_train_diff, zero=False, method='ywm', ax=ax2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'acf_pacf.png'))
plt.show(block=False)
plt.close()

# Perform spectral density to see the periodicity
_, ax = plt.subplots(figsize=(10, 3))
ax.psd(detrend.Revenue)
ax.set_title('Spectral Density of Detrended Revenue')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'spectral_density.png'))
plt.show(block=False)
plt.close()

# Decompose to see components of time series
decomp_results = seasonal_decompose(df_train.Revenue, model='additive',
                                    period=90)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 12), sharex=True)
fig.suptitle(f'Seasonal Decomposition (period=90)')
decomp_results.observed.plot(grid=True, ax=ax1, title='Original')
decomp_results.trend.plot(grid=True, ax=ax2, title='Trend')
decomp_results.seasonal.plot(grid=True, ax=ax3, title='Seasons')
decomp_results.resid.plot(grid=True, ax=ax4, title='Residuals')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'seasonal_decompose.png'))
plt.show(block=False)
plt.close()

# Confirmation of lack of trends in residuals
mean_residuals = decomp_results.resid.rolling(90).mean()
mean_residuals.dropna(inplace=False)
_, ax = plt.subplots(figsize=(10, 3))
decomp_results.resid.plot(ax=ax, label='Residuals')
mean_residuals.plot(ax=ax, label='Rolling 90-day Mean Residuals',
                    grid=True, title='Residuals')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'residuals.png'))
plt.show(block=False)
plt.close()

# Create and fit ARIMA(p,d,q) model
mod = ARIMA(df_train, order=(1, 1, 0))
res = mod.fit()
print(res.summary(), '\n')

# Evaluate the model
mae = np.mean(np.abs(res.resid))
print('Mean Absolute Error:', mae, '\n')
res.plot_diagnostics(figsize=(9, 6))
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'arima_diagnostic'))
plt.show(block=False)
plt.close()

# Predict with the test data
forecast = res.get_forecast(30)
ci = forecast.conf_int()
_, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
df.plot(ax=ax1, color='b')
forecast.predicted_mean.plot(ax=ax1, color='r', label='Forecasted Revenue')
ax1.fill_between(ci.index, ci['lower Revenue'], ci['upper Revenue'],
                 color='r', alpha=0.2, label='95% Confidence Interval')
ax1.set_title('Observed Revenue with 30-day Forecast')
df_test[:'2021-12-30'].plot(ax=ax2, color='b')
forecast.predicted_mean.plot(ax=ax2, color='r', label='Forecasted Revenue')
ax2.fill_between(ci.index, ci['lower Revenue'], ci['upper Revenue'],
                 color='r', alpha=0.2, label='95% Confidence Interval')
ax2.set_title('30-day Observed vs Forecasted Revenue')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'test_predict.png'))
plt.show(block=False)
plt.close()

# Save the model
joblib.dump(res, os.path.join(OUT, 'model_results.pkl'))

# Evaluate forecast mean absolute error for 1-30 days
metric_df = df_test.join(forecast.predicted_mean)
metric_df['resid'] = metric_df.predicted_mean.sub(metric_df.Revenue)
metric_df['cum_mae'] = metric_df.resid.abs().cumsum().div(metric_df.index.day)
metric_df['cum_mae_pct_Revenue'] = metric_df.cum_mae.div(metric_df.Revenue)
print(metric_df.to_string())

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 10))
fig.suptitle('30-day Forecast Metrics')
metric_df.resid.plot(ax=ax1, grid=True, marker='o', title='Error')
metric_df.cum_mae.plot(ax=ax2, grid=True, marker='o',
                       title='Cumulative Mean Absolute Error')
metric_df.cum_mae_pct_Revenue.plot(ax=ax3, grid=True, marker='o',
                                   title='Percent Cumulative Mean '
                                         'Absolute Error of Revenue')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'forecast_metrics.png'))
plt.show(block=False)
plt.close()
