import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.tsa.api as tsa
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def test_stationarity(timeseries, alpha=1e-3):
    dftest = tsa.adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput["Critical Value(%s)" % key] = value

    print(dfoutput)
    critical_value = dftest[4]["5%"]
    test_statistic = dftest[0]
    pvalue = dftest[1]
    if pvalue < alpha and test_statistic < critical_value:
        print("X is stationary")
        return True
    else:
        print("X is not stationary")
        return False

data = pd.read_csv('../utils/counts_serial.csv', engine='python', skipfooter=3)
# A bit of pre-processing to make it nicer
data['time']=pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
data.set_index(['time'], inplace=True)
NGE = data["fault_nums"]
print(NGE.head())

fig, ax = plt.subplots(figsize=(15, 15))
NGE.plot(ax=ax, fontsize=15)
ax.set_title("故障数量变化图", fontsize=25)
ax.set_xlabel("时间", fontsize=25)
ax.set_ylabel("故障数量", fontsize=25)
ax.legend(loc="best", fontsize=15)
ax.grid()
plt.show(ax)

test_stationarity(NGE)

nge_seasonal = NGE.diff(96)
test_stationarity(nge_seasonal.dropna())

from statsmodels.stats.diagnostic import acorr_ljungbox
def test_white_noise(data):
    return acorr_ljungbox(data.dropna(), return_df=True)

test_white_noise(nge_seasonal)

fig = plot_acf(nge_seasonal.dropna(), lags=40)
fig = plot_pacf(nge_seasonal.dropna(), lags=40)
plt.show()


def grid_search(data):
    p = q = range(0, 3)
    s = [12]
    d = [1]
    PDQs = list(itertools.product(p, d, q, s))
    pdq = list(itertools.product(p, d, q))
    params = []
    seasonal_params = []
    results = []
    grid = pd.DataFrame()

    for param in pdq:
        for seasonal_param in PDQs:
            mod = tsa.SARIMAX(data, order=param, seasonal_order=seasonal_param, enforce_stationarity=False,
                              enforce_invertibility=False)
            result = mod.fit()
            print("ARIMA{}x{} - AIC:{}".format(param, seasonal_param, result.aic))
            params.append(param)
            seasonal_params.append(seasonal_param)
            results.append(result.aic)

    grid["pdq"] = params
    grid["PDQs"] = seasonal_params
    grid["aic"] = results
    print(grid[grid["aic"] == grid["aic"].min()])

grid_search(nge_seasonal.dropna())

mod = tsa.SARIMAX(NGE, order=(1, 1, 2), seasonal_order=(2, 1, 2, 12))
results = mod.fit()
test_white_noise(results.resid)
fig_result = results.plot_diagnostics(figsize=(15, 12))