from datetime import date, timedelta
import pandas as pd
import numpy as np
import time
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings('ignore')


def predict(data, date, p, q, stat_params, window=60):

    data['date'] = data.index
    data['date'] = data['date'].apply(lambda d: pd.to_datetime(d))
    a1_list = data.columns.unique().drop("date")

    test_start = pd.Timestamp(date)
    test_end = test_start + timedelta(days=window)

    pred_dict = {}
    test_dict = {}

    t0 = time.time()

    for cnt, a in enumerate(a1_list):
        d1 = data[[a, "date"]].copy(deep=True)
        if stat_params is None:
            dftest = adfuller(d1[a], autolag='AIC')
            print(dftest)
            bl = dftest[0] < dftest[4]['5%']
            if not bl:
                stat_param = 1
            else:
                stat_param = 0
        else:
            stat_param = stat_params[a]
        print("stationary parameter ", stat_param)
        var = d1.copy()
        var.set_index('date', inplace=True)
        var = var.rolling(10).mean()
        var = var.dropna()
        train = var[:test_start - timedelta(days=1)]
        test = var[test_start:test_end - timedelta(days=1)]
        print(a, " start")

        model = ARIMA(train, order=(p, stat_param, q))
        trained_arima = model.fit()
        pred = trained_arima.predict(start=train.shape[0], end=train.shape[0] + 59)


        pred.index = test.index

        # pred30 = pred[:30]
        predwindow = pred[:window].to_frame()
        predwindow.columns = [f"prediction {a}"]
        test.columns = [f"target {a}"]
        pred_dict[a] = predwindow
        test_dict[a] = test

    t1 = time.time()

    all_pred = pd.concat(list(pred_dict.values()), axis=1)
    all_test = pd.concat(list(test_dict.values()), axis=1)

    res = pd.concat([all_pred, all_test], axis=1)

    rmse = np.sqrt(np.mean((np.array(all_pred) - np.array(all_test))**2))

    res["RMSE"] = rmse
    res["training time"] = t1 -t0

    return res


if __name__ == "__main__":
    for country in ["Yemen", "Syria", "Mali", "Nigeria"]:
        full_data = pd.read_csv(
            f"data/{country}/full_timeseries_daily.csv",
            index_col=0,
            header=[0, 1],
        )["FCS"]
        dtindex = pd.DatetimeIndex(full_data.index)
        full_data.index = dtindex
        full_data = full_data[:full_data.last_valid_index()]
        stat_params = {}
        hp_df = pd.read_csv(f"best_hyperparameters/HP_ARIMA_{country}.csv")
        dates = list(hp_df["date"])

        for a in full_data.columns:
            dftest = adfuller(full_data[a], autolag='AIC')
            print(dftest)
            bl = dftest[0] < dftest[4]['5%']
            if not bl:
                stat_params[a] = 1
            else:
                stat_params[a] = 0

        for date in dates:
            print(date)
            hp = hp_df.loc[hp_df["date"] == date]
            p = hp_df["p"].iloc[0]
            q = hp_df["q"].iloc[0]
            res = predict(data=full_data.copy(deep=True), date=date, p=p, q=q, stat_params=stat_params, window=60)
            res.to_csv(f"./forecasts/ARIMA/{country}_{date}.csv")




