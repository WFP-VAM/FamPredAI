
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os


def shift_dataframe_by_date(dataframe, target_date):
    """
    Shift each column of the pandas DataFrame so that the last valid index is at least the specified date.

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame with a datetime index and numerical columns.
        target_date (str or pd.Timestamp): The desired date as a string in 'yyyy-mm-dd' format or as a pandas Timestamp.

    Returns:
        pandas.DataFrame: The new DataFrame with shifted values.
    """
    # Convert the target_date to a pandas Timestamp object if it's provided as a string.
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)

    # Shift the last valid index for each column.
    shifted_dataframe = dataframe.copy()
    for col in dataframe.columns:
        last_valid_index = dataframe[col].last_valid_index()

        if isinstance(last_valid_index, str):
            last_valid_index = pd.Timestamp(last_valid_index)

        shift = (target_date - last_valid_index).days
        if (target_date - last_valid_index).days > 0:
            shifted_dataframe[col] = dataframe[col].shift(shift)

    return shifted_dataframe


def extrapolate_with_noise(df, freq="daily"):
    if freq == "daily":
        step = pd.DateOffset(days=1)
    elif freq == "monthly":
        step = pd.DateOffset(months=1)
    elif freq == "decade":
        step = pd.DateOffset(days=10)

    df_new = df.copy()
    for col in df.columns:
        df_col = df_new[col].copy()
        scale = np.std(df_col)*0.01

        lvi = df_col.last_valid_index()
        li = df_col.index[-1]

        x = float(df_col[lvi])
        # m = x-float(df_col.iloc[lvi-1])

        for i in pd.date_range(start=lvi + step, end=li, freq=step):
            df_col[i.date()] = x + scale*(np.random.random()-0.5)*2

        df_col = df_col.interpolate()
        df_new[col] = list(df_col)

    return df_new


def shuffle_io(io_data: tuple) -> tuple:
    l = io_data[0].shape[0]
    r = np.arange(l)

    new_input = io_data[0][r]
    new_output = io_data[1][r]

    new_io_data = (new_input, new_output)

    return new_io_data


def smooth_past_data(data, delta_t):
    new_data = data.copy()

    for t in range(len(data)):
        if t >= delta_t:
            new_data[t] = np.nanmean(data[t - delta_t: t+1])
        else:
            new_data[t] = np.nanmean(data[0:t+1])
    #
    return new_data


def multi_to_single(columns: pd.MultiIndex) -> pd.Index:
    """
    Creates a single index from a two-level multiindex by combining the levels to a string of the form 'level0-level1'.
    """
    columns_new = []
    for col in list(columns):
        columns_new.append(str(col[0]) + "-" + str(col[1]))
    return columns_new


def rmse(v1, v2):
    sqrt = np.sqrt(np.mean((v1 - v2) ** 2))
    return sqrt


def merge_predictions_and_rtm(country: str, preds: pd.DataFrame):
    """
    Merge data and Predictions
    Args:
        country: Name of the country
        preds: file containing the predictions ( use function forecast)
        forecast_window: the length of the forecasts
        show: bollean to show the comparison between data and predictions
    Returns:
    """
    preds['adm1_code'] = preds['adm1_code'].astype(int)
    data = pd.read_csv(f"data/{country}/full_timeseries_daily.csv", header=[0, 1], index_col=0)
    data.index.name = 'date'
    data.index = pd.to_datetime(data.index)
    # the algorithms work with a smoothing of 10 days
    fcs = data['FCS'].rolling('10D').mean()
    fcs = fcs.reset_index().melt(id_vars='date', value_name='data', var_name='adm1_code')
    fcs['adm1_code'] = fcs['adm1_code'].astype(int)

    fcs = fcs.merge(preds, on=['date', 'adm1_code'], how='left')
    fcs = fcs[~fcs.prediction.isnull()]
    return fcs


def my_diff(data, col):
    return data[col].iloc[-1] - data[col].iloc[0]


def all_performances(model, country):
    path = 'forecasts/'+model+'/'
    performance_list = []
    target_diff_list = []
    prediction_diff_list = []

    for file in os.listdir(path):
        if country in file:
            dfn = pd.read_csv(path + file)
            split_date = file.split('_')[1].replace('.csv', '')
            dfn = dfn.sort_values(by=['adm1_code', 'date'])

            perf = dfn.groupby('adm1_code').apply(lambda d: rmse(d['data'], d['prediction'])).reset_index(name='rmse')
            delta_t = dfn.groupby('adm1_code').apply(lambda d: my_diff(d, 'data')).reset_index(name='delta_data')
            delta_p = dfn.groupby('adm1_code').apply(lambda d: my_diff(d, 'prediction')).reset_index(name='delta_pred')

            # Add date column to each DataFrame
            for df in (perf, delta_t, delta_p):
                df['date'] = split_date

            # Append results to lists
            performance_list.append(perf)
            target_diff_list.append(delta_t)
            prediction_diff_list.append(delta_p)

    # Concatenate all DataFrames in each list
    performance = pd.concat(performance_list, ignore_index=True)
    target_diff = pd.concat(target_diff_list, ignore_index=True)
    prediction_diff = pd.concat(prediction_diff_list, ignore_index=True)

    res = performance.merge(target_diff, on=['adm1_code', 'date']).merge(prediction_diff, on=['adm1_code', 'date'])
    res['model'] = model
    res['country'] = country
    return res


def categorise_curves(dfres, column, step, n_steps):
    """
    Caterogizes curves using the difference between the initial and final value.
    Args:
        dfres: data frame containing the data ( output from all_performances)
        column: data or prediction, depending on what needs to be caterorised
        step: step to define a bin
        n_steps: number of steps( will be simmetrical around zero.
    Returns:

    """
    data = dfres.copy()
    col_name = column + "_cat"
    data[col_name] = 100

    bool0 = data[column] <= -(n_steps * step)
    data.loc[bool0, col_name] = -n_steps
    # Negative Values (improvement)
    for n in np.arange(n_steps, 0, -1):
        bool_var = (data[column] >= -n * step) * (data[column] < (n - 1) * step)
        data.loc[bool_var, col_name] = -n + 1
    # Positive Values (Deterioration)
    for n in np.arange(0, n_steps):
        bool_var = (data[column] >= n * step) * (data[column] < (n + 1) * step)
        data.loc[bool_var, col_name] = n + 1
    bool0 = data[column] >= n_steps * step
    data.loc[bool0, col_name] = n_steps + 1
    return data


feature_dict = {"FCS": ["FCS"],
                "FCS+": ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"],
                "calendar": ["FCS", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"],
                "climate": ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality",
                             "rainfall", "NDVI", "log rainfall 1 month anomaly", "log rainfall 3 months anomaly",
                             "log NDVI anomaly"],
                'economics': ["FCS", "rCSI", "Ramadan", "day of the year",
                             "CE official", "CE unofficial","PEWI", "headline inflation", "food inflation"],
                "all":["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality",
                         "rainfall", "NDVI", "log rainfall 1 month anomaly", "log rainfall 3 months anomaly",
                         "log NDVI anomaly", "CE official", "CE unofficial","PEWI", "headline inflation",
                       "food inflation"]}
