from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from utilities import rmse
import pandas as pd
import numpy as np
import os
from utilities import categorise_curves
from datetime import datetime, timedelta


def error_in_time():
    """
    Error in time
    Returns:
    """
    error_time = {}
    for m in ['LSTM', 'CNN', 'ARIMA', 'RC']:
        dftot = pd.DataFrame()
        for c in ['Mali', 'Syria', 'Yemen', 'Nigeria']:
            path = f'forecasts/{m}'
            for file in os.listdir(path):
                if c in file:
                    df = pd.read_csv(os.path.join(path, file))
                    dftot = pd.concat([dftot, df])
        error_time[m] = dftot.groupby(['forecast_step']).apply(
            lambda d: np.median(np.abs(d['data']-d['prediction']))).values
    f = px.line(pd.DataFrame(error_time),
                color_discrete_map={'RC': 'red',
                                    'LSTM': 'darkgreen',
                                    'CNN':  '#ffa500',
                                    'ARIMA': 'purple'})
    tvals = np.arange(0, 70, 5)
    print(tvals)
    f.update_layout(font_size=20,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    yaxis=dict(title='RMSE'),
                    xaxis=dict(title='Forecast Step (days)',
                               tickvals=tvals,
                               range=(0,60)),
                    width=600, height=500, legend=dict(
                        yanchor="bottom",
                        y=0.3,
                        xanchor="right",
                        x=0.99,
                        font_size=15)
                    )
    f.write_image('error_in_time.jpg')
    return f


def curve_distibution(dfres, n_steps, step):
    data = categorise_curves(dfres=dfres, column='delta_data', step=step, n_steps=n_steps)
    r = data.delta_data_cat.value_counts().reset_index()
    r['delta_data_cat'] /= 4
    f = px.bar(r, x='index', y='delta_data_cat', text='delta_data_cat', log_y=True)
    start = -(n_steps + 1)
    end = (n_steps + 2)

    tickvals = np.arange(start, end, 1) - 0.5

    text = []
    text.append('<'+str(-(n_steps + 1) * step * 100))
    for m in range(len(tickvals)):
        tick_val = -(n_steps + 1) * step + m * step
        tick_val = np.round(tick_val, 2)
        tick_val *= 100
        text.append(str(int(tick_val)))
    text.append(str(((n_steps + 1) * step) * 100)+'>')

    f.update_layout(xaxis=dict(title=r'$\Delta(\%) = \text{fcs}_{t=60} - \text{fcs}_{t=0}$',
                               tickvals=tickvals,
                               tickangle=0,
                               ticktext=text,
                               ),
                    font_size=17,
                    yaxis=dict(title='Number of Curves', title_font_size=20, range=(0, 2.5)),
                    width=930, height=500)
    f.write_image('curve_distibution.jpg')
    f.show()


def rmse_per_country():
    """
    Compute and plot  RMSE per country and model
    Returns:
    """
    dfres = pd.DataFrame()
    c_list = []
    m_list = []
    perf_list = []
    for c in ['Mali', 'Syria', 'Yemen', 'Nigeria']:
        for m in ['LSTM', 'CNN', 'ARIMA', 'RC']:
            data = pd.read_csv(f'data/{c}/full_timeseries_daily.csv', header=[0, 1], index_col=0)
            fcs = data['FCS']
            fcs = fcs.reset_index().melt(id_vars='index').dropna()
            fcs = fcs.rename(columns={"index": 'dates'})
            fcs['dates'] = pd.to_datetime(fcs['dates'])
            RMSE_NEW = []
            path = f'forecasts/{m}'
            for file in os.listdir(path):
                file_name = os.path.join(path, file)
                country = file.split('_')[0]
                if country == c:
                    df = pd.read_csv(file_name)
                    perf = df.groupby('adm1_code').apply(lambda d: rmse(d.data, d.prediction)).median()
                    RMSE_NEW.append(perf*100)
            perf_list.append(np.median(RMSE_NEW).round(1))
            c_list.append(c)
            m_list.append(m)
    dfres['country'] = c_list
    dfres['model'] = m_list
    dfres['rmse'] = perf_list

    f = px.bar(dfres, x='country', y='rmse', text='rmse',
               color='model',
               barmode='group', color_discrete_sequence=['darkgreen', '#ffa500', 'purple', 'red'])
    f.update_layout(font_size=18,
                    width=600,
                    height=500,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    yaxis=dict(title='RMSE'),
                    xaxis=dict(title=""),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        font_size=14)
                    )
    f.show()
    f.write_image('rmse_per_country.png')
    return


def rmse_per_category(dfres, n_steps, step):
    """
    Plots rmse for the curves that show deterioration
    Args:
        dfres: data with performances
        n_steps: number of steps
        step: step to define caterogies
    Returns:
    """
    dfres = categorise_curves(dfres=dfres, column='delta_data', step=step, n_steps=n_steps)

    cat = dfres.groupby(['delta_data_cat', 'model'])['rmse'].median().unstack(0)
    cat = cat.sort_values(by='model', ascending=False)
    cat = cat.reset_index().melt(id_vars='model')
    cat['value'] *= 100
    cat['value'] = cat['value'].round(2)

    cat = cat[cat["delta_data_cat"] > 1]
    tick_text = []
    for t in np.arange(1, n_steps):
        tick_text.append(r'$' + str(step * t) + '<\Delta<' + str(step * (t + 1)) + '$')

    tick_text.append(r'$\Delta>' + str(0.04 * n_steps) + '$')
    f = px.bar(cat, x='delta_data_cat', y='value', text='value', color='model', barmode='group'
               , color_discrete_sequence=['red', 'darkgreen', '#ffa500', 'purple'])
    f.update_layout(yaxis=dict(title='RMSE', tickfont_size=16, ),
                    xaxis=dict(title='',
                               tickvals=np.arange(2, 8),
                               ticktext=tick_text,
                               tickfont_size=16,), showlegend=False,
                    font=dict(size=22, color='black'),width=1000)
    f.write_image('rmse_per_category.jpg')
    f.show()


def curve_classification(dfres):
    """
    bar plot of the accuracy of the classification of curves in 3 classe: improvement, deterioration and
    no change.
    Args:
        dfres: data with performances
    Returns:
    """
    T = 0.04
    dfres['data_class'] = np.nan
    dfres.loc[dfres['delta_data'] >= T, 'data_class'] = 1
    dfres.loc[dfres['delta_data'] <= -T, 'data_class'] = -1
    dfres.loc[(dfres['delta_data'] > -T) & (dfres['delta_data'] < T), 'data_class'] = 0

    dfres['pred_class'] = np.nan
    dfres.loc[dfres['delta_pred'] >= T, 'pred_class'] = 1
    dfres.loc[dfres['delta_pred'] <= -T, 'pred_class'] = -1
    dfres.loc[(dfres['delta_pred'] > -T) & (dfres['delta_pred'] < T), 'pred_class'] = 0

    dfres.groupby('model').apply(lambda d: (d['data_class'] == d['pred_class']).mean())

    acc = []
    acc1 = []
    Tp = []
    Fp = []
    Fn = []
    Tn = []
    acc2 = []
    for m in ['RC', 'LSTM', 'CNN', 'ARIMA']:

        dfm = dfres[dfres.model == m]
        acc.append((dfm['data_class'] == dfm['pred_class']).mean())
        Tp.append(((dfm['data_class'] == dfm['pred_class'])*(dfm['pred_class'] == 1)).mean())
        Tn.append(((dfm['data_class'] == dfm['pred_class'])*(dfm['pred_class'] == 0)).mean())
        Fp.append(((dfm['data_class'] != dfm['pred_class'])*(dfm['pred_class'] == 1)).mean())
        Fn.append(((dfm['data_class'] != dfm['pred_class'])*(dfm['pred_class'] == 0)).mean())

        df1 = dfm[dfm['data_class'] == 1]
        acc1.append((df1['data_class'] == df1['pred_class']).mean())

        df2 = dfm[dfm['data_class'] == -1]
        acc2.append((df2['data_class'] == df2['pred_class']).mean())

    dfacc = pd.DataFrame()
    dfacc['models'] = ['RC', 'LSTM', 'CNN', 'ARIMA']
    dfacc['Det. Accuracy'] = acc1
    dfacc['Imp. Accuracy'] = acc2
    dfacc['Tot. Accuracy'] = acc
    dfacc['Det. Precision'] = np.array(Tp) / (np.array(Tp) +np.array(Fp))
    dfacc['Det. Recall'] = np.array(Tp) / (np.array(Tp) + np.array(Fn))
    dfacc = dfacc.melt(id_vars='models')
    dfacc = dfacc.sort_values(by=['variable'], ascending=True)
    print(dfacc)
    dfacc['value'] = dfacc['value'].round(2)

    dfacc = dfacc.sort_values(by=['variable'])

    f = px.bar(dfacc, x='variable', y='value', color='models', text='value',
               barmode='group',
               color_discrete_map={'RC':'red', 'LSTM':'darkgreen', 'CNN':'#ffa500', 'ARIMA':'purple'})
    f.update_layout(font_size=22, xaxis=dict(title=""), yaxis=dict(title='Accuracy'), width=1000)
    f.write_image('accuracy_classificaton.jpg')
    f.show()


def plot(data, country, ncols):
    admin1 = pd.read_csv("data/adm1_list.csv")
    admin1 = admin1[admin1.adm0_name == country]
    adm1_list = admin1['adm1_code'].to_list()
    adm1_name = admin1[['adm1_code', 'adm1_name']].set_index('adm1_code').to_dict()['adm1_name']

    nrows = int(np.ceil(len(adm1_list) / ncols))

    f = make_subplots(nrows,
                      ncols,
                      vertical_spacing=0.18,
                      subplot_titles=tuple([adm1_name[a1] for a1 in adm1_list])
                      )

    for num_plot, a in enumerate(adm1_list):
        df = data[data.adm1_code == a].copy()
        col = num_plot % ncols + 1
        row = num_plot // ncols + 1

        # prepare subplot: data, predictions and confidence interval
        f.add_trace(
            go.Scatter(name='data',
                       x=df['date'],
                       y=df['data'],
                       marker_color='blue',
                       showlegend=False),
            row=row, col=col
        )
        f.add_trace(
            go.Scatter(x=df['date'],
                       y=df['prediction'],
                       marker_color='red',
                       showlegend=True if num_plot == 0 else False,
                       ),
            row=row, col=col
        )

    f.update_layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=30,  # top margin
        ),
        height=nrows * 150, width=200 * ncols
    )
    return f


def plot_curves_per_category(dfres, cat, country):
    dfi = dfres[dfres['delta_data_cat'] == cat]
    dfi = dfi[dfi['country'] == country]
    dfi = dfi.drop_duplicates(subset=['adm1_code', 'delta_data_cat', 'date'])

    full_data = pd.read_csv(f'data/{country}/full_timeseries_daily.csv', header=[0, 1], index_col=0)
    full_data = full_data['FCS']
    full_data.index = pd.to_datetime(full_data.index)

    for ind, row in dfi.iterrows():
        adm1_code = row['adm1_code']
        date = datetime.strptime(row['date'], '%Y-%m-%d')

        data_to_plot = full_data[date - timedelta(days=100):date + timedelta(days=60)]
        data_to_plot = data_to_plot[str(adm1_code)]
        data_to_plot = data_to_plot.rolling(10).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_to_plot.index, y=data_to_plot.values, showlegend=False))

        for m in ['LSTM', 'CNN', 'RC', 'ARIMA']:
            if m == 'RC':
                pred_file_name = f"forecasts/{m}/{country}_{row['date']}_100.csv"

            else:
                pred_file_name = f"forecasts/{m}/{country}_{row['date']}.csv"
            pred = pd.read_csv(pred_file_name)
            pred = pred[pred['adm1_code'] == adm1_code]
            fig.add_trace(go.Scatter(x=pred['date'], y=pred['prediction'], name=m))

        fig.update_layout(title=country + ' - admin1 ' + str(adm1_code))
        fig.add_vline(date)
        fig.show()

    return