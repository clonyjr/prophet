# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tools import datacleaner as dtclean
from evaluation import forecast_metrics as fm
import locale
# import scipy.optimize as optim
import random
from fbprophet import Prophet

# time series analysis

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

''' 
@author: Clony Abreu

this python script prepares the data to be explored by fbprophet. 
'''


def detect_outlier(data_1):
    """Função para identificar os dados outliers de um dataframe, baseado no z-score"""
    outliers = []
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)
    for y in data_1:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


def calculate_iqr_score(sorted_dataset):
    """Calcula o iqr_score de um dataset ordenado"""
    q1, q3 = np.percentile(sorted_dataset, [25, 75])
    iqr = q3 - q1
    limite_inferior = q1 - (1.5 * iqr)
    limite_superior = q3 + (1.5 * iqr)
    return limite_inferior, limite_superior


def hr_func(ts):
    """ Função que retorna a hora de um objeto timestamp."""
    return ts.hour


def func_logistic(t, a, b, c):
    """Função que define os coeficientes para estimar."""
    return c / (1 + a * np.exp(-b * t))


def find_daily_maxes(x, is_serie=False):
    """Retorna a medida maxima de cada dia e quando ocorreu em um dataframe. O objeto retornado é um dataframe"""
    if is_serie:
        x = x.copy().to_frame()
    else:
        x = x.groupby('y').max()
    result = pd.concat([x.groupby('y').max(),
                        x.groupby('y').idxmax()], axis=1).iloc[:, [0, 1]]
    result.columns = ['date', 'value']
    return result.set_index('date')


def calc_z_score(df):
    """Calcula o z_score do dataframe"""
    z = np.abs(stats.zscore(df))
    return z

def tuning_model(df_treino, df_original, periodo=None, frequencia=None, cap=None, floor=None, loja=None, tipo=None,
                 changepoint_prior_scale=[15,20,25],
                 seasonality_prior_scale=[15,20,25],
                 holidays_prior_scale=[15,20,25],
                 n_changepoints=[30,50,66]):
    """ Realiza testes com os parâmetros especificados para o modelo, a fim de identificar combinação que obtém
        o melhor mape"""
    nome_arquivo_csv = 'model_parameters_' + loja + '_' + tipo + '.csv'
    params_grid = {'seasonality_mode': ('multiplicative', 'additive'),
                   'changepoint_prior_scale': changepoint_prior_scale,
                   'seasonality_prior_scale': seasonality_prior_scale,
                   'holidays_prior_scale': holidays_prior_scale,
                   'n_changepoints': n_changepoints}
    grid = ParameterGrid(params_grid)
    strt = '2020-03-01 06:00:00'
    end = '2020-03-08 18:00:00'
    prophet_parameters = pd.DataFrame(columns=['MAPE', 'Parameters'])
    for p in grid:
        test = pd.DataFrame()
        print(p)
        random.seed(0)
        train_model = Prophet(changepoint_prior_scale=p['changepoint_prior_scale'],
                              holidays_prior_scale=p['holidays_prior_scale'],
                              n_changepoints=p['n_changepoints'],
                              seasonality_mode=p['seasonality_mode'],
                              weekly_seasonality=True,
                              daily_seasonality=True,
                              yearly_seasonality=True,
                              holidays=dtclean.get_Holiday(),
                              interval_width=0.95)
        train_model.add_country_holidays(country_name='BR')
        train_model.fit(df_treino)
        train_forecast = train_model.make_future_dataframe(periods=periodo, freq=frequencia, include_history=False)
        train_forecast = train_model.predict(train_forecast)
        test = train_forecast[['ds', 'yhat']]
        Actual = df_original[(df_original['ds'] > strt) & (df_original['ds'] <= end)]
        MAPE = fm.mape(Actual['y'], test['yhat'])
        print('Mean Absolute Percentage Error(MAPE)------------------------------------', MAPE)
        prophet_parameters = prophet_parameters.append({'MAPE': MAPE, 'Parameters': p}, ignore_index=True)
        prophet_parameters.to_csv(nome_arquivo_csv, sep='\t', encoding='utf-8')


def date_features(df):
    """Separa as features de data como colunas do dataframe."""
    locale.setlocale(locale.LC_ALL, "pt_PT")
    df = df.copy()
    df['mes'] = df['ds'].dt.strftime('%B')
    df['diasemana'] = df['ds'].dt.strftime('%A')
    df['num'] = df['ds'].dt.strftime('%w')
    df['semana'] = df['ds'].dt.strftime('%U')
    df['diames'] = df['ds'].dt.day
    df['hora'] = df['ds'].dt.hour
    df['minuto'] = df['ds'].dt.minute

    X = df[['mes', 'diames', 'semana', 'diasemana', 'num', 'hora', 'minuto']]
    y = df['y']
    df_analise = pd.concat([X, y], axis=1)
    return df_analise
    # else: return X


def configura_dataframe_treino_teste(df, inicio=None, fim=None):
    """Configura dos dataframes para treino e teste respetivamente retornando o dataframe de treino e o dataframe
    de teste"""
    data_final = '2020-05-23 21:00:00'

    try:
        if inicio is None and fim is None:
            mascara_treino = (df['ds'] <= data_final)
            mascara_teste = (df['ds'] >= data_final)
        else:
            mascara_treino = (df['ds'] >= inicio) & (df['ds'] <= fim)
            mascara_teste = (df['ds'] >= fim) & (df['ds'] <= data_final)
    except ValueError as err:
        print("Deve existir a definição de inicio e fim que seja válida:\n"+err.args)

    df_treino = df.loc[mascara_treino]
    df_teste = df.loc[mascara_teste]
    return df_treino, df_teste


def remove_outliers(df,outliers):
    df.loc[df['y'].isin(outliers)]

def normalize_dataframe(df):
    values = df.values
    values = values.reshape((len(values),1))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized