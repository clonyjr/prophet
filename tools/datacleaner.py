# -*- coding: utf-8 -*-

import config
import pandas as pd
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def prep_new_dataset(
        store=None,
        variable=None,
        full_dataframe=False,
        sheet_name='Base',
        filename='/data/Venda_274_432.xlsx',
        cols=None):
    path = getattr(config,'path','default_value')
    cli_data_file = path+filename
    raw_data =pd.read_excel(cli_data_file,
                            sheet_name=sheet_name,
                            header=0,
                            index_col=None,
                            parse_dates={'period':[1,2,3]},
                            keep_default_na=True
                            )
    pd.to_datetime(raw_data['period'])
    raw_data.astype({'Dia Semana': 'str'}).dtypes
    raw_data.head()

    portuguese_week_day = {'Segunda-Feira':0
        ,'Segunda-feira':0
        ,'Terça-feira':1
        ,'Quarta-feira':2
        ,'Quinta-feira':3
        ,'Sexta-feira':4
        ,'Sábado':5
        ,'Domingo':6
                           }
    raw_data.replace({'Dia Semana':portuguese_week_day}, inplace=True)
    df_operate = raw_data.drop('base', axis=1)
    df_operate['Dia Semana'] = pd.to_numeric(df_operate['Dia Semana'])
    df_operate = df_operate.rename(columns={'period':'ds','[CLIENTES QUE HAN COMPRADO]':'y'},inplace=True)
    #df_operate.to_csv('sisqual_dataset.csv')
    raw_data = raw_data.rename(columns={'[Codigo de Tienda]': 'loja', '[TIPO MEDIA HORA]':'tp_meia_hora',
                                        '[IMPORTE EFECTIVO]':'imposto','[NUM, ART, EN LA MEDIA HORA]': 'no_art_vend',
                                        '[CLIENTES QUE HAN COMPRADO]':'no_cli_comprou'})
    # Creating dataframe for store 274 and for store 432
    df_274 = raw_data.query('loja == "274"')
    df_274.reset_index(drop=True)
    df_432 = raw_data.query('loja == "432"')
    df_432.reset_index(drop=True)

    # Reading grouped data of store 274 and 432 (separating y as number of sales and number of clients)
    df_274_sales = df_274.rename(columns={'period':'ds','no_art_vend':'y'})
    df_432_sales = df_432.rename(columns={'period':'ds','no_art_vend':'y'})
    df_274_cli = df_274.rename(columns={'period':'ds','no_cli_comprou':'y'})
    df_432_cli = df_432.rename(columns={'period':'ds','no_cli_comprou':'y'})

    #df_274_sales = df_274_sales = df_274_sales[['ds','y']] ???
    df_274_sales = df_274_sales[['ds','y']]
    df_274_sales = df_274_sales.groupby('ds').sum()
    df_432_sales = df_432_sales[['ds','y']]
    df_432_sales = df_432_sales.groupby('ds').sum()
    df_274_cli = df_274_cli[['ds','y']]
    df_274_cli = df_274_cli.groupby('ds').sum()
    df_432_cli = df_432_cli[['ds','y']]
    df_432_cli = df_432_cli.groupby('ds').sum()
    if store == 274:
        if variable == 's': result = df_274_sales
        elif variable == 'c': result = df_274_cli
    if store == 432:
        if variable == 's': result = df_432_sales
        elif variable == 'c': result = df_432_cli
    if full_dataframe:
        result = raw_data

    return result

def get_Dataframes(store,variable,full_dataframe=False,sheet_name='Base',filename='/data/Venda_274_432.xlsx'):
    path = getattr(config,'path','default_value')
    cli_data_file = path+filename
    raw_data =pd.read_excel(cli_data_file,
    sheet_name=sheet_name,
    header=0,
    index_col=None,
    parse_dates={'period':[1,2,3]},
    keep_default_na=True
    )
    pd.to_datetime(raw_data['period'])
    raw_data.astype({'Dia Semana': 'str'}).dtypes
    raw_data.head()

    portuguese_week_day = {'Segunda-Feira':0
    ,'Segunda-feira':0
    ,'Terça-feira':1
    ,'Quarta-feira':2
    ,'Quinta-feira':3
    ,'Sexta-feira':4
    ,'Sábado':5
    ,'Domingo':6
    }
    raw_data.replace({'Dia Semana':portuguese_week_day}, inplace=True)
    df_operate = raw_data.drop('base', axis=1)
    df_operate['Dia Semana'] = pd.to_numeric(df_operate['Dia Semana'])
    df_operate = df_operate.rename(columns={'period':'ds','[CLIENTES QUE HAN COMPRADO]':'y'},inplace=True)
    #df_operate.to_csv('sisqual_dataset.csv')
    raw_data = raw_data.rename(columns={'[Codigo de Tienda]': 'loja', '[TIPO MEDIA HORA]':'tp_meia_hora',
    '[IMPORTE EFECTIVO]':'imposto','[NUM, ART, EN LA MEDIA HORA]': 'no_art_vend',
    '[CLIENTES QUE HAN COMPRADO]':'no_cli_comprou'})
    # Creating dataframe for store 274 and for store 432
    df_274 = raw_data.query('loja == "274"')
    df_274.reset_index(drop=True)
    df_432 = raw_data.query('loja == "432"')
    df_432.reset_index(drop=True)

    # Reading grouped data of store 274 and 432 (separating y as number of sales and number of clients)
    df_274_sales = df_274.rename(columns={'period':'ds','no_art_vend':'y'})
    df_432_sales = df_432.rename(columns={'period':'ds','no_art_vend':'y'})
    df_274_cli = df_274.rename(columns={'period':'ds','no_cli_comprou':'y'})
    df_432_cli = df_432.rename(columns={'period':'ds','no_cli_comprou':'y'})

    #df_274_sales = df_274_sales = df_274_sales[['ds','y']] ???
    df_274_sales = df_274_sales[['ds','y']]
    df_274_sales = df_274_sales.groupby('ds').sum()
    df_432_sales = df_432_sales[['ds','y']]
    df_432_sales = df_432_sales.groupby('ds').sum()
    df_274_cli = df_274_cli[['ds','y']]
    df_274_cli = df_274_cli.groupby('ds').sum()
    df_432_cli = df_432_cli[['ds','y']]
    df_432_cli = df_432_cli.groupby('ds').sum()
    if store == 274:
        if variable == 's': result = df_274_sales
        elif variable == 'c': result = df_274_cli
    if store == 432:
        if variable == 's': result = df_432_sales
        elif variable == 'c': result = df_432_cli
    if full_dataframe:
        result = raw_data
    
    return result

def get_Dataframes_time(store,variable,sheet_name='Base',filename='/data/Venda_274_432.xlsx'):
    path = getattr(config,'path','default_value')
    cli_data_file = path+filename
    raw_data =pd.read_excel(cli_data_file,
                            sheet_name=sheet_name,
                            header=0,
                            index_col=None,
                            parse_dates={'period':[1,2,3,9]},
                            keep_default_na=True
                            )
    pd.to_datetime(raw_data['period'])
    raw_data.astype({'Dia Semana': 'str'}).dtypes
    raw_data.head()

    portuguese_week_day = {'Segunda-Feira':0
        ,'Segunda-feira':0
        ,'Terça-feira':1
        ,'Quarta-feira':2
        ,'Quinta-feira':3
        ,'Sexta-feira':4
        ,'Sábado':5
        ,'Domingo':6
                           }
    raw_data.replace({'Dia Semana':portuguese_week_day}, inplace=True)
    df_operate = raw_data.drop('base', axis=1)
    df_operate['Dia Semana'] = pd.to_numeric(df_operate['Dia Semana'])
    df_operate = df_operate.rename(columns={'period':'ds','[CLIENTES QUE HAN COMPRADO]':'y'},inplace=True)
    raw_data = raw_data.rename(columns={'[Codigo de Tienda]': 'loja', '[TIPO MEDIA HORA]':'tp_meia_hora',
                                        '[IMPORTE EFECTIVO]':'imposto','[NUM, ART, EN LA MEDIA HORA]': 'no_art_vend',
                                        '[CLIENTES QUE HAN COMPRADO]':'no_cli_comprou'})
    # Creating dataframe for store 274 and for store 432
    df_274 = raw_data.query('loja == "274"')
    df_274.reset_index(drop=True)
    df_432 = raw_data.query('loja == "432"')
    df_432.reset_index(drop=True)

    # Reading grouped data of store 274 and 432 (separating y as number of sales and number of clients)
    df_274_sales = df_274.rename(columns={'period':'ds','no_art_vend':'y'})
    df_432_sales = df_432.rename(columns={'period':'ds','no_art_vend':'y'})
    df_274_cli = df_274.rename(columns={'period':'ds','no_cli_comprou':'y'})
    df_432_cli = df_432.rename(columns={'period':'ds','no_cli_comprou':'y'})

    df_274_sales = df_274_sales[['ds','y']]
    df_274_sales = df_274_sales.groupby('ds').sum()
    df_432_sales = df_432_sales[['ds','y']]
    df_432_sales = df_432_sales.groupby('ds').sum()
    df_274_cli = df_274_cli[['ds','y']]
    df_274_cli = df_274_cli.groupby('ds').sum()
    df_432_cli = df_432_cli[['ds','y']]
    df_432_cli = df_432_cli.groupby('ds').sum()
    if store == 274:
        if variable == 's': result = df_274_sales
        elif variable == 'c': result = df_274_cli
    if store == 432:
        if variable == 's': result = df_432_sales
        elif variable == 'c': result = df_432_cli

    return result

def get_Dataframes(store,variable,full_dataframe=False,sheet_name='Base',filename='/data/Venda_274_432.xlsx'):
    path = getattr(config,'path','default_value')
    cli_data_file = path+filename
    raw_data =pd.read_excel(cli_data_file,
                            sheet_name=sheet_name,
                            header=0,
                            index_col=None,
                            parse_dates={'period':[1,2,3]},
                            keep_default_na=True
                            )
    pd.to_datetime(raw_data['period'])
    raw_data.astype({'Dia Semana': 'str'}).dtypes
    raw_data.head()

    portuguese_week_day = {'Segunda-Feira':0
        ,'Segunda-feira':0
        ,'Terça-feira':1
        ,'Quarta-feira':2
        ,'Quinta-feira':3
        ,'Sexta-feira':4
        ,'Sábado':5
        ,'Domingo':6
                           }
    raw_data.replace({'Dia Semana':portuguese_week_day}, inplace=True)
    df_operate = raw_data.drop('base', axis=1)
    df_operate['Dia Semana'] = pd.to_numeric(df_operate['Dia Semana'])
    df_operate = df_operate.rename(columns={'period':'ds','[CLIENTES QUE HAN COMPRADO]':'y'},inplace=True)
    #df_operate.to_csv('sisqual_dataset.csv')
    raw_data = raw_data.rename(columns={'[Codigo de Tienda]': 'loja', '[TIPO MEDIA HORA]':'tp_meia_hora',
                                        '[IMPORTE EFECTIVO]':'imposto','[NUM, ART, EN LA MEDIA HORA]': 'no_art_vend',
                                        '[CLIENTES QUE HAN COMPRADO]':'no_cli_comprou'})
    # Creating dataframe for store 274 and for store 432
    df_274 = raw_data.query('loja == "274"')
    df_274.reset_index(drop=True)
    df_432 = raw_data.query('loja == "432"')
    df_432.reset_index(drop=True)

    # Reading grouped data of store 274 and 432 (separating y as number of sales and number of clients)
    df_274_sales = df_274.rename(columns={'period':'ds','no_art_vend':'y'})
    df_432_sales = df_432.rename(columns={'period':'ds','no_art_vend':'y'})
    df_274_cli = df_274.rename(columns={'period':'ds','no_cli_comprou':'y'})
    df_432_cli = df_432.rename(columns={'period':'ds','no_cli_comprou':'y'})

    #df_274_sales = df_274_sales = df_274_sales[['ds','y']] ???
    df_274_sales = df_274_sales[['ds','y']]
    df_274_sales = df_274_sales.groupby('ds').sum()
    df_432_sales = df_432_sales[['ds','y']]
    df_432_sales = df_432_sales.groupby('ds').sum()
    df_274_cli = df_274_cli[['ds','y']]
    df_274_cli = df_274_cli.groupby('ds').sum()
    df_432_cli = df_432_cli[['ds','y']]
    df_432_cli = df_432_cli.groupby('ds').sum()
    if store == 274:
        if variable == 's': result = df_274_sales
        elif variable == 'c': result = df_274_cli
    if store == 432:
        if variable == 's': result = df_432_sales
        elif variable == 'c': result = df_432_cli
    if full_dataframe:
        result = raw_data

    return result

def get_Holiday():
    parana_feriados = pd.DataFrame({
        'holiday': 'parana',
        'ds': pd.to_datetime(['2020-01-01','2020-02-24','2020-02-25',
                              '2020-02-26','2020-04-10','2020-04-21',
                              '2020-05-01','2020-06-11','2020-09-07',
                              '2020-10-12','2020-10-15','2020-10-28',
                              '2020-11-02','2020-11-15','2020-12-25']),
    'lower_window': 0,
    'upper_window': 1,
})
    return parana_feriados

def format_time(dt):
    print(dt)
    if pd.isnull(dt):
        return "NaT"
    else:
        return datetime.strftime(dt, "%a <br> %H:%M %p")

'''Função que retorna um colormap para desenhar colunas/barras de gráficos com coloração mais intensa
para maior quantidade dos dados'''
def return_cmap(data):
    # Function to create a colormap
    v = data["y"].values
    colors = plt.cm.RdBu_r((v - v.min()) / (v.max() - v.min()))
    return colors

def get_negative_values(df, variavel):
    return np.sum((df[variavel] < 0).values.ravel())

def remove_negative_values(df):
    df['yhat'].loc[(df['yhat'] < 0)] = 0
    df['yhat_lower'].loc[(df['yhat_lower'] < 0)] = 0
    df['yhat_upper'].loc[(df['yhat_upper'] < 0)] = 0
    return df

def convert_date_time_values(df):
    df['ds'] = pd.to_datetime(
        df['period'].astype(str)+ " " +df['Hour'].astype(str)+ ":"+df['Minute'].astype(str),
        format="%Y-%m-%d %H:%M:%S")
    return df