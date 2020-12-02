ano_agregado = pd.DataFrame(df_387_new.groupby("ano")["y"].sum()).reset_index().sort_values('y')
mes_agregado = pd.DataFrame(df_387_new.groupby("mes")["y"].sum()).reset_index().sort_values('y')
semana_agregado = pd.DataFrame(df_387_new.groupby("semana")["y"].sum()).reset_index().sort_values('y')
dia_agregado = pd.DataFrame(df_387_new.groupby("num")["y"].sum()).reset_index().sort_values('num')
iplot(dtview.plot_dados_agregados_semana_mes(mes_agregado,tipo_agregacao='m',xaxes_title='Mês',yaxis_title='Total Vendas Realizadas',
                                      title='Vendas por Mês'))
iplot(dtview.plot_dados_agregados_semana_mes(semana_agregado,tipo_agregacao='s',xaxes_title='Nº da semana',yaxis_title='Total Vendas Realizadas',
                                      title='Vendas por Semana'))

iplot(dtview.plot_dados_agregados_semana_mes(dia_agregado,tipo_agregacao='d',xaxes_title='Nº do dia (0 = Domingo, 1 = Segunda ...)',yaxis_title='Total Vendas Realizadas',
                                      title='Vendas por dia'))

iplot(dtview.plot_dados_agregados_semana_mes(ano_agregado,tipo_agregacao='a',xaxes_title='Ano',yaxis_title='Total Vendas Realizadas',
                                      title='Vendas por Ano'))
                                      
                                      
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.plot(subplots=True, legend=False)

df_non_negative_test.set_index('ds',inplace=True)
df_groupby_year_test=df_non_negative_test.groupby(pd.Grouper(freq="y")),
df_groupby_year_test.head()



#%%



#%% md

<div class="alert alert-block alert-info">
<b>Realizando a previsão de 14 dias no futuro para o 3º dataset (anos 2019/2020)</b>
</div>

#%%

df_274_sale_treino_2020, df_274_sale_teste_2020
df_treino_2020=df_274_sale_treino_2020.groupby('ds').sum()
df_treino_2020.reset_index(level=0,inplace=True)
df_treino_2020.ds = pd.to_datetime(df_treino_2020.ds)
df_treino_2020.index = df_treino_2020.ds
df_treino_2020.y.plot(figsize=(10,6))
plt.show()


#%% md

# Identificando as Séries Temporais de Intervenção loja 274 anos 2019/2020

#%%

alpha = 0.98
mdl_prophet = Prophet(interval_width=alpha,holidays=dtclean.get_Holiday(years=[2019,2020]))
today_index = 30
print('Cutoff date: ', df_treino_2020.index[today_index])
mdl_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=2)
mdl_prophet.add_seasonality(name='weekly', period=7, fourier_order=2, prior_scale=0.5)
predict_n = 14
fig, forecast, mdl_prophet = prophet_fit(df_treino_2020, mdl_prophet, today_index, predict_days=predict_n)
outliers, df_pred = get_outliers(df_treino_2020, forecast, today_index, predict_days=predict_n)
prophet_plot(df_treino_2020, fig, today_index, predict_days=predict_n, outliers=outliers)


#%%

print('Previsões')
print(df_pred.head())
df_outlier = df_pred[(df_pred.actual > df_pred.upper_y) | (df_pred.actual < df_pred.lower_y)]
print('Outliers')
print(df_outlier.head())

#%%

holiday_2020=dtclean.get_Holiday(years=[2019,2020])
holiday_2020['day of week']=holiday_2020['ds'].dt.day_name()
holiday_2020.head(30)

df_274_sale_treino_2020, df_274_sale_teste_2020 = dtexp.configura_dataframe_treino_teste(
    df_store_274,
    inicio='2019-06-15 08:00:00',
    fim='2020-05-23 21:00:00',
    data_final='2020-05-31 21:00:00')
periods_2018=df_274_sale_teste_2018.shape[0]
periods_2019=df_274_sale_teste_2019.shape[0]
periods_2020=df_274_sale_teste_2020.shape[0]