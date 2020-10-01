#%%

# Prevendo em periodos horários. horizon -> nº de períodos futuros a ser produzido
cv_results = cross_validation(prophet, initial='2000 hours', period='1440 minutes', horizon='1 day')
# Calculando MAPE
mape_baseline = dtexp.mean_absolute_percentage_error(cv_results.y,cv_results.yhat)
mape_baseline


#%%

from fbprophet.diagnostics import performance_metrics
performance_results = performance_metrics(cv_results)
performance_results.head()

#%%

from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(cv_results, metric='mape')

#%%

print(df_274_time_sale.iloc[1])
df_274_time_sale.reset_index(drop=True)
feriados = dtclean.get_Holiday()
df_teste = df_274_time_sale.iloc[0:28]


#%%

df_temp = forecast_feriados.copy()
df_temp = df_temp.iloc[0:28]
df_teste['yhat'] = df_temp['yhat']
df_teste.head()

#%%

import plotly.express as px

fig = px.bar(forecast_feriados.iloc[0:28],y='yhat', x='ds',text='yhat')
# Customize aspect
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Quantidade de produtos vendidos/hora em 01/03/2020')
fig.show()

#%%

import plotly.express as px
#
cmap = dtclean.return_cmap(df_274_time_sale.iloc[0:28])
fig = px.bar(df_274_time_sale.iloc[0:28],y='y', x='ds',text='y')
# Customize aspect
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Quantidade de produtos vendidos/hora em 01/03/2020')
fig.show()

#%%

my_range = range(1,len(df_teste.index)+1)
df_teste['hour']=df_teste['ds'].apply(dtexp.hr_func)
plt.figure(figsize=(10, 10), dpi=80)
plt.hlines(y=my_range, xmin=df_teste['y'], xmax=df_teste['yhat'], color='grey', alpha=0.4)
plt.scatter(df_teste['y'], my_range, color='skyblue', alpha=1, label='Valor original')
plt.scatter(df_teste['yhat'], my_range, color='green', alpha=0.4 , label='Valor previsto')
plt.legend()

# Add title and axis names
plt.yticks(my_range, df_teste['hour'])
plt.title("Comparação do valor original (y) com o valor previsto (yhat)", loc='left')
plt.xlabel('Valor das variáveis')
plt.ylabel('Data')

#%%

fig = go.Figure()
layout = go.Layout(height=800,
                   width=1000,
                   title='Comparativo entre valor original (y) e previsto (yhat) em 01/03/2020')
# Add traces
fig.add_trace(go.Scatter(x=df_teste['y'], y=df_teste['hour'],
                    mode='lines+markers',
                    name='original'))
fig.add_trace(go.Scatter(x=df_teste['yhat'].apply(np.ceil)
, y=df_teste['hour'],
                    mode='lines+markers',
                    name='previsto'))

fig.show()

#%% md

# Optimizando o modelo
`O parâmetro growth é utilizado com dois valores possíveis: linear -> utilizado quando não há previsão de saturação e a tendência mantém um crescimento relativamente constante, ou quando o expert do negócio informar; logistic -> utilizado quando há previsão de saturação, isto normalmente ocorre quando se conhece previamente a possibiilidade de saturação (limitação populacional ou econômica). No caso desse estudo utilizaremos growth='logistic'. Para esse parâmetro é necessário estabelecer o valor máximo (cap) que o dado poderá atingir e o mínimo` (floor). Utilizaremos 65% dos primeiros valores como o floor. O cap será de 2724 (cerca de 2 vezes o valor máximo alcançado no período dos dados reais)`

#%%

# definindo o cap (carrying capacity)
df_274_time_sale['cap'] = 1362*2
df_274_time_sale['floor'] = df_274_time_sale.iloc[0,1] * 0.65
df_274_time_sale.reset_index(drop=True)


#%%

prophet_optimized = Prophet(
                    growth='logistic',
                    interval_width=0.95,
                    holidays=feriados,
                    seasonality_mode="multiplicative",
                    changepoint_prior_scale=30,
                    seasonality_prior_scale=15,
                    holidays_prior_scale=20,
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    ).add_seasonality(
                        name='monthly',
                        period=30.5,
                        fourier_order=55
                    ).add_seasonality(
                        name='daily',
                        period=1,
                        fourier_order=15
                    ).add_seasonality(
                        name='weekly',
                        period=7,
                        fourier_order=20
                    )
prophet_optimized.add_country_holidays('BR')
# Treinando o modelo optimizado
prophet_optimized.fit(df_274_time_sale)

# Criando as datas futuras previstas pelo prophet optimizado
future_optimized = prophet_optimized.make_future_dataframe(freq='H', periods=30)
future_optimized['cap'] = 2724
future_optimized['floor'] = 1.3
# Removendo valores fora do range de atendimento da loja (22:30 - 05:59)
future_optimized['ds'] = pd.to_datetime(future_optimized['ds'])
future_optimized = future_optimized.set_index(pd.DatetimeIndex(future_optimized['ds']))
future_optimized = future_optimized.between_time('06:00','22:30')
# Previsão optimizada dos valores
forecast_optimized = prophet_optimized.predict(future_optimized)
# Prevendo em periodos horários
cv_results_optimized = cross_validation(prophet_optimized, initial='2000 hours', period='780 minutes', horizon='1 day')
# Calculando MAPE
mape_baseline_optimized = dtexp.mean_absolute_percentage_error(cv_results_optimized.y,cv_results_optimized.yhat)
mape_baseline_optimized

#%%

df_teste_opt = df_teste.copy()
df_temp_opt = forecast_optimized.copy()
df_temp_opt = df_temp_opt.iloc[0:28]
df_teste_opt['yhat'] = df_temp_opt['yhat']
df_teste_opt.head()

#%%

fig = px.bar(forecast_optimized.iloc[0:28],y='yhat', x='ds',text='yhat')
# Customize aspect
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Quantidade de produtos vendidos/hora em 01/03/2020')
fig.show()

#%%

my_range = range(1,len(df_teste_opt.index)+1)
df_teste_opt['hour']=df_teste['ds'].apply(dtexp.hr_func)
plt.figure(figsize=(10, 10), dpi=80)
plt.hlines(y=my_range, xmin=df_teste_opt['y'], xmax=df_teste_opt['yhat'], color='grey', alpha=0.4)
plt.scatter(df_teste_opt['y'], my_range, color='skyblue', alpha=1, label='Valor original')
plt.scatter(df_teste_opt['yhat'], my_range, color='green', alpha=0.4 , label='Valor previsto')
plt.legend()

# Add title and axis names
plt.yticks(my_range, df_teste_opt['hour'])
plt.title("Comparação do valor original (y) com o valor previsto (yhat)", loc='left')
plt.xlabel('Valor das variáveis')
plt.ylabel('Data')

#%%

fig = go.Figure()
layout = go.Layout(height=800,
                   width=1000,
                   title='Comparativo entre valor original (y) e previsto (yhat) em 01/03/2020')
# Add traces
fig.add_trace(go.Scatter(x=df_teste_opt['y'], y=df_teste_opt['hour'],
                    mode='lines+markers',
                    name='original'))
fig.add_trace(go.Scatter(x=df_teste_opt['yhat'].apply(np.ceil)
, y=df_teste['hour'],
                    mode='lines+markers',
                    name='previsto'))

fig.show()

#%%

fig1 = prophet_optimized.plot(forecast_optimized,xlabel='Meses do ano',ylabel='Qtd Artigos Vendidos',figsize=(15,6))

#%%

fig = go.Figure()
layout = go.Layout(height=800,
                   width=1000,
                   title='Comparativo entre valor original (y) e previsto (yhat) em 01/03/2020')
# Add traces
fig.add_trace(go.Scatter(x=df_teste['y'], y=df_teste['hour'],
                    mode='lines+markers',
                    name='original'))
fig.add_trace(go.Scatter(x=df_teste['yhat'].apply(np.ceil)
, y=df_teste['hour'],
                    mode='lines+markers',
                    name='previsto'))

fig.show()

#%% md

# Optimizando o modelo
`O parâmetro growth é utilizado com dois valores possíveis: linear -> utilizado quando não há previsão de saturação e a tendência mantém um crescimento relativamente constante, ou quando o expert do negócio informar; logistic -> utilizado quando há previsão de saturação, isto normalmente ocorre quando se conhece previamente a possibiilidade de saturação (limitação populacional ou econômica). No caso desse estudo utilizaremos growth='logistic'. Para esse parâmetro é necessário estabelecer o valor máximo (cap) que o dado poderá atingir e o mínimo` (floor). Utilizaremos 65% dos primeiros valores como o floor. O cap será de 2724 (cerca de 2 vezes o valor máximo alcançado no período dos dados reais)`

#%%

# definindo o cap (carrying capacity)
df_274_time_sale['cap'] = 1362*2
df_274_time_sale['floor'] = df_274_time_sale.iloc[0,1] * 0.65
df_274_time_sale.reset_index(drop=True)


#%%

prophet_optimized = Prophet(
                    growth='logistic',
                    interval_width=0.95,
                    holidays=feriados,
                    seasonality_mode="multiplicative",
                    changepoint_prior_scale=30,
                    seasonality_prior_scale=15,
                    holidays_prior_scale=20,
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    ).add_seasonality(
                        name='monthly',
                        period=30.5,
                        fourier_order=55
                    ).add_seasonality(
                        name='daily',
                        period=1,
                        fourier_order=15
                    ).add_seasonality(
                        name='weekly',
                        period=7,
                        fourier_order=20
                    )
prophet_optimized.add_country_holidays('BR')
# Treinando o modelo optimizado
prophet_optimized.fit(df_274_time_sale)

# Criando as datas futuras previstas pelo prophet optimizado
future_optimized = prophet_optimized.make_future_dataframe(freq='H', periods=30)
future_optimized['cap'] = 2724
future_optimized['floor'] = 1.3
# Removendo valores fora do range de atendimento da loja (22:30 - 05:59)
future_optimized['ds'] = pd.to_datetime(future_optimized['ds'])
future_optimized = future_optimized.set_index(pd.DatetimeIndex(future_optimized['ds']))
future_optimized = future_optimized.between_time('06:00','22:30')
# Previsão optimizada dos valores
forecast_optimized = prophet_optimized.predict(future_optimized)
# Prevendo em periodos horários
cv_results_optimized = cross_validation(prophet_optimized, initial='2000 hours', period='780 minutes', horizon='1 day')
# Calculando MAPE
mape_baseline_optimized = dtexp.mean_absolute_percentage_error(cv_results_optimized.y,cv_results_optimized.yhat)
mape_baseline_optimized

#%%

df_temp_opt = forecast_optimized.copy()
df_temp_opt = df_temp_opt.iloc[0:28]
df_teste_opt['yhat'] = df_temp_opt['yhat']
df_teste_opt.head()

#%%

fig = px.bar(forecast_optimized.iloc[0:28],y='yhat', x='ds',text='yhat')
# Customize aspect
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Quantidade de produtos vendidos/hora em 01/03/2020')
fig.show()

#%%

my_range = range(1,len(df_teste_opt.index)+1)
df_teste_opt['hour']=df_teste['ds'].apply(dtexp.hr_func)
plt.figure(figsize=(10, 10), dpi=80)
plt.hlines(y=my_range, xmin=df_teste_opt['y'], xmax=df_teste_opt['yhat'], color='grey', alpha=0.4)
plt.scatter(df_teste_opt['y'], my_range, color='skyblue', alpha=1, label='Valor original')
plt.scatter(df_teste_opt['yhat'], my_range, color='green', alpha=0.4 , label='Valor previsto')
plt.legend()

# Add title and axis names
plt.yticks(my_range, df_teste_opt['hour'])
plt.title("Comparação do valor original (y) com o valor previsto (yhat)", loc='left')
plt.xlabel('Valor das variáveis')
plt.ylabel('Data')

#%%

fig = go.Figure()
layout = go.Layout(height=800,
                   width=1000,
                   title='Comparativo entre valor original (y) e previsto (yhat) em 01/03/2020')
# Add traces
fig.add_trace(go.Scatter(x=df_teste_opt['y'], y=df_teste_opt['hour'],
                    mode='lines+markers',
                    name='original'))
fig.add_trace(go.Scatter(x=df_teste_opt['yhat'].apply(np.ceil)
, y=df_teste['hour'],
                    mode='lines+markers',
                    name='previsto'))

fig.show()

#%%

fig1 = prophet_optimized.plot(forecast_optimized,xlabel='Meses do ano',ylabel='Qtd Artigos Vendidos',figsize=(15,6))

#%%

# Figura dinâmica
py.init_notebook_mode()
fig = plot_plotly(prophet_optimized, forecast_optimized)
py.iplot(fig)

#%% md

#### Visualizando os trends changepoints de venda

#%%

figtrend_changepoints = prophet_optimized.plot(forecast_optimized)
trend_changepoints_diario = add_changepoints_to_plot(figtrend_changepoints.gca(), prophet_optimized, forecast_optimized)

#%% md

#### Visualizando os componentes da série

#%%

fig_comp_time_sale = prophet_optimized.plot_components(forecast_optimized, uncertainty=True)

#%%
