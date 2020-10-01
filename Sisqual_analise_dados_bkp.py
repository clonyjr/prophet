#%%

from tools import datacleaner
import config
import os
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
# Offline mode
from plotly.offline import init_notebook_mode, iplot

# "high resolution"
%config InlineBackend.figure_format = 'retina'
init_notebook_mode(connected=True)

#%%

## Cria 3 dataframes:
## df_274_time_sale = quantidade de produtos vendidos na loja por hora
## df_274_time_cli = quantidade de clientes na loja por hora

#df_complete = datacleaner.get_Dataframes(None,None,full_dataframe=True)
#df_complete.reset_index(inplace=True)

df_274_time_sale = datacleaner.get_Dataframes_time(274, 's')
df_274_time_cli = datacleaner.get_Dataframes_time(274, 'c')
df_274_time_sale.reset_index(inplace=True)
df_274_time_cli.reset_index(inplace=True)

## Cria dois dataframes:
## df_274_sale = quantidade de produtos vendidos na loja por data
## df_274_cli  = quantidade de clientes na loja por data

df_274_sale = datacleaner.get_Dataframes(274, 's')
df_274_sale.reset_index(inplace=True)

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

#%% md

# Quantidade de produtos vendidos/hora (loja 274)

#%%

sales_data_l274 = go.Scatter(x=df_274_time_sale.ds,y=df_274_time_sale.y)
layout = go.Layout(height=800,
                   width=1000,
                   title='Vendas/hora (março-maio de 2020)')
fig_sales_hour = go.Figure(data=[sales_data_l274],layout=layout)
iplot(fig_sales_hour)

#%% md

`Analisando o gráfico acima, observamos visualmente uma sazonalidade pelos picos de alto e baixo no gráfico.`


#%% md

# Quantidade de clientes/hora (loja 274)

#%%

cli_data_l274 = go.Scatter(x=df_274_time_cli.ds,y=df_274_time_cli.y)
layout = go.Layout(height=800,
                   width=1000,
                   title='Vendas ao dia (Loja 274, de março à maio de 2020)')
fig_cli = go.Figure(data=[cli_data_l274],layout=layout)
iplot(fig_cli)

#%% md

`O gráfico de quantidade de clientes por hora na loja 274 apresenta semelhanças com o gráfico de quantidade de produtos por hora na mesma loja`
`logo, inferimos que o comportamento da série é sazonal como a série anterior.`

#%% md

# Loja 432

#%%

df_432_time_sale = datacleaner.get_Dataframes_time(432, 's')
df_432_time_cli = datacleaner.get_Dataframes_time(432, 'c')
df_432_time_sale.reset_index(inplace=True)
df_432_time_cli.reset_index(inplace=True)


#%%

sales_data_l432 = go.Scatter(x=df_432_time_sale.ds,y=df_432_time_sale.y)
layout = go.Layout(height=800,
                   width=1000,
                   title='Vendas/hora (março-maio de 2020)')
fig_sales_hour_432 = go.Figure(data=[sales_data_l432],layout=layout)
iplot(fig_sales_hour_432)

#%% md

`Analisando o gráfico acima, também observamos visualmente o mesmo tipo de sazonalidade que os gráficos anteriores.`
`Além da sazonalidade também observamos alguns dados negativos.`

#%%

df_432_time_sale.loc[df_432_time_sale['y'] < 0]

#%% md

1. Em 20-03 houve -24 produtos vendidos e -1 cliente na loja
2. Em 02-04 houve -8 produtos vendidos e -1 cliente na loja
3. Em 23-04 houve -2 produtos vendidos e 0 cliente na loja
4. Em 20-05 houve -3 produtos vendidos e -1 cliente na loja
5. Em 21-05 houve -6 produtos vendidos e -1 cliente na loja
6. Em 26-05 houve -16 produtos vendidos e 1 cliente na loja

`Nesse caso notamos que esse comportamento ocorre sempre após as 21 horas, aparentemente próximo do horário de encerramento do atendimento. Resta a dúvida sobre qual motivo leva os dados a terem esses valores`
`Verificamos que esse comportamento não se repete na loja 274.`

#%%

df_274_time_sale.loc[df_274_time_sale['y'] < 0]


#%% md

`Removendo os valores negativos do dataframe de vendas da loja 432`

#%%

df_432_time_sale_val_positivo = df_432_time_sale.copy()
df_432_time_sale_val_positivo = df_432_time_sale_val_positivo.loc[df_432_time_sale_val_positivo['y'] > 0]
df_432_time_sale_val_positivo


#%% md

# Quantidade de clientes/dia (loja 432)

#%%

cli_data_l432 = go.Scatter(x=df_432_time_cli.ds,y=df_432_time_cli.y)
layout = go.Layout(height=800,
                   width=1000,
                   title='Vendas/hora (Loja 432, de março à maio de 2020)')
fig_cli_432 = go.Figure(data=[cli_data_l432],layout=layout)
iplot(fig_cli_432)

#%% md

`O gráfico de quantidade de clientes por hora na loja 432 apresenta semelhanças com o gráfico de quantidade de produtos por hora na mesma loja. Também notamos nessa loja momentos em que o número de clientes foi negativo:`

#%%

df_432_time_cli.loc[df_432_time_cli['y'] < 0]

#%%

df_432_time_cli_val_positivo = df_432_time_cli.copy()
df_432_time_cli_val_positivo=df_432_time_cli_val_positivo.loc[df_432_time_cli_val_positivo['y'] > 0]


#%% md

<div class="alert alert-block alert-info">
<p> Na linha acima removemos os valores negativos do dataframe.</p>
<b> Dúvida: </b> Os valores negativos verificados provocam as seguintes dúvidas:
 <p>- Há necessidade de normalizar os dados por esse motivo?</p>
 <p>- Não deveríamos questionar à Sisqual se sabem sobre essa situação e porque ocorre dessa forma? </p>
</div>


#%% md

# Seasonality

#%% md

O Prophet, para realizar a previsão, utiliza dois modelos Aditivo e Multiplicativo e podem ser determinados pelos 
parâmetros.
O modelo aditivo sugere que os componentes são adicionados conforme a equação abaixo:

$ y(t) = Level + Trend + Seasonality + Noise $

O modelo multiplicativo sugere que os componentes são multiplicados:

$ y(t) = Level + Trend + Seasonality + Noise $

Onde:

+ **Level** = Valores médios da série.
+ **Trend** = Aumento ou a redução dos valores da série.
+ **Seasonality** = Ciclo de repetições da série.
+ **Noise** = Variações randômicas da série.

O Prophet decompõe o seu modelo em trend, seasonality (semanal, mensal, diário) e holidays (feriados).

#%% md

`Para isso é preciso primeiro criar o modelo no Prophet. Fazemos isso preenchendo os seguintes parâmetros:`

+ Parâmetro growth (saturação)
> Quando a previsão cresce, alguns pontos atingem o valor máximo possível, como um teto,
isso é chamado de carrying capacity. Por essa razão deve-se saturar o "growth" da previsão para que o modelo comporte essas
variações.
> É possível definir o carrying capacity (cap) adicionando esse valor como uma coluna do dataframe.

`Por padrão o Prophet usa um modelo de saturação linear. Utilizaremos nesse caso o modelo logistico.`

+ Parâmetro interval_width

> O interval_width de confiança = 95%; isso define o intervalo de incerteza para produzir um intervalo
de confiança em torno do valor previsto.

+ Parâmetros Trend Changepoints

> Para os dados que estamos a analisar, os *changepoints* representam a data, momento ou índice de tempo que define
um ponto quando uma data começa a mudar sua direção, quer seja crescente ou decrescente, seria uma espécie de ponto de
inflexão.
> Prophet usa um entre dois métodos para definir a tendência de *changepoints*
    > Especificar a flexibilidade das tendências dos *changepoints*
    > Especificar o local dos *changepoints*, significa definir os *changepoints* no próprio dataframe como uma série
    (quando a tendência começa a mudar).
`No nosso caso vamos utilizar a flexibilidade de tendência, ou seja o modelo será flexível a tendência.`

+ Parâmetro changepoint_prior_scale
> Representa o quão flexível o modelo irá se comportar contra os *trend changepoints*

#%% md

+ Parâmetros de sazonalidade

> Definem o período de sazonalidade a ser obtido pelo prophet. Em nosso caso desejamos que o algoritmo considere
sazonalidades (a frequência de repetição de um evento) anual e mensal.

+ Parâmetro holiday
> Os feriados e eventos influenciam no comportamento de compra e deslocamento das pessoas. Para esse estudo foram
adicionados os feriados nacionais conhecidos, entretanto, cabe destacar que podem existir eventos ou feriados municipais
que impactam o comércio local onde as lojas se encontram.

#%%

# definindo o cap (carrying capacity)
df_274_time_sale['cap'] = 1362
df_274_time_sale.reset_index(drop=True)

#%% md

`O valor cap gerado teve por base o máximo valor atingido no período da série`
![Cap Vendas](img/cap_vendas.png)

#%% md

## Criação do dataframe para futuro
+ Identificando os valores criados fora do range de atendimento da loja
+ Frequência horária
+ Cada 30 minutos de cada hora
+ Carry Capacity (teto) = 1362 (valor carece de validação)
    - Carry Capacity (base) = Não utilizado

#%%

m_274_time_sale = Prophet(growth='logistic',
                          interval_width=0.95,
                          changepoint_prior_scale=0.095,
                          yearly_seasonality=True,
                          weekly_seasonality=True,
                          holidays=datacleaner.get_Holiday())
m_274_time_sale.add_country_holidays(country_name='BR')
m_274_time_sale.fit(df_274_time_sale)

#%%

m_274_time_sale_linear = Prophet(growth='linear',
                                 interval_width=0.95,
                                 changepoint_prior_scale=0.095,
                                 yearly_seasonality=True,
                                 weekly_seasonality=True,
                                 holidays=datacleaner.get_Holiday())
m_274_time_sale_linear.add_country_holidays(country_name='BR')
m_274_time_sale_linear.fit(df_274_time_sale)


#%% md

`Criar um modelo com saturação aditiva para posterior comparação.`

#%%

m_274_time_sale.train_holiday_names

#%%

future_274_time_sale = m_274_time_sale.make_future_dataframe(periods=1800,freq='D')
future_274_time_sale['cap'] = 1362
teste_future_274_time_sale = future_274_time_sale.loc[future_274_time_sale.ds > '2020-06-01']
teste_future_274_time_sale

#%% md

`Criando o dataframe de previsão futura para uma frequência diária e um período de 5 anos`

#%%

#future_274_time_sale['ds'] = pd.to_datetime(future_274_time_sale['ds'])
#future_274_time_sale=future_274_time_sale.set_index(pd.DatetimeIndex(future_274_time_sale['ds']))
#future_274_time_sale.loc[future_274_time_sale['ds'].between_time('00:00','05:00')]

#%% md

`O prophet está projetando dados para horários em que a loja não possui dados, isto é, após 20:30 até 07:00, como não há dados o Prophet cria dados negativos que podem impactar nas análises. Dessa forma para evitar essa ocorrência foi necessário remover do dataframe, projetado no futuro, esses valores.`
`Para fazer isso há um tratamento nos dados importante, pois constatamos que provavelmente, não existe um horário fixo para a abertura e encerramento da loja`
`Existem dias em que a loja teve cliente e produtos vendidos a partir das 06:00 da manhã e dias em que o encerramento ocorreu as 20:30, 21:00, 21:30 e 22:00.`

#%%

df_sale_test = df_274_time_sale.copy()
df_sale_test['ds'] = pd.to_datetime(df_sale_test['ds'])
df_sale_test = df_sale_test.set_index(pd.DatetimeIndex(df_sale_test['ds']))
df_sale_test.loc[df_sale_test['ds'].between_time('06:00','07:00')]

#%%

df_sale_test.between_time('00:00','05:00')

#%%

future_274_time_sale_adjusted = future_274_time_sale.copy()
future_274_time_sale_adjusted['ds'] = pd.to_datetime(future_274_time_sale_adjusted['ds'])
future_274_time_sale_adjusted = future_274_time_sale_adjusted.set_index(pd.DatetimeIndex(future_274_time_sale_adjusted['ds']))
#future_274_time_sale_adjusted = future_274_time_sale_adjusted.between_time('06:00','22:00')
#future_274_time_sale_adjusted.reset_index(drop=True)
future_274_time_sale_adjusted.between_time('00:00','05:00')

#%%

future_274_time_sale_adjusted = future_274_time_sale.copy()
future_274_time_sale_adjusted['ds'] = pd.to_datetime(future_274_time_sale_adjusted['ds'])
future_274_time_sale_adjusted = future_274_time_sale_adjusted.set_index(pd.DatetimeIndex(future_274_time_sale_adjusted['ds']))
future_274_time_sale_adjusted = future_274_time_sale_adjusted.between_time('06:00','22:00')
#future_274_time_sale_adjusted.reset_index(drop=True)
future_274_time_sale_adjusted

#%% md

#### O dataframe projetado para 5 anos no futuro. O parâmetro uncertainty=True assume uma média da frequência e magnitude da tendência verificada no passado, que seja a mesma para o futuro.

#%%

forecast_time_sale = m_274_time_sale.predict(future_274_time_sale_adjusted)
fig_time_sale = m_274_time_sale.plot(forecast_time_sale, uncertainty=True,xlabel='Meses do ano',ylabel='Qtd Artigos Vendidos',figsize=(10,6))
plt.title('Previsão de Vendas Loja 274')

#%% md

#### Visualizando os trends changepoints anual de venda

#%%

fig = m_274_time_sale.plot(forecast_time_sale)
trend_change_points_anual = add_changepoints_to_plot(fig.gca(), m_274_time_sale, forecast_time_sale)

#%% md

#### Visualizando os componentes da série

#%%

fig_comp_time_sale = m_274_time_sale.plot_components(forecast_time_sale, uncertainty=True)

#%% md

###### Visualizando os dados previstos para o futuro

#%%

future_no_art_sales_pred = forecast_time_sale.loc[forecast_time_sale.ds > '2020-06-01']
future_no_art_sales_pred = future_no_art_sales_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_no_art_sales_pred.sample(5)

#%% md

## Previsão de artigos vendidos por hora com intervalo de 30 minutos para cada hora

#%%

future_274_time_sale = m_274_time_sale.make_future_dataframe(periods=1800,freq='H')
future_274_time_sale['cap'] = 1362
teste_future_274_time_sale = future_274_time_sale.loc[future_274_time_sale.ds > '2020-06-01']
teste_future_274_time_sale

#%%

df_sale_test = df_274_time_sale.copy()
df_sale_test['ds'] = pd.to_datetime(df_sale_test['ds'])
df_sale_test = df_sale_test.set_index(pd.DatetimeIndex(df_sale_test['ds']))
df_sale_test.loc[df_sale_test['ds'].between_time('06:00','07:00')]

#%%

df_sale_test.between_time('00:00','05:00')

#%%

future_274_time_sale_adjusted = future_274_time_sale.copy()
future_274_time_sale_adjusted['ds'] = pd.to_datetime(future_274_time_sale_adjusted['ds'])
future_274_time_sale_adjusted = future_274_time_sale_adjusted.set_index(pd.DatetimeIndex(future_274_time_sale_adjusted['ds']))
#future_274_time_sale_adjusted = future_274_time_sale_adjusted.between_time('06:00','22:00')
#future_274_time_sale_adjusted.reset_index(drop=True)
future_274_time_sale_adjusted.between_time('00:00','05:00')

#%%

future_274_time_sale_adjusted = future_274_time_sale.copy()
future_274_time_sale_adjusted['ds'] = pd.to_datetime(future_274_time_sale_adjusted['ds'])
future_274_time_sale_adjusted = future_274_time_sale_adjusted.set_index(pd.DatetimeIndex(future_274_time_sale_adjusted['ds']))
future_274_time_sale_adjusted = future_274_time_sale_adjusted.between_time('06:00','22:00')
#future_274_time_sale_adjusted.reset_index(drop=True)
future_274_time_sale_adjusted

#%%

forecast_time_sale = m_274_time_sale.predict(future_274_time_sale_adjusted)
fig_time_sale = m_274_time_sale.plot(forecast_time_sale, uncertainty=True,xlabel='Meses do ano',ylabel='Qtd Artigos Vendidos',figsize=(10,6))
plt.title('Previsão de Vendas Loja 274')

#%% md

#### Visualizando os trends changepoints de venda

#%%

figtrend_changepoints = m_274_time_sale.plot(forecast_time_sale)
trend_changepoints_diario = add_changepoints_to_plot(figtrend_changepoints.gca(), m_274_time_sale, forecast_time_sale)

#%% md

#### Visualizando os componentes da série

#%%

fig_comp_time_sale = m_274_time_sale.plot_components(forecast_time_sale, uncertainty=True)

#%% md

###### Visualizando os dados previstos para o futuro

#%%

future_no_art_sales_pred = forecast_time_sale.loc[forecast_time_sale.ds > '2020-06-01']
future_no_art_sales_pred = future_no_art_sales_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_no_art_sales_pred.sample(5)

#%% md

***
# Previsão de clientes/hora (loja 274)
***

#%%

df_274_time_cli['cap'] = 67
df_274_time_cli.reset_index(drop=True)
m_274_time_cli = Prophet(growth='logistic',
                         interval_width=0.95,
                         changepoint_prior_scale=0.095,
                         yearly_seasonality=True,
                         weekly_seasonality=True,
                         holidays=datacleaner.get_Holiday())
m_274_time_cli.add_country_holidays(country_name='BR')
m_274_time_cli.fit(df_274_time_cli)

#%% md

## Criando previsão diária para um período de 5 anos

#%%

future_274_time_cli = m_274_time_cli.make_future_dataframe(periods=1800, freq='D')
future_274_time_cli['cap'] = 67
teste_future_274_time_cli = future_274_time_cli.loc[future_274_time_cli.ds > '2020-06-01']
teste_future_274_time_cli

#%%

df_cli_test = df_274_time_cli.copy()
df_cli_test['ds'] = pd.to_datetime(df_cli_test['ds'])
df_cli_test = df_cli_test.set_index(pd.DatetimeIndex(df_cli_test['ds']))
df_cli_test.loc[df_cli_test['ds'].between_time('06:00','07:00')]

#%%

future_274_time_cli_adjusted = future_274_time_cli.copy()
future_274_time_cli_adjusted['ds'] = pd.to_datetime(future_274_time_cli_adjusted['ds'])
future_274_time_cli_adjusted = future_274_time_cli_adjusted.set_index(pd.DatetimeIndex(future_274_time_cli_adjusted['ds']))
future_274_time_cli_adjusted = future_274_time_cli_adjusted.between_time('06:00','22:00')
future_274_time_cli_adjusted

#%%

forecast_time_cli = m_274_time_cli.predict(future_274_time_cli_adjusted)
fig_time_cli = m_274_time_cli.plot(forecast_time_cli, uncertainty=True,xlabel='Meses do ano',ylabel='Qtd Clientes/hora',figsize=(10,6))
plt.title('Quantidade de clientes previstos na Loja 274')

#%% md

#### Visualizando os trends changepoints de venda

#%%

figtrend_changepoints_anual_cli = m_274_time_cli.plot(forecast_time_cli)
trend_changepoints_anual_cli = add_changepoints_to_plot(figtrend_changepoints_anual_cli.gca(), m_274_time_cli, forecast_time_cli)

#%% md

#### Visualizando os componentes da série

#%%

fig_comp_time_cli = m_274_time_cli.plot_components(forecast_time_cli, uncertainty=True)

#%% md

###### Visualizando os dados previstos para o futuro

#%%

future_no_cli_pred = forecast_time_cli.loc[forecast_time_cli.ds > '2020-06-01']
future_no_cli_pred = future_no_cli_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_no_cli_pred.sample(5)

#%% md

## Criando previsão com frequência horária e com intervalos de 30 minutos

#%%

future_274_time_cli = m_274_time_cli.make_future_dataframe(periods=1800, freq='H')
future_274_time_cli['cap'] = 67
teste_future_274_time_cli = future_274_time_cli.loc[future_274_time_cli.ds > '2020-06-01']
teste_future_274_time_cli

#%%

df_cli_test = df_274_time_cli.copy()
df_cli_test['ds'] = pd.to_datetime(df_cli_test['ds'])
df_cli_test = df_cli_test.set_index(pd.DatetimeIndex(df_cli_test['ds']))
df_cli_test.loc[df_cli_test['ds'].between_time('06:00','07:00')]

#%%

future_274_time_cli_adjusted = future_274_time_cli.copy()
future_274_time_cli_adjusted['ds'] = pd.to_datetime(future_274_time_cli_adjusted['ds'])
future_274_time_cli_adjusted = future_274_time_cli_adjusted.set_index(pd.DatetimeIndex(future_274_time_cli_adjusted['ds']))
future_274_time_cli_adjusted = future_274_time_cli_adjusted.between_time('06:00','22:00')
#future_274_time_cli_adjusted.reset_index(drop=True)
future_274_time_cli_adjusted

#%%

forecast_time_cli = m_274_time_cli.predict(future_274_time_cli_adjusted)
fig_time_cli = m_274_time_cli.plot(forecast_time_cli, uncertainty=True,xlabel='Meses do ano',ylabel='Qtd Clientes/hora',figsize=(10,6))
plt.title('Quantidade de clientes previstos na Loja 274')

#%% md

#### Visualizando os trends changepoints de venda

#%%

figtrend_changepoints_diario_cli = m_274_time_cli.plot(forecast_time_cli)
trend_changepoints_diario_cli = add_changepoints_to_plot(figtrend_changepoints_diario_cli.gca(), m_274_time_cli, forecast_time_cli)

#%% md

#### Visualizando os componentes da série

#%%

fig_comp_time_cli = m_274_time_cli.plot_components(forecast_time_cli, uncertainty=True)

#%% md

###### Visualizando os dados previstos para o futuro

#%%

future_no_cli_pred = forecast_time_cli.loc[forecast_time_cli.ds > '2020-06-01']
future_no_cli_pred = future_no_cli_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_no_cli_pred.sample(5)

#%% md

# Saturação linear para quantidade de produtos vendidos

#%%

future_274_time_sale_l = m_274_time_sale_linear.make_future_dataframe(periods=1800, freq='H')
future_274_time_sale_l['cap'] = 1362
teste_future_274_time_sale_l = future_274_time_sale_l.loc[future_274_time_sale_l.ds > '2020-06-01']
teste_future_274_time_sale_l


#%%

df_sale_test_a = df_274_time_sale.copy()
df_sale_test_a['ds'] = pd.to_datetime(df_sale_test_a['ds'])
df_sale_test_a = df_sale_test_a.set_index(pd.DatetimeIndex(df_sale_test_a['ds']))
df_sale_test_a.loc[df_sale_test_a['ds'].between_time('06:00','07:00')]

#%%

future_274_time_sale_adjusted_l = future_274_time_sale_l.copy()
future_274_time_sale_adjusted_l['ds'] = pd.to_datetime(future_274_time_sale_adjusted_l['ds'])
future_274_time_sale_adjusted_l = future_274_time_sale_adjusted_l.set_index(pd.DatetimeIndex(future_274_time_sale_adjusted_l ['ds']))
future_274_time_sale_adjusted_l = future_274_time_sale_adjusted_l.between_time('06:00','22:00')
#future_274_time_sale_adjusted.reset_index(drop=True)
future_274_time_sale_adjusted_l

#%%

forecast_time_sale_l = m_274_time_sale_linear.predict(future_274_time_sale_adjusted_l)
fig_time_sale_a = m_274_time_sale_linear.plot(forecast_time_sale_l, uncertainty=True,xlabel='Meses do ano',ylabel='Qtd Artigos Vendidos',figsize=(10,6))
plt.title('Previsão de Vendas Loja 274')

#%% md

#### Visualizando os trends changepoints de venda

#%%

figtrend_changepoints_diario_sale = m_274_time_sale_linear.plot(forecast_time_sale_l)
trend_changepoints_diario_sale = add_changepoints_to_plot(figtrend_changepoints_diario_sale.gca(), m_274_time_sale_linear, forecast_time_sale_l)

#%% md

#### Visualizando os componentes da série

#%%

fig_comp_time_sale_a = m_274_time_sale_linear.plot_components(forecast_time_sale_l, uncertainty=True)

#%% md

###### Visualizando os dados previstos para o futuro

#%%

future_no_art_sales_pred_l = forecast_time_sale_l.loc[forecast_time_sale_l.ds > '2020-06-01']
future_no_art_sales_pred_l = future_no_art_sales_pred_l[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_no_art_sales_pred_l.sample(5)

#%% md

# Saturação linear para quantidade de clientes/hora

#%%

m_274_time_cli_linear = Prophet(growth='linear',
                                interval_width=0.95,
                                changepoint_prior_scale=0.095,
                                yearly_seasonality=True,
                                weekly_seasonality=True,
                                holidays=datacleaner.get_Holiday())
m_274_time_cli_linear.add_country_holidays(country_name='BR')
m_274_time_cli_linear.fit(df_274_time_cli)
m_274_time_cli_linear.train_holiday_names

#%%

future_274_time_cli_l = m_274_time_cli_linear.make_future_dataframe(periods=1800, freq='H')
future_274_time_cli_l['cap'] = 67
teste_future_274_time_cli_l = future_274_time_cli_l.loc[future_274_time_cli_l.ds > '2020-06-01']
teste_future_274_time_cli_l

#%%

df_cli_test_a = df_274_time_cli.copy()
df_cli_test_a['ds'] = pd.to_datetime(df_cli_test_a['ds'])
df_cli_test_a = df_cli_test_a.set_index(pd.DatetimeIndex(df_cli_test_a['ds']))
df_cli_test_a.loc[df_cli_test_a['ds'].between_time('06:00','07:00')]

#%%

future_274_time_cli_adjusted_l = future_274_time_cli_l.copy()
future_274_time_cli_adjusted_l['ds'] = pd.to_datetime(future_274_time_cli_adjusted_l['ds'])
future_274_time_cli_adjusted_l = future_274_time_cli_adjusted_l.set_index(pd.DatetimeIndex(future_274_time_cli_adjusted_l['ds']))
future_274_time_cli_adjusted_l = future_274_time_cli_adjusted_l.between_time('06:00','22:00')
#future_274_time_cli_adjusted_l.reset_index(drop=True)
future_274_time_cli_adjusted_l

#%%

forecast_time_cli_l = m_274_time_cli_linear.predict(future_274_time_cli_adjusted_l)
fig_time_cli_a = m_274_time_cli_linear.plot(forecast_time_cli_l, uncertainty=True,xlabel='Meses do ano',ylabel='Qtd Clientes/hora',figsize=(10,6))
plt.title('Quantidade de clientes previstos na Loja 274')

#%%

fig_comp_time_cli_a = m_274_time_cli_linear.plot_components(forecast_time_cli_l, uncertainty=True)

#%% md

###### Visualizando os dados previstos para o futuro

#%%

future_no_cli_pred_l = forecast_time_cli_l.loc[forecast_time_cli_l.ds > '2020-06-01']
future_no_cli_pred_l = future_no_cli_pred_l[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_no_cli_pred_l.sample(5)

#%% md

`Analisando o gráfico sem o parâmetro de incerteza`

#%%

fig_comp_time_cli_a_uncertainty_false = m_274_time_cli_linear.plot_components(forecast_time_cli_l, uncertainty=False)

#%% md

## Visualizando a previsão para 5 anos

#%%

future_274_time_cli_l = m_274_time_cli_linear.make_future_dataframe(periods=1800, freq='D')
future_274_time_cli_l['cap'] = 67
teste_future_274_time_cli_l = future_274_time_cli_l.loc[future_274_time_cli_l.ds > '2020-06-01']
teste_future_274_time_cli_l

#%%

future_274_time_cli_adjusted_l = future_274_time_cli_l.copy()
future_274_time_cli_adjusted_l['ds'] = pd.to_datetime(future_274_time_cli_adjusted_l['ds'])
future_274_time_cli_adjusted_l = future_274_time_cli_adjusted_l.set_index(pd.DatetimeIndex(future_274_time_cli_adjusted_l['ds']))
future_274_time_cli_adjusted_l = future_274_time_cli_adjusted_l.between_time('06:00','22:00')
#future_274_time_cli_adjusted_l.reset_index(drop=True)
future_274_time_cli_adjusted_l

#%%

forecast_time_cli_l = m_274_time_cli_linear.predict(future_274_time_cli_adjusted_l)
fig_time_cli_a = m_274_time_cli_linear.plot(forecast_time_cli_l, uncertainty=True,xlabel='Meses do ano',ylabel='Qtd Clientes/hora',figsize=(10,6))
plt.title('Quantidade de clientes previstos na Loja 274')

#%%

fig_comp_time_cli_a = m_274_time_cli_linear.plot_components(forecast_time_cli_l, uncertainty=True)

#%% md

###### Visualizando os dados previstos para o futuro

#%%

future_no_cli_pred_l = forecast_time_cli_l.loc[forecast_time_cli_l.ds > '2020-06-01']
future_no_cli_pred_l = future_no_cli_pred_l[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_no_cli_pred_l.sample(5)

#%% md

`Analisando o gráfico sem o parâmetro de incerteza`

#%%

fig_comp_time_cli_a_uncertainty_false = m_274_time_cli_linear.plot_components(forecast_time_cli_l, uncertainty=False)

