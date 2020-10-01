import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def compare_dataframes_with_bar(df1,df2,nametrace1=None,nametrace2=None,idx_start=0,idx_end=28):
    trace1 = go.Bar(x=df1['ds'].iloc[idx_start:idx_end],y=df1['y'], name=nametrace1)
    trace2 = go.Bar(x=df2['ds'].iloc[idx_start:idx_end],y=df2['yhat'],name=nametrace2)
    data=[trace1,trace2]
    layout = go.Layout(barmode='group')
    fig=go.Figure(data=data,layout=layout)
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
    )
    return fig

def compare_dataframes_with_scatter(df_1, df_2, title=None, mode=None, name1=None,name2=None,is_forecast=False):
    trace_treino = go.Scatter(x=df_1.ds, y=df_1.y, mode='lines', name=name1)
    if is_forecast:
        trace_teste = go.Scatter(x=df_2.ds, y=df_2.yhat, mode='lines', name=name2)
    else: trace_teste = go.Scatter(x=df_2.ds, y=df_2.y, mode='lines', name=name2)
    data=[trace_treino,trace_teste]
    layout = go.Layout(height=500,
                       width=1000,
                       title={'text':title,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=data,layout=layout)
    return fig

def plot_forecast(forecast,model,f,ax):
    f,ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    fig = model.plot(forecast,ax)
    return plt,fig

def plot_dataframe_as_table(df):
    trace=go.Table(header=dict(values=list(df.columns)),
                   cells=dict(values=[df.ds,df.y]))

    data=[trace]
    fig=go.Figure(data=data)
    return fig

def plot_forecast_as_table(df):
    trace_df=go.Table(header=dict(values=list(df[['ds','yhat','yhat_lower','yhat_upper']])),
                      cells=dict(values=[df.ds,
                                         df.yhat,
                                         df.yhat_lower,
                                         df.yhat_upper]))
    data=[trace_df]
    fig_df=go.Figure(data=data)
    return fig_df

def plot_scatter(df, title=None):
    data=go.Scatter(x=df.ds,y=df.y)
    layout = go.Layout(height=500,
                       width=1000,
                       title={'text':title,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=[data],layout=layout)
    return fig

def plot_df_features(df,fig,ax):
    palette = sns.color_palette("mako_r", 4)
    a = sns.barplot(x="mes", y="y",hue = 'mes',data=df)
    a.set_title("274 Dados de Venda",fontsize=15)
    plt.legend(loc='upper right')
    plt.show()
    return plt

def plot_df_features_as_table(df):
    trace=go.Table(header=dict(values=list(df.columns)),
                   cells=dict(values=[df.mes,
                                      df.diames,
                                      df.semana,
                                      df.diasemana,
                                      df.num,
                                      df.hora,
                                      df.minuto,
                                      df.y]))
    data=[trace]
    fig=go.Figure(data=data)
    return fig

def check_outliers_univariate(df):
    '''Cria uma figura boxplot em plotly com os dados do dataframe e retorna uma figura'''
    trace1=go.Box(y=df['y'],
                  boxpoints='outliers',
                  marker_color='rgb(107,174,214)',
                  line_color='rgb(107,174,214)',
                  name='Produtos vendidos',
                  boxmean='sd')
    data=[trace1]
    fig=go.Figure(data)
    return fig

def check_outliers_multivariate(df):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(df['ds'], df['y'])
    ax.set_xlabel('Data e hora do registo')
    ax.set_ylabel('Quantidade')

    return fig

def view_optimum_parameters(nome_arquivo):
    parameters_df = pd.read_csv(nome_arquivo,sep='\t')
    parameters_df = parameters_df.sort_values(by=['MAPE'])
    parameters_df = parameters_df.reset_index(drop=True)
    parameters_df.drop(['Unnamed: 0'],axis=1,inplace=True)
    trace_df_parameters=go.Table(header=dict(values=list(parameters_df[['MAPE','Parameters']])),
                                 cells=dict(values=[parameters_df.MAPE,
                                                    parameters_df.Parameters]))
    data=[trace_df_parameters]
    fig_df_parameters=go.Figure(data=data)
    return fig_df_parameters

def plot_dados_agregados_semana_mes(
        df,
        tipo_agregacao=None,
        title=None,
        xaxes_title=None,
        yaxis_title=None,
        hover=None,
        is_client_df=False):
    if is_client_df:
        yhover=449
    else: yhover=5704
    if tipo_agregacao == 'm':
        agregacao = 'semana'
    else: agregacao = 'num'
    trace = go.Bar(x=df[agregacao],
                   y=df.y,
                   name=title,
                   marker_color=df.y)
    data=[trace]
    layout = go.Layout(barmode='group',
                       title={'text':title,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig=go.Figure(data=data,layout=layout)
    fig.update_xaxes(title_text=xaxes_title,showgrid=True,ticks="outside",tickson="boundaries",ticklen=20)
    fig.update_yaxes(title_text=yaxis_title)

    if hover is not None:
        fig.update_layout(
            showlegend=False,
            annotations=[
                dict(
                    x=22,
                    y=yhover,
                    xref="x",
                    yref="y",
                    text=hover,
                    showarrow=True,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                    ),
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=20,
                    ay=-30,
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=0.8
                )
            ]
        )
    return fig


def plot_results(x_values, y_values, title=None,xaxes_title=None,yaxis_title=None):
    trace_mape = go.Bar(x=x_values, y=y_values, name=title, marker_color=y_values)
    data_mape=[trace_mape]
    layout = go.Layout(barmode='group',
                       title={'text':title,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig_mape=go.Figure(data=data_mape,layout=layout)
    fig_mape.update_xaxes(title_text=xaxes_title,showgrid=True,ticks="outside",tickson="boundaries",ticklen=20)
    fig_mape.update_yaxes(title_text=yaxis_title)
    return fig_mape

def plot_total_dataframe_data(df1,
                              df2,
                              title=None,
                              title2=None,
                              labels_1=None,
                              labels_2=None,
                              annot_text_1=None,
                              annot_text_2=None):
    colors = ['orange', 'mediumturquoise']
    fig = go.Figure(data=[go.Pie(labels=labels_1,
                                 values=[df1['y'].count(),df2['y'].count()])])
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.update_layout(
        title_text=title,
        annotations=[dict(text=annot_text_1, x=0.12, y=1, font_size=20, showarrow=False),
                     dict(text=annot_text_2, x=0.87, y=1, font_size=20, showarrow=False)])

    fig.show()

    colors = ['lightturquoise', 'red']

    fig = go.Figure(data=[go.Pie(labels=labels_2,
                                 values=[np.sum((df1['y'] < 0).values.ravel()),
                                         np.sum((df2['y'] < 0).values.ravel())])])
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.update_layout(
        title_text=title2)

    return fig