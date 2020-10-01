# -*- coding: utf-8 -*-

from tools import dataexplore as dtexp, datacleaner as dtclean


def main():
    df_274_time_sale = dtclean.get_Dataframes_time(274,'s')
    df_274_time_sale.reset_index(inplace=True)
    df_274_sale_treino, df_274_sale_teste = dtexp.configura_dataframe_treino_teste(df_274_time_sale)
    df_274_sale_treino['cap'] = 2724
    df_274_sale_treino['floor'] = 1.3
    dtexp.tuning_model(df_274_sale_treino,periodo=7,frequencia='D',cap=2724,floor=1.3)

if __name__ == "__main__":
    main()