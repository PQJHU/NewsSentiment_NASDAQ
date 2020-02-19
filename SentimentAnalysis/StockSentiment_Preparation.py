# -*- coding:utf-8 -*-

import pandas as pd
import re
import os
import datetime
import numpy as np
import sqlite3 as sq


def sort_df(df):
    df['_date'] = df.index
    df.sort_values(by=['symbol', '_date'], axis=0, ascending=True, inplace=True)
    df.drop(labels='_date', axis=1, inplace=True)
    return df


def buller(pos, neg, neut):
    total = pos + neg + neut
    bn = np.log((1 + pos / total) / (1 + neg / total))
    return bn / np.log(2)


def fetch_database(sql_com):
    conn = sq.connect(financial_data_path + 'lexica/nasdaqnews.db')
    cursor = conn.cursor()
    cursor.execute(sql_com)
    data = cursor.fetchall()
    return data


def stock_sentiment(df):
    df_grouped = df.groupby(by=[df.index, df['symbol']], axis=0).sum(axis=0)

    df_grouped['bl_bull'] = buller(pos=df_grouped['bl_pos'],
                                   neg=df_grouped['bl_neg'],
                                   neut=df_grouped['bl_neut'])
    df_grouped['lm_bull'] = buller(pos=df_grouped['lm_pos'],
                                   neg=df_grouped['lm_neg'],
                                   neut=df_grouped['lm_neut'])
    df_grouped['sm_bull'] = buller(pos=df_grouped['sm_pos'],
                                   neg=df_grouped['sm_neg'],
                                   neut=df_grouped['sm_neut'])

    df_grouped.ix[df_grouped['bl_bull'].isnull(), 'bl_bull'] = 0
    df_grouped.ix[df_grouped['lm_bull'].isnull(), 'lm_bull'] = 0
    df_grouped.ix[df_grouped['sm_bull'].isnull(), 'sm_bull'] = 0

    # negative sentiment
    df_grouped['bl_bull_neg'] = df_grouped['bl_bull']
    df_grouped.ix[df_grouped['bl_bull_neg'] > 0, 'bl_bull_neg'] = 0

    df_grouped['lm_bull_neg'] = df_grouped['lm_bull']
    df_grouped.ix[df_grouped['lm_bull_neg'] > 0, 'lm_bull_neg'] = 0

    df_grouped['sm_bull_neg'] = df_grouped['sm_bull']
    df_grouped.ix[df_grouped['sm_bull_neg'] > 0, 'sm_bull_neg'] = 0

    # positive sentiment
    df_grouped['bl_bull_pos'] = df_grouped['bl_bull']
    df_grouped.ix[df_grouped['bl_bull_pos'] < 0, 'bl_bull_pos'] = 0

    df_grouped['lm_bull_pos'] = df_grouped['lm_bull']
    df_grouped.ix[df_grouped['lm_bull_pos'] < 0, 'lm_bull_pos'] = 0

    df_grouped['sm_bull_pos'] = df_grouped['sm_bull']
    df_grouped.ix[df_grouped['sm_bull_pos'] < 0, 'sm_bull_pos'] = 0

    # df.to_csv(output_data_path + 'market_sentiment_raw_20170827.csv')

    # abs sentiment
    df_grouped['bl_bull_abs'] = np.abs(df_grouped['bl_bull'])
    df_grouped['lm_bull_abs'] = np.abs(df_grouped['lm_bull'])
    df_grouped['sm_bull_abs'] = np.abs(df_grouped['sm_bull'])

    # square sentiment
    df_grouped['bl_bull_square'] = np.square(df_grouped['bl_bull'])
    df_grouped['lm_bull_square'] = np.square(df_grouped['lm_bull'])
    df_grouped['sm_bull_square'] = np.square(df_grouped['sm_bull'])

    # prepare to output
    df_grouped.drop(labels=['bl_neg', 'bl_neut', 'bl_pos',
                            'lm_neg', 'lm_neut', 'lm_pos',
                            'sm_neg', 'sm_neut', 'sm_pos'],
                    axis=1, inplace=True)
    df_grouped.index.names = ['Date', 'symbol']

    # basic statistics
    basic_stat = df_grouped.describe()
    basic_stat.to_csv(output_data_path + 'StockSentimentStat.csv')

    # set panel to single index
    df_grouped.reset_index(level=[1], inplace=True)
    df_grouped.set_index(pd.to_datetime(df_grouped.index, format='%Y-%m-%d'), inplace=True)
    # take date after 2012
    df_grouped = df_grouped[df_grouped.index.year >= 2012]
    # merge date frame, leaves only trading day
    # df_grouped = df_grouped[df_grouped.index.isin(trading_date_frame)]

    return df_grouped


direct = '/Volumes/JeremyHu/DataBase/NasdaqSentiment/'
output_data_path = os.path.join(direct, 'output/csv/')
financial_data_path = os.path.join(direct, 'data/')
# trading_date_frame = pd.read_csv(financial_data_path+ 'financial/' + 'RawDataWithAttentionRatio_to_Matthias.csv')['date'].unique()
# trading_date_frame = [pd.to_datetime(dt, format='%Y/%m/%d') for dt in trading_date_frame]


def main():
    # execute processing if file doesn't exist
    sentiment_data_raw = pd.read_csv(output_data_path + '20170824_sentiment.csv', sep=';')
    articles_data = fetch_database(sql_com='Select linkid, date from articles;')
    articles_data_df = pd.DataFrame(articles_data, columns=['linkid', 'date'])

    symbol_data = fetch_database(sql_com='Select * from symbols;')
    symbol_data_df = pd.DataFrame(symbol_data, columns=['linkid', 'symbol'])

    # merge sentiment and article date
    sentiment_date = pd.merge(left=sentiment_data_raw, right=articles_data_df, on='linkid')
    # 1 row has no date info, drop it
    sentiment_date = sentiment_date[~ (sentiment_date['date'].isnull())]

    # sentiment_date = clean_df(sentiment_date, delta=time_interval)  # format date, round time, set date as index

    linkid_clean = [re.sub('-\w+\d{1,6}', '', title) for title in sentiment_date.linkid.values]
    sentiment_date.loc[:, 'linkid_clean'] = linkid_clean
    sentiment_date_uni = sentiment_date[~ (sentiment_date.duplicated(subset=['lm_neg', 'lm_neut', 'lm_pos',
                                                                             'sm_neg', 'sm_neut', 'sm_pos',
                                                                             'bl_neg', 'bl_neut', 'bl_pos',
                                                                             'linkid_clean', 'date'
                                                                             ]))]  # articles with no duplicate

    # ===duplicate check
    # linkid = sentiment_date.linkid.values
    # ran_int = np.random.randint(0, len(linkid))
    # linkid_clean[ran_int]
    # linkid[ran_int]

    # drop duplicated linkid rows
    # test_1 = sentiment_date.sort_values(by='linkid', axis=0, ascending=True)
    # test = sentiment_date[sentiment_date.duplicated()]
    # test.to_csv(output_data_path + 'dup_linkid_check_all.csv')
    # ===duplicate check end

    # merge sentiment, date, company ticker
    sentiment_date_symbol = pd.merge(left=sentiment_date_uni, right=symbol_data_df, on='linkid')

    # sentiment_date_uni.drop(labels='date', axis=1, inplace=True)
    sentiment_date_symbol.date = pd.to_datetime(sentiment_date_symbol.date, format='%Y-%m-%d %H:%M:%S')
    sentiment_date_symbol.set_index(keys='date', drop=True, inplace=True)
    sentiment_date_symbol.sort_index(axis=0, ascending=True, inplace=True)
    sentiment_date_symbol.drop(labels=['linkid', 'linkid_clean'], axis=1, inplace=True)
    # drop rows with symbol value is null
    sentiment_date_symbol = sentiment_date_symbol[~ (sentiment_date_symbol['symbol'].isnull())]
    # all 3 different ways to separate time
    # + 8 hous, to move time frame forward
    sentiment_date_symbol['8h_lag_date'] = [(dt + datetime.timedelta(hours=8)).date() for dt in
                                            sentiment_date_symbol.index]
    sentiment_date_symbol['8h_lag_time'] = [(dt + datetime.timedelta(hours=8)).time() for dt in
                                            sentiment_date_symbol.index]
    sentiment_date_symbol_overnight = sentiment_date_symbol[
        sentiment_date_symbol['8h_lag_time'] < datetime.time(hour=17, minute=30)]
    sentiment_date_symbol_trading = sentiment_date_symbol[
        sentiment_date_symbol['8h_lag_time'] >= datetime.time(hour=17, minute=30)]
    # sentiment_date_symbol_overnight.to_csv(output_data_path + 'time_test_overnight.csv')
    # sentiment_date_symbol_trading.to_csv(output_data_path + 'time_test_trading.csv')
    # sentiment_date_symbol.to_csv(output_data_path + 'test_time.csv')

    # daily sentiment
    sentiment_date_symbol_all = sentiment_date_symbol.set_index(keys='8h_lag_date', drop=True)
    sentiment_date_symbol_all.drop(labels=['8h_lag_time'], axis=1, inplace=True)
    # overnight sentiment
    sentiment_date_symbol_overnight = sentiment_date_symbol_overnight.set_index(keys='8h_lag_date', drop=True)
    sentiment_date_symbol_overnight.drop(labels=['8h_lag_time'], axis=1, inplace=True)
    # trading sentiment
    sentiment_date_symbol_trading = sentiment_date_symbol_trading.set_index(keys='8h_lag_date', drop=True)
    sentiment_date_symbol_trading.drop(labels=['8h_lag_time'], axis=1, inplace=True)

    # aggregate by date and symbol
    stock_sentiment_daily = stock_sentiment(df=sentiment_date_symbol_all)
    stock_sentiment_overnight = stock_sentiment(df=sentiment_date_symbol_overnight)
    stock_sentiment_trading = stock_sentiment(df=sentiment_date_symbol_trading)

    # save
    stock_sentiment_daily = sort_df(stock_sentiment_daily)
    stock_sentiment_daily.to_csv(output_data_path + 'StockSentimentDaily.csv')
    print("finished all")

    stock_sentiment_overnight = sort_df(stock_sentiment_overnight)
    stock_sentiment_overnight.to_csv(output_data_path + 'StockSentimentOvernight_Raw.csv')
    print("finished overnight")

    stock_sentiment_trading = sort_df(stock_sentiment_trading)
    stock_sentiment_trading.to_csv(output_data_path + 'StockSentimentTrading_Raw.csv')
    print("finished trading")


# load test
# stock_sentiment_daily = pd.read_csv(output_data_path + 'StockSentimentDaily_Raw.csv', index_col='Date')
# stock_sentiment_overnight = pd.read_csv(output_data_path + 'StockSentimentOvernight_Raw.csv', index_col='Date')
# stock_sentiment_trading = pd.read_csv(output_data_path + 'StockSentimentTrading_Raw.csv', index_col='Date')

# trading_date_frame_df = pd.DataFrame(trading_date_frame, columns=['Date'])
# stock_sentiment_daily_by_symbol = stock_sentiment_daily.groupby(by=stock_sentiment_daily['symbol'])


# test set
# test1 = sentiment_aggregated[sentiment_aggregated['bl_bull_market'].isnull()]
# test4 = sentiment_aggregated[sentiment_aggregated['lm_bull_market'].isnull()]
# test5 = sentiment_aggregated[sentiment_aggregated['sm_bull_market'].isnull()]
# test2 = sentiment_aggregated[sentiment_aggregated.index.date == datetime.date(2010,5,17)]
# test3 = articles_count[articles_count.index.date == datetime.date(2010,5,17)]

if __name__ == '__main__':
    main()

