# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import sqlite3 as sq
import datetime
import re

"""
Description:
 - Join 
"""


def ceil_dt(dt, delta):
    print(dt)
    dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    delta = datetime.timedelta(minutes=delta)
    return dt + ((datetime.datetime.min - dt) % delta)


def index_func(sentiment_value):
    if sentiment_value > 0: return 1
    if sentiment_value < 0: return -1
    if sentiment_value == 0: return 0


def fetch_database(sql_com):
    conn = sq.connect(financial_data_path + 'lexica/nasdaqnews.db')
    cursor = conn.cursor()
    cursor.execute(sql_com)
    data = cursor.fetchall()
    return data


def clean_df(df, delta=5):
    # format date
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d %H:%M:%S')
    # round time to time interval
    df['RoundedTime'] = [ceil_dt(dt, delta=delta) for dt in df.date]
    df.set_index(keys='RoundedTime', drop=True, inplace=True)  # set rounded time as index
    df.sort_index(axis=0, ascending=True, inplace=True)
    return df


def buller(pos, neg, neut):
    total = pos + neg + neut
    bn = np.log((1 + pos / total) / (1 + neg / total))
    return bn / np.log(2)

# direct define
direct = '/Volumes/JH_PD/DataBase/NasdaqSentiment/'
output_data_path = os.path.join(direct, 'output/csv/')
financial_data_path = os.path.join(direct, 'data/')



sentiment_data_raw = pd.read_csv(output_data_path + '20170824_sentiment.csv', sep=';')
articles_data = fetch_database(sql_com='Select linkid, date from articles;')
articles_data_df = pd.DataFrame(articles_data, columns=['linkid', 'date'])  # no duplicated before dump last words
sentiment_date = pd.merge(left=sentiment_data_raw, right=articles_data_df, on='linkid')
sentiment_date = sentiment_date[~ (sentiment_date['date'].isnull())]
sentiment_date['date'] = pd.to_datetime(sentiment_date.date, format='%Y-%m-%d %H:%M:%S')
sentiment_date.set_index(keys='date', inplace=True)  # set rounded time as index
sentiment_date['date'] = sentiment_date.index
sentiment_date.sort_index(axis=0, ascending=True, inplace=True)

linkid_clean = [re.sub('-\w+\d{1,6}', '', title) for title in sentiment_date.linkid.values]
sentiment_date.loc[:, 'linkid'] = linkid_clean
sentiment_date_uni = sentiment_date[~ (sentiment_date.duplicated())]  # articles with no duplicate
sentiment_date_uni['date'] = [dt.date() for dt in sentiment_date_uni.index]
sentiment_date_uni.drop(labels='linkid', axis=1, inplace=True)

# compute sentiment for each article
sentiment_date_uni['bl_bull'] = buller(pos=sentiment_date_uni['bl_pos'],
                                       neg=sentiment_date_uni['bl_neg'],
                                       neut=sentiment_date_uni['bl_neut'])
sentiment_date_uni['lm_bull'] = buller(pos=sentiment_date_uni['lm_pos'],
                                       neg=sentiment_date_uni['lm_neg'],
                                       neut=sentiment_date_uni['lm_neut'])
sentiment_date_uni['sm_bull'] = buller(pos=sentiment_date_uni['sm_pos'],
                                       neg=sentiment_date_uni['sm_neg'],
                                       neut=sentiment_date_uni['sm_neut'])
# negative sentiment
sentiment_date_uni['bl_bull_neg'] = sentiment_date_uni['bl_bull']
sentiment_date_uni.ix[sentiment_date_uni['bl_bull_neg'] > 0, 'bl_bull_neg'] = 0

sentiment_date_uni['lm_bull_neg'] = sentiment_date_uni['lm_bull']
sentiment_date_uni.ix[sentiment_date_uni['lm_bull_neg'] > 0, 'lm_bull_neg'] = 0

sentiment_date_uni['sm_bull_neg'] = sentiment_date_uni['sm_bull']
sentiment_date_uni.ix[sentiment_date_uni['sm_bull_neg'] > 0, 'sm_bull_neg'] = 0

# positive sentiment
sentiment_date_uni['bl_bull_pos'] = sentiment_date_uni['bl_bull']
sentiment_date_uni.ix[sentiment_date_uni['bl_bull_pos'] < 0, 'bl_bull_pos'] = 0

sentiment_date_uni['lm_bull_pos'] = sentiment_date_uni['lm_bull']
sentiment_date_uni.ix[sentiment_date_uni['lm_bull_pos'] < 0, 'lm_bull_pos'] = 0

sentiment_date_uni['sm_bull_pos'] = sentiment_date_uni['sm_bull']
sentiment_date_uni.ix[sentiment_date_uni['sm_bull_pos'] < 0, 'sm_bull_pos'] = 0

sentiment_date_uni.drop(labels=['bl_neg', 'bl_neut', 'bl_pos',
                                'lm_neg', 'lm_neut', 'lm_pos',
                                'sm_neg', 'sm_neut', 'sm_pos',
                                'bl_bull', 'lm_bull', 'sm_bull'],
                        axis=1, inplace=True)

# set index function for sentiment
sentiment_date_uni.loc[:, 'bl_bull_neg'] = [index_func(st_value) for st_value in sentiment_date_uni['bl_bull_neg'].values]
sentiment_date_uni.loc[:, 'lm_bull_neg'] = [index_func(st_value) for st_value in sentiment_date_uni['lm_bull_neg'].values]
sentiment_date_uni.loc[:, 'sm_bull_neg'] = [index_func(st_value) for st_value in sentiment_date_uni['sm_bull_neg'].values]
sentiment_date_uni.loc[:, 'bl_bull_pos'] = [index_func(st_value) for st_value in sentiment_date_uni['bl_bull_pos'].values]
sentiment_date_uni.loc[:, 'lm_bull_pos'] = [index_func(st_value) for st_value in sentiment_date_uni['lm_bull_pos'].values]
sentiment_date_uni.loc[:, 'sm_bull_pos'] = [index_func(st_value) for st_value in sentiment_date_uni['sm_bull_pos'].values]

# counting positive and negative articles each day
article_sentiment_groupby_date = sentiment_date_uni.groupby(by=sentiment_date_uni['date'], axis=0).sum(axis=0)
# article_sentiment_groupby_date = abs(article_sentiment_groupby_date)
article_sentiment_groupby_date.to_csv(output_data_path + 'CountPosNegArticles.csv')
article_sentiment_groupby_date.loc[:, ['sm_bull_pos', 'sm_bull_neg']].plot(kind='bar')
article_sentiment_groupby_date.describe()

#test for stock sentiment

stock_sentiment_daily = pd.read_csv(output_data_path + 'StockSentimentDaily.csv', index_col='Date')
daily_symbol = stock_sentiment_daily['symbol']
daily_symbol_unique = pd.DataFrame(daily_symbol.unique(), columns=['Unique Symbol'])
daily_symbol_unique.to_csv(output_data_path + 'uni_symbol_daily.csv')

stock_sentiment_overnight = pd.read_csv(output_data_path + 'StockSentimentOvernight_Raw.csv', index_col='Date')
overnight_symbol = stock_sentiment_overnight['symbol']
overnight_symbol_unique = pd.DataFrame(overnight_symbol.unique(), columns=['Unique Symbol'])
overnight_symbol_unique.to_csv(output_data_path + 'uni_symbol_overnight.csv')


stock_sentiment_trading = pd.read_csv(output_data_path + 'StockSentimentTrading_Raw.csv', index_col='Date')
trading_symbol = stock_sentiment_trading['symbol']
trading_symbol_unique = pd.DataFrame(trading_symbol.unique(), columns=['Unique Symbol'])
trading_symbol_unique.to_csv(output_data_path + 'uni_symbol_trading.csv')


set_daily = daily_symbol.values
set_overnight = overnight_symbol.values
set_trading = trading_symbol.values

chk = [syb for syb in set_daily if syb not in set_overnight]
