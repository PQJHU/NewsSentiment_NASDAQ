# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import sqlite3 as sq
import datetime
import re
import matplotlib.pyplot as plt


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


def clean_df(df, delta=15):
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
output_data_path = os.path.join(direct, 'output/')
financial_data_path = os.path.join(direct, 'data/')

# Read in the sentiment data and article data
sentiment_data_raw = pd.read_csv(output_data_path + 'csv/20170824_sentiment.csv', sep=';')
articles_data = fetch_database(sql_com='Select linkid, date from articles;')
articles_data_df = pd.DataFrame(articles_data, columns=['linkid', 'date'])  # no duplicated before dump last words
# Merge them together by linkid
sentiment_date = pd.merge(left=sentiment_data_raw, right=articles_data_df, on='linkid')
sentiment_date = sentiment_date[~ (sentiment_date['date'].isnull())]

sentiment_date = clean_df(sentiment_date, delta=15)

# sentiment_date['date'] = pd.to_datetime(sentiment_date.date, format='%Y-%m-%d %H:%M:%S')
# sentiment_date.set_index(keys='date', inplace=True, drop=False)  # set rounded time as index
# sentiment_date.sort_index(axis=0, ascending=True, inplace=True)

# truncate the last numeric part of the link
linkid_clean = [re.sub('-\w+\d{1,6}', '', title) for title in sentiment_date.linkid.values]
sentiment_date.loc[:, 'linkid'] = linkid_clean
sentiment_date_uni = sentiment_date[~ (sentiment_date.duplicated())]  # articles with no duplicate

# calculate how many articles in yearly
# only after 2012
sentiment_date_uni = sentiment_date_uni[sentiment_date_uni.index.year >= 2012]

# take only trading hour articles
# only trading hour, 9:30am - 16:00pm
sentiment_date_uni_trading = sentiment_date_uni[
    (sentiment_date_uni.index.time >= datetime.time(hour=9, minute=30))
    & (sentiment_date_uni.index.time <= datetime.time(hour=16, minute=0))
    ]

# ====================count yearly articles===================
sentiment_date_yearly = sentiment_date_uni_trading.groupby(
    by=sentiment_date_uni_trading.index.year, axis=0)['linkid'].apply(lambda x: len(x))
sentiment_date_yearly.values.sum()
sentiment_date_yearly.values.mean()
# count 15-min articles
sentiment_date_15_min = sentiment_date_uni_trading.groupby(
    by=sentiment_date_uni_trading.index, axis=0)['linkid'].apply(lambda x: len(x))
sentiment_date_15_min.to_csv(output_data_path + 'nasdaq_articles_every_15_min_tradinghour.csv')
sentiment_date_15_min.mean()

fig = plt.figure()
plt.plot(sentiment_date_15_min)
plt.title('Number of articles of 15-min')
plt.savefig(output_data_path + 'nasdaq_articles_every_15_min_tradinghour.png', dpi=300)

# ===============Count number of type of sentiment===========
# drop linkid as its not using any more
sentiment_date_uni_trading.drop(['linkid', 'date'], axis=1, inplace=True)

# compute sentiment for each article
sentiment_date_uni_trading['bl_bull'] = buller(pos=sentiment_date_uni_trading['bl_pos'],
                                               neg=sentiment_date_uni_trading['bl_neg'],
                                               neut=sentiment_date_uni_trading['bl_neut'])
sentiment_date_uni_trading['lm_bull'] = buller(pos=sentiment_date_uni_trading['lm_pos'],
                                               neg=sentiment_date_uni_trading['lm_neg'],
                                               neut=sentiment_date_uni_trading['lm_neut'])
sentiment_date_uni_trading['sm_bull'] = buller(pos=sentiment_date_uni_trading['sm_pos'],
                                               neg=sentiment_date_uni_trading['sm_neg'],
                                               neut=sentiment_date_uni_trading['sm_neut'])

# fill na with 0, because the na value is caused by 0 sentiment score which means neutral sentiment
sentiment_date_uni_trading.fillna(0, inplace=True)

# drop all the score columns
sentiment_date_uni_trading.drop(labels=['bl_neg', 'bl_neut', 'bl_pos',
                                        'lm_neg', 'lm_neut', 'lm_pos',
                                        'sm_neg', 'sm_neut', 'sm_pos',
                                        ],
                                axis=1, inplace=True)
# mark pos/neutral/neg sentiment
sentiment_date_uni_trading.loc[:, 'bl_bull'] = [index_func(st_value) for st_value in
                                                sentiment_date_uni_trading['bl_bull'].values]
sentiment_date_uni_trading.loc[:, 'lm_bull'] = [index_func(st_value) for st_value in
                                                sentiment_date_uni_trading['lm_bull'].values]
sentiment_date_uni_trading.loc[:, 'sm_bull'] = [index_func(st_value) for st_value in
                                                sentiment_date_uni_trading['sm_bull'].values]

# Count pos/neut/neg sentiment articles annually
bl_bull_year_count = sentiment_date_uni_trading.groupby(
    by=sentiment_date_uni_trading.index.year, axis=0
)['bl_bull'].apply(lambda x: x.value_counts())

print(bl_bull_year_count)
bl_bull_year_count.sum()

lm_bull_year_count = sentiment_date_uni_trading.groupby(
    by=sentiment_date_uni_trading.index.year, axis=0
)['lm_bull'].apply(lambda x: x.value_counts())

print(lm_bull_year_count)
lm_bull_year_count.sum()

sm_bull_year_count = sentiment_date_uni_trading.groupby(
    by=sentiment_date_uni_trading.index.year, axis=0
)['sm_bull'].apply(lambda x: x.value_counts())

print(sm_bull_year_count)
sm_bull_year_count.sum()

# count pos/neut/neg sentiment articles
bl_bull_15min_count = sentiment_date_uni_trading.groupby(
    by=sentiment_date_uni_trading.index, axis=0
)['bl_bull'].apply(lambda x: x.value_counts())

lm_bull_15min_count = sentiment_date_uni_trading.groupby(
    by=sentiment_date_uni_trading.index, axis=0
)['lm_bull'].apply(lambda x: x.value_counts())

sm_bull_15min_count = sentiment_date_uni_trading.groupby(
    by=sentiment_date_uni_trading.index, axis=0
)['sm_bull'].apply(lambda x: x.value_counts())

articles_sentiment_15min = pd.concat([bl_bull_15min_count, lm_bull_15min_count, sm_bull_15min_count], axis=1)


articles_sentiment_15min.reset_index(level=1, inplace=True)
articles_sentiment_15min.columns = ['sentiment_mark', 'BL_articles_num', 'LM_articles_num', 'SM_articles_num']


def count_sentiment_pct(x):
    return x[['BL_articles_num', 'LM_articles_num', 'SM_articles_num']] / x[['BL_articles_num', 'LM_articles_num', 'SM_articles_num']].sum(axis=0)


articles_sentiment_15min_pct = articles_sentiment_15min.groupby(
    by=articles_sentiment_15min.index, axis=0).apply(count_sentiment_pct)


articles_sentiment_15min / articles_sentiment_15min.sum()

articles_sentiment_15min.fillna(0, inplace=True)
articles_sentiment_15min.to_csv(output_data_path + 'csv/count_articles_sentiment_15min.csv')




import datetime
range_date = pd.date_range(start='9:30', end='16:00', periods=None, freq='15min')
daily_range = pd.date_range(start=datetime.datetime.time(9,30), end=datetime.datetime(16,0), periods=None,)
range_date = [range_date for range_date.time() in ]
sentiment_date = set(sentiment_date_uni_trading.index.date)
test2 = [dt for dt in range_date if dt.date() in sentiment_date]








