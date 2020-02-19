# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import sqlite3 as sq
import datetime
import re


def ceil_dt(dt, delta):
    print(dt)
    dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    delta = datetime.timedelta(minutes=delta)
    return dt + ((datetime.datetime.min - dt) % delta)


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

direct = '/Users/hujunjie/Desktop/CodeBase/NasdaqSentiment/fss-code-master/'
output_data_path = os.path.join(direct, 'output/csv/')
financial_data_path = os.path.join(direct, 'data/')

time_interval=5

sentiment_data_raw = pd.read_csv(output_data_path + '20170824_sentiment.csv', sep=';')
articles_data = fetch_database(sql_com='Select linkid, date from articles;')
articles_data_df = pd.DataFrame(articles_data, columns=['linkid', 'date'])  # no duplicated before dump last words
symbol_data = fetch_database(sql_com='Select * from symbols;')
symbol_data_df = pd.DataFrame(symbol_data, columns=['linkid', 'symbol'])

# merge sentiment and article date
sentiment_date = pd.merge(left=sentiment_data_raw, right=articles_data_df, on='linkid')
# merge sentiment, date, company ticker
sentiment_date_symbol = pd.merge(left=sentiment_date, right=symbol_data_df, on='linkid')

# 1 row has no date info, drop it
sentiment_date_symbol = clean_df(sentiment_date_symbol, delta=time_interval)  # format date, round time, set date as index

linkid_clean = [re.sub('-\w+\d{1,6}', '', title) for title in sentiment_date_symbol.linkid.values]
sentiment_date.ix[:, 'linkid'] = linkid_clean
sentiment_date_uni = sentiment_date[~ (sentiment_date.duplicated())]  # articles with no duplicate

dup_check_data = pd.read_csv(output_data_path + 'dup_linkid_check_all.csv', index_col=0)

dup_check_data_count = dup_check_data['linkid'].unique()

rand_num = np.random.randint(0, len(dup_check_data_count))

dup_rand_check = sentiment_date[sentiment_date['linkid'] == dup_check_data_count[rand_num]]

df = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'], index=pd.date_range('1/1/2000', periods=10))
df.iloc[3:7] = np.nan
df.agg(['sum', 'min'])
