# -*- coding:utf-8 -*-
import pandas as pd
import re
import os
import datetime
import numpy as np
import sqlite3 as sq


def ceil_dt(dt, delta):
    print(dt)
    dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    delta = datetime.timedelta(minutes=delta)
    return dt + ((datetime.datetime.min - dt) % delta)


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


def clean_df(df, delta=5):
    # format date
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d %H:%M:%S')
    # round time to time interval
    df['RoundedTime'] = [ceil_dt(dt, delta=delta) for dt in df.date]
    df.set_index(keys='RoundedTime', drop=True, inplace=True)  # set rounded time as index
    df.sort_index(axis=0, ascending=True, inplace=True)
    return df


direct = '/Volumes/JeremyHu/DataBase/NasdaqSentiment/'
output_data_path = os.path.join(direct, 'output/csv/')
financial_data_path = os.path.join(direct, 'data/')


def main(refresh=False, time_interval=5):
    if (not os.path.exists(output_data_path + str(time_interval) + 'AggregatedMarketSentiment_Raw.csv')) or (
        refresh is True):
        # execute processing if file doesn't exist
        """
        sentiment_date file contains articles and corresponding company ticker and date.
        This function does:
        1. round the article time to fixed interval
        2. calculate sentiment and number of articles for each time interval
        3. merge calendar time frame and output
        """
        # read all data needed
        sentiment_data_raw = pd.read_csv(output_data_path + '20170824_sentiment.csv', sep=';')
        articles_data = fetch_database(sql_com='Select linkid, date from articles;')
        articles_data_df = pd.DataFrame(articles_data, columns=['linkid', 'date'])  # no duplicated before dump last words

        # symbol_data = fetch_database(sql_com='Select * from symbols;')
        # symbol_data_df = pd.DataFrame(symbol_data, columns=['linkid', 'symbol'])

        # merge sentiment and article date
        sentiment_date = pd.merge(left=sentiment_data_raw, right=articles_data_df, on='linkid')
        # 1 row has no date info, drop it
        sentiment_date = sentiment_date[~ (sentiment_date['date'].isnull())]
        # merge sentiment, date, company ticker
        # sentiment_date_symbol = pd.merge(left=sentiment_date, right=symbol_data_df, on='linkid')

        sentiment_date = clean_df(sentiment_date, delta=time_interval)  # format date, round time, set date as index

        linkid_clean = [re.sub('-\w+\d{1,6}', '', title) for title in sentiment_date.linkid.values]
        sentiment_date.ix[:, 'linkid'] = linkid_clean
        sentiment_date_uni = sentiment_date[~ (sentiment_date.duplicated())]  # articles with no duplicate

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

        sentiment_date_uni.drop(labels='date', axis=1, inplace=True)
        # sentiment_date_symbol = clean_df(sentiment_date_symbol, delta=time_interval)
        # sentiment_date_symbol.drop(labels=['date', 'symbol'], axis=1, inplace=True)

        # since the linkid has redundant suffix which will make duplicated linkid for one same news
        # clean the linkid so that we can drop the redundant linkid


        # sentiment_date_symbol_by_time = sentiment_date_symbol.groupby(by=sentiment_date_symbol.index, axis=0).sum(axis=0)
        sentiment_date_uni.drop(labels='linkid', axis=1, inplace=True)
        # aggregate by RoundedTime
        sentiment_date_aggregated_by_time = sentiment_date_uni.groupby(by=sentiment_date_uni.index,
                                                                       axis=0, sort=True).sum(axis=0)

        # counting articles
        # aggregate by time to count number of articles in a fixed time interval
        articles_count = sentiment_date_uni.groupby(by=sentiment_date_uni.index, axis=0).count()['bl_neg']
        articles_count = articles_count.to_frame(name='Number of Articles')
        articles_count.to_csv(output_data_path + 'articles_count.csv')

        sentiment_date_aggregated_by_time['bl_bull_market'] = buller(pos=sentiment_date_aggregated_by_time['bl_pos'],
                                                                     neg=sentiment_date_aggregated_by_time['bl_neg'],
                                                                     neut=sentiment_date_aggregated_by_time['bl_neut'])
        sentiment_date_aggregated_by_time['lm_bull_market'] = buller(pos=sentiment_date_aggregated_by_time['lm_pos'],
                                                                     neg=sentiment_date_aggregated_by_time['lm_neg'],
                                                                     neut=sentiment_date_aggregated_by_time['lm_neut'])
        sentiment_date_aggregated_by_time['sm_bull_market'] = buller(pos=sentiment_date_aggregated_by_time['sm_pos'],
                                                                     neg=sentiment_date_aggregated_by_time['sm_neg'],
                                                                     neut=sentiment_date_aggregated_by_time['sm_neut'])

        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['bl_bull_market'].isnull(), 'bl_bull_market'] = 0
        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['lm_bull_market'].isnull(), 'lm_bull_market'] = 0
        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['sm_bull_market'].isnull(), 'sm_bull_market'] = 0

        # negative sentiment
        sentiment_date_aggregated_by_time['bl_bull_market_neg'] = sentiment_date_aggregated_by_time['bl_bull_market']
        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['bl_bull_market_neg'] > 0, 'bl_bull_market_neg'] = 0

        sentiment_date_aggregated_by_time['lm_bull_market_neg'] = sentiment_date_aggregated_by_time['lm_bull_market']
        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['lm_bull_market_neg'] > 0, 'lm_bull_market_neg'] = 0

        sentiment_date_aggregated_by_time['sm_bull_market_neg'] = sentiment_date_aggregated_by_time['sm_bull_market']
        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['sm_bull_market_neg'] > 0, 'sm_bull_market_neg'] = 0

        # positive sentiment
        sentiment_date_aggregated_by_time['bl_bull_market_pos'] = sentiment_date_aggregated_by_time['bl_bull_market']
        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['bl_bull_market_pos'] < 0, 'bl_bull_market_pos'] = 0

        sentiment_date_aggregated_by_time['lm_bull_market_pos'] = sentiment_date_aggregated_by_time['lm_bull_market']
        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['lm_bull_market_pos'] < 0, 'lm_bull_market_pos'] = 0

        sentiment_date_aggregated_by_time['sm_bull_market_pos'] = sentiment_date_aggregated_by_time['sm_bull_market']
        sentiment_date_aggregated_by_time.ix[
            sentiment_date_aggregated_by_time['sm_bull_market_pos'] < 0, 'sm_bull_market_pos'] = 0

        sentiment_date_aggregated_by_time.to_csv(output_data_path + 'market_sentiment_raw_20170827.csv')

        # abs sentiment
        sentiment_date_aggregated_by_time['bl_bull_market_abs'] = np.abs(
            sentiment_date_aggregated_by_time['bl_bull_market'])
        sentiment_date_aggregated_by_time['lm_bull_market_abs'] = np.abs(
            sentiment_date_aggregated_by_time['lm_bull_market'])
        sentiment_date_aggregated_by_time['sm_bull_market_abs'] = np.abs(
            sentiment_date_aggregated_by_time['sm_bull_market'])

        # square sentiment
        sentiment_date_aggregated_by_time['bl_bull_market_square'] = np.square(
            sentiment_date_aggregated_by_time['bl_bull_market'])
        sentiment_date_aggregated_by_time['lm_bull_market_square'] = np.square(
            sentiment_date_aggregated_by_time['lm_bull_market'])
        sentiment_date_aggregated_by_time['sm_bull_market_square'] = np.square(
            sentiment_date_aggregated_by_time['sm_bull_market'])

        # save
        sentiment_date_aggregated_by_time.to_csv(output_data_path + str(time_interval) +
                                                 'AggregatedMarketSentiment_Raw.csv')

    else:
        sentiment_date_aggregated_by_time = pd.read_csv(output_data_path + str(time_interval)
                                                        + 'AggregatedMarketSentiment_Raw.csv',
                                                        index_col=0)
        sentiment_date_aggregated_by_time.set_index(pd.to_datetime(sentiment_date_aggregated_by_time.index,
                                                                   format='%Y-%m-%d %H:%M:%S'), inplace=True)
        articles_count = pd.read_csv(output_data_path + 'articles_count.csv', index_col=0)
        articles_count.set_index(pd.to_datetime(articles_count.index, format='%Y-%m-%d %H:%M:%S'))

    # prepare to output
    sentiment_date_aggregated_by_time.drop(labels=['bl_neg', 'bl_neut', 'bl_pos',
                                                   'lm_neg', 'lm_neut', 'lm_pos',
                                                   'sm_neg', 'sm_neut', 'sm_pos'],
                                           axis=1, inplace=True)
    sentiment_date_aggregated_by_time = pd.concat([sentiment_date_aggregated_by_time, articles_count], axis=1)
    sentiment_date_aggregated_by_time.index.name = 'Date'

    # basic statistics
    basic_stat = sentiment_date_aggregated_by_time.describe()
    basic_stat.to_csv(output_data_path + str(time_interval) + 'minInterval_basic_stat.csv')
    # take date after 2012
    sentiment_date_aggregated_by_time = sentiment_date_aggregated_by_time[
        sentiment_date_aggregated_by_time.index.year >= 2012
        ]

    # merge date frame
    begin_date = min(sentiment_date_aggregated_by_time.index)
    end_date = max(sentiment_date_aggregated_by_time.index)
    date_freq = str(time_interval) + 'min'
    time_axis = pd.date_range(start=begin_date, end=end_date, freq=date_freq)
    sentiment_aggregated_even_time = sentiment_date_aggregated_by_time.reindex(time_axis)
    sentiment_aggregated_even_time.fillna(0, inplace=True)
    # separate date into 2 columns
    sentiment_aggregated_even_time['date'] = [dt.date() for dt in sentiment_aggregated_even_time.index]
    sentiment_aggregated_even_time['time'] = [dt.time() for dt in sentiment_aggregated_even_time.index]
    name_now = datetime.datetime.now().date()
    out_put_name = 'MarketSentiment_{0}min_EvenTimeFrame_{1}.csv'.format(time_interval, name_now)
    sentiment_aggregated_even_time.to_csv(output_data_path + out_put_name)


# test set
# test1 = sentiment_aggregated[sentiment_aggregated['bl_bull_market'].isnull()]
# test4 = sentiment_aggregated[sentiment_aggregated['lm_bull_market'].isnull()]
# test5 = sentiment_aggregated[sentiment_aggregated['sm_bull_market'].isnull()]
# test2 = sentiment_aggregated[sentiment_aggregated.index.date == datetime.date(2010,5,17)]
# test3 = articles_count[articles_count.index.date == datetime.date(2010,5,17)]

if __name__ == '__main__':
    main(refresh=True, time_interval=5)
    main(refresh=True, time_interval=15)
