from Code.GlobalParams import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Code.SentimentAnalysis.CAL_HolidaysCalendar import generate_holidays
import datetime as dt


# from Code.SentimentGenerator.RID_ReadData import *


def market_status_split(score_df):
    holidays = generate_holidays()
    score_df.sort_index(axis=0, ascending=True, inplace=True)
    score_df = score_df[score_df.index.year >= 2012]

    # score_df['Time'] = score_df.index.ceil(article_round_time)
    # score_df.set_index(keys='Time', inplace=True)
    score_df['days'] = score_df.index.dayofweek

    score_workdays = score_df[~(score_df.days.isin([5, 6]))]
    score_holidays = pd.DataFrame()
    for date in holidays:
        score_holiday = score_workdays[score_workdays.index.date == date]
        score_holidays = pd.concat([score_holidays, score_holiday], axis=0)
        score_workdays = score_workdays[~(score_workdays.index.date == date)]
        print(f'{date}: # {len(score_holiday)} articles')

    score_weekends = score_df[score_df.days.isin([5, 6])]
    return score_workdays, score_holidays, score_weekends


def delta_sentiment_AF(score_df):
    pos = score_df['Pos']
    neg = score_df['Neg']
    neut = score_df['Neutral']
    # pos = pos - pos.mean()
    # neg = neg - neg.mean()
    total = pos + neg + neut
    bn = np.log((1 + pos / total) / (1 + neg / total))
    return bn / np.log(2)


# def delta_sentiment_intraday(pos, neg, neut):
#     total = pos + neg + neut
#
#     bn = (pos - neg) / total
#     return bn


def delta_sentiment_New(score_daily):
    print(f'# Invervals: {len(score_daily)}')
    total = score_daily.sum(axis=1).sum(axis=0)
    sentiment_intraday = (score_daily['Pos'] - score_daily['Neg']) / total
    return sentiment_intraday


def split_symbols(symbol_entry, target_symbol):
    print(symbol_entry)
    if (not isinstance(symbol_entry, str)) and (np.isnan(symbol_entry)):
        return False
    else:
        symbols_list = symbol_entry.split(',')
        if any([symbol == target_symbol for symbol in symbols_list]) is True:
            return True
        else:
            return False


def rows_select_by_symbol(score_full_df, target_symbol):
    score_full_df['if_targeted'] = [split_symbols(symbol_entry, target_symbol) for symbol_entry in
                                    score_full_df['symbols']]
    nasdaq_score_targeted = score_full_df[score_full_df['if_targeted'] == True]
    return nasdaq_score_targeted


def generate_sentiment_process(score_df, round_time=article_round_time, b_type='new', agg=None):
    score_df['Time'] = score_df.index.ceil(round_time)
    score_df.set_index(keys='Time', inplace=True)
    score_df_agg = score_df.groupby(by=score_df.index, axis=0, sort=True).apply(lambda x: x.sum(axis=0))

    if b_type == 'new':
        sentiment_intraday = score_df_agg.groupby(by=score_df_agg.index.date, axis=0, sort=True).apply(
            lambda x: delta_sentiment_New(x))
        sentiment_intraday = sentiment_intraday.to_frame()
        sentiment_intraday.set_index(sentiment_intraday.index.levels[1], inplace=True, drop=True, append=False)
        sentiment_intraday.columns = ['Sentiment']

    elif b_type == 'af':
        if agg == 'daily':

            sentiment_intraday = score_df_agg.groupby(by=score_df_agg.index.date, axis=0, sort=True).apply(
                lambda x: delta_sentiment_AF(x).mean(axis=0))
            # sentiment_intraday = sentiment_intraday.groupby(by=sentiment_intraday.index.levels[1].date, sort=True, axis=0).apply(lambda x: x.mean(axis=0))
            sentiment_intraday = sentiment_intraday.to_frame()
            sentiment_intraday.columns = ['Sentiment']


        elif agg is None:
            sentiment_intraday = score_df_agg.groupby(by=score_df_agg.index, axis=0, sort=True).apply(
                lambda x: delta_sentiment_AF(x))
            sentiment_intraday = sentiment_intraday.to_frame()
            sentiment_intraday.set_index(sentiment_intraday.index.levels[1], inplace=True, drop=True, append=False)
            sentiment_intraday.columns = ['Sentiment']

        else:
            raise KeyError('Keyword agg wrong for type selection')

    else:
        raise KeyError('Keyword b_type wrong for type selection')

    return sentiment_intraday

def Sentiment_with_News_Filtering(subset='all', agg='15min', ovn='day', sp_days=None):

    nasdaq_score = pd.read_csv(outdata_path + 'nasdaq_news/LinearClf_Polarity.csv',
                               index_col='article_time',
                               parse_dates=True)

    # Sorting by time, taking news after 2012, and around the news to 5-minute
    nasdaq_score.sort_index(axis=0, ascending=True, inplace=True)
    nasdaq_score = nasdaq_score[nasdaq_score.index.year >= 2012]
    nasdaq_score['Time'] = nasdaq_score.index.ceil(article_round_time)
    nasdaq_score.set_index(keys='Time', inplace=True)
    # nasdaq_score_check = nasdaq_score[nasdaq_score.index == dt.datetime(2019, 4, 25, 10, 45, 0)]

    # ======== ROW SELECTION ========

    if subset == 'all':
    # All news
        score_rowselected = nasdaq_score[['Pos', 'Neg', 'Neutral']]
    else:
        score_rowselected = rows_select_by_symbol(score_full_df=nasdaq_score, target_symbol=subset)
        score_rowselected = score_rowselected[['Pos', 'Neg', 'Neutral']]

    if sp_days is not None:
        score_sel_workingdays, score_sel_holidays, score_sel_weekends = market_status_split(score_rowselected)
        if sp_days == 'workingdays':
            score_row_days_selected = score_sel_workingdays
        elif sp_days == 'holidays':
            score_row_days_selected = score_sel_holidays
        elif sp_days == 'weekends':
            score_row_days_selected = score_sel_weekends
        else:
            raise KeyError("Special days keywords wrong!")
    else:
        score_row_days_selected = score_rowselected


    # ======== Sentiment Process ============
    # Generate sentiment processes using score
    if agg == 'daily':
        sentiment = generate_sentiment_process(score_df=score_row_days_selected, b_type='af', agg=agg)
    else:
        sentiment = generate_sentiment_process(score_df=score_row_days_selected, b_type='af', agg=None)


    # Save sentiment processes for further use
    sentiment.to_csv(outdata_path + f'sentiment/Sentiment_{subset}_{sp_days}_Agg{agg}.csv')

    # === company selection by symbol
    # Company specified news

    # nasdaq_score_aapl = rows_select_by_symbol(score_full_df=nasdaq_score, target_symbol='AAPL')
    # nasdaq_score_aapl = nasdaq_score_aapl[['Pos', 'Neg', 'Neutral']]

    # nasdaq_score_amzn = rows_select_by_symbol(score_full_df=nasdaq_score, target_symbol='AMZN')
    # nasdaq_score_amzn = nasdaq_score_amzn[['Pos', 'Neg', 'Neutral']]

# Sentiment_with_News_Filtering(subset='all', agg='daily', ovn='full', sp_days='workingdays')
# Sentiment_with_News_Filtering(subset='all', agg='daily', ovn='full', sp_days='holidays')
# Sentiment_with_News_Filtering(subset='all', agg='daily', ovn='full', sp_days='weekends')
# Sentiment_with_News_Filtering(subset='AAPL', agg='15min', ovn='full', sp_days=None)
# Sentiment_with_News_Filtering(subset='AMZN', agg='15min', ovn='full', sp_days=None)
Sentiment_with_News_Filtering(subset='AAPL', agg='daily', ovn='full', sp_days=None)
Sentiment_with_News_Filtering(subset='AMZN', agg='daily', ovn='full', sp_days=None)


# Market related news
# TBD


# sentiment_all_workingdays = generate_sentiment_process(score_df=nasdaq_score_all_workingdays[['Pos', 'Neg', 'Neutral']],
#                                                        b_type='af', agg=None)
#
#
# sentiment_all_holidays = generate_sentiment_process(score_df=nasdaq_score_all_holidays[['Pos', 'Neg', 'Neutral']],
#                                                     b_type='af', agg=None)
#
# sentiment_all_weekends = generate_sentiment_process(score_df=nasdaq_score_all_weekends[['Pos', 'Neg', 'Neutral']],
#                                                     b_type='af', agg=None)
#
# sentiment_aapl_daily = generate_sentiment_process(score_df=nasdaq_score_aapl, b_type='af', agg=None)
# sentiment_amzn_daily = generate_sentiment_process(score_df=nasdaq_score_amzn, b_type='af', agg=None)

# Save sentiment processes for further use
# sentiment_all_workingdays.to_csv(outdata_path + f'sentiment/Sentiment_All_Workingdays_Agg{article_round_time}.csv')
# sentiment_all_holidays.to_csv(outdata_path + f'sentiment/Sentiment_All_Holidays_Agg{article_round_time}.csv')
# sentiment_all_weekends.to_csv(outdata_path + f'sentiment/Sentiment_All_Weekends_Agg{article_round_time}.csv')
# sentiment_aapl_daily.to_csv(outdata_path + f'sentiment/Sentiment_AAPL_Agg{article_round_time}.csv')
# sentiment_amzn_daily.to_csv(outdata_path + f'sentiment/Sentiment_AMZN_Agg{article_round_time}.csv')


# Test

# nasdaq_score_all['Time'] = nasdaq_score_all.index.ceil(article_round_time)
# nasdaq_score_all.set_index(keys='Time', inplace=True)
# score_df_agg = nasdaq_score_all.groupby(by=nasdaq_score_all.index, axis=0, sort=True).apply(lambda x: x.sum(axis=0))
#
# sentiment_bullish = score_df_agg.groupby(by=score_df_agg.index, axis=0, sort=True).apply(lambda x: delta_sentiment_AF(x))
# sentiment_bullish = sentiment_bullish.to_frame()
# sentiment_bullish.set_index(sentiment_bullish.index.levels[1], inplace=True, drop=True, append=False)
# sentiment_bullish.columns = ['Sentiment']
# sentiment_bullish_nonzero = sentiment_bullish[~(sentiment_bullish['Sentiment'] == 0)]
# plt.hist(sentiment_bullish['Sentiment'].values, bins=5000)
#
# sentiment_intraday = score_df_agg.groupby(by=score_df_agg.index.date, axis=0, sort=True).apply(
#     lambda x: delta_sentiment_New(x))
# sentiment_intraday = sentiment_intraday.to_frame()
# sentiment_intraday.set_index(sentiment_intraday.index.levels[1], inplace=True, drop=True, append=False)
# sentiment_intraday.columns = ['Sentiment']
#
# sentiment_nonzero = sentiment_intraday[~(sentiment_intraday['Sentiment'] == 0)]
# plt.hist(sentiment_nonzero['Sentiment'].values, bins=5000)
#
# plt.plot(sentiment_intraday['Sentiment'])

# sentiment_intraday['logsentiment'] = np.log(sentiment_intraday['Sentiment']+1)
#
# sentiment_intraday['diff'] = sentiment_intraday['logsentiment'] - sentiment_intraday['logsentiment'].shift(1)
# plt.plot(sentiment_intraday['diff'])

# nasdaq_score_all_intraday = nasdaq_score_all.groupby(by=nasdaq_score_all.index.date, axis=0, sort=True).apply(
#     lambda x: delta_sentiment_New(x))
#
# sentiment_all = generate_sentiment_process(score_df=nasdaq_score_all)
#
# ts = sentiment_all['sentiment']
#
# ts_pos = ts[ts > 0]
# ts_neg = ts[ts < 0]
