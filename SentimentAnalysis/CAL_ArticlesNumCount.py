from Code.GlobalParams import *
import pandas as pd
import matplotlib.pyplot as plt
from Code.SentimentAnalysis.CAL_HolidaysCalendar import generate_holidays



# def sentiment_process(pos, neg, neut):
#     total = pos + neg + neut
#     bn = np.log((1 + pos / total) / (1 + neg / total))
#     return bn / np.log(2)

holidays = generate_holidays()
nasdaq_score = pd.read_csv(outdata_path + 'nasdaq_news/LinearClf_Polarity.csv', index_col='article_time', parse_dates=True)[
    ['Pos', 'Neg', 'Neutral']]
nasdaq_score.sort_index(axis=0, ascending=True, inplace=True)
nasdaq_score = nasdaq_score[nasdaq_score.index.year >= 2012]

nasdaq_score['Time'] = nasdaq_score.index.ceil(article_round_time)
nasdaq_score.set_index(keys='Time', inplace=True)
nasdaq_score['days'] = nasdaq_score.index.dayofweek

nasdaq_score_workdays = nasdaq_score[~(nasdaq_score.days.isin([5,6]))]

nasdaq_score_holidays = pd.DataFrame()
for date in holidays:
    score_holiday = nasdaq_score_workdays[nasdaq_score_workdays.index.date == date]
    nasdaq_score_holidays = pd.concat([nasdaq_score_holidays, score_holiday], axis=0)
    nasdaq_score_workdays = nasdaq_score_workdays[~(nasdaq_score_workdays.index.date == date)]
    print(f'{date}: # {len(score_holiday)} articles')

nasdaq_score_weekends = nasdaq_score[nasdaq_score.days.isin([5,6])]


# Number of articles in working days and holidays
nasdaq_article_num_workdays = nasdaq_score_workdays.groupby(by=nasdaq_score_workdays.index.date, axis=0, sort=True).apply(lambda x: len(x))
nasdaq_article_num_weekends =nasdaq_score_weekends.groupby(by=nasdaq_score_weekends.index.date, axis=0, sort=True).apply(lambda x: len(x))
nasdaq_article_num_holidays = nasdaq_score_holidays.groupby(by=nasdaq_score_holidays.index.date, axis=0, sort=True).apply(lambda x: len(x))


# Counting number of article in different periods
# nasdaq_score['Time'] =

plt.figure(figsize=(15,7))
plt.scatter(x=nasdaq_article_num_workdays.index, y=nasdaq_article_num_workdays.values, s=20, c='b', marker='.', label='NASDAQ Open Day')
plt.scatter(x=nasdaq_article_num_weekends.index, y=nasdaq_article_num_weekends.values, s=20, c='r', marker="x", label='Weekends')
plt.scatter(x=nasdaq_article_num_holidays.index, y=nasdaq_article_num_holidays.values, s=20, c='g', marker="*", label='NASDAQ Holiday Day')
plt.xlabel('Time', fontsize=15)
plt.ylabel('# Articles', fontsize=15)
plt.legend()
plt.tight_layout()

plt.savefig(outplot_path + 'NumberOfArticles.png', dpi=300)

