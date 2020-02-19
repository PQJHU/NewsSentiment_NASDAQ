import matplotlib.pyplot as plt
from Code.GlobalParams import *
import pandas as pd


# read sentiment processes
sentiment_all_workingdays = pd.read_csv(outdata_path + f'sentiment/Sentiment_all_Workingdays_Aggdaily.csv', index_col='Time', parse_dates=True)
sentiment_all_holidays = pd.read_csv(outdata_path + f'sentiment/Sentiment_all_Holidays_Aggdaily.csv', index_col='Time', parse_dates=True)
sentiment_all_weekends = pd.read_csv(outdata_path + f'sentiment/Sentiment_all_Weekends_Aggdaily.csv', index_col='Time', parse_dates=True)
sentiment_aapl_daily = pd.read_csv(outdata_path + f'sentiment/Sentiment_AAPL_None_Aggdaily.csv', index_col='Time', parse_dates=True)
sentiment_amzn_daily = pd.read_csv(outdata_path + f'sentiment/Sentiment_AMZN_None_Aggdaily.csv', index_col='Time', parse_dates=True)


sentiment_all_workingdays['ma'] = sentiment_all_workingdays.rolling(window=10).mean()
sentiment_aapl_daily['ma'] = sentiment_aapl_daily.rolling(window=10).mean()
sentiment_amzn_daily['ma'] = sentiment_amzn_daily.rolling(window=10).mean()


plt.figure(figsize=(15, 7))
plt.plot(sentiment_all_workingdays['Sentiment'], color='blue', linestyle='--', label=r'\textbf{Daily Average Bullish}')
plt.plot(sentiment_all_workingdays['ma'], color='r', linestyle='-', label='MA(10)')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Values', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig(outplot_path + 'MarketSentiment_MA.png', dpi=300)


plt.figure(figsize=(15, 7))
plt.plot(sentiment_aapl_daily['Sentiment'], color='blue', linestyle='--', label='Daily Average Bullish')
plt.plot(sentiment_aapl_daily['ma'], color='r', linestyle='-', label='MA(10)')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Values', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig(outplot_path + 'AAPLSentiment_MA.png', dpi=300)

plt.figure(figsize=(15, 7))
plt.plot(sentiment_amzn_daily['Sentiment'], color='blue', linestyle='--', label='Daily Average Bullish')
plt.plot(sentiment_amzn_daily['ma'], color='r', linestyle='-', label='MA(10)')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Values', fontsize=15)
# plt.legend()
plt.tight_layout()
plt.savefig(outplot_path + 'AMZNSentiment_MA_nolegend.png', dpi=300)


# ======== Plot sentiment process ==============


# Score process
pos_score_all = nasdaq_score_all['Pos']
neg_score_all = nasdaq_score_all['Neg']


def fig_score_process(pos_score, neg_score):
    fig = plt.figure(figsize=(15, 7))
    plt.scatter(x=pos_score.index, y=pos_score.values, s=5, c='g', marker=6, label='Positive Score')
    plt.scatter(x=neg_score.index, y=-neg_score.values, s=5, c='r', marker=7, label='Negative Score')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.legend()
    plt.tight_layout()
    return fig


all_score_fig = fig_score_process(pos_score_all, neg_score_all)
all_score_fig.savefig(outplot_path + 'Pos_Neg_Score.png', dpi=300)

plt.figure(figsize=(15, 7))
plt.scatter(x=sentiment_intraday['Sentiment'].index, y=sentiment_intraday['Sentiment'].values, s=3, c='b')

t = sentiment_all['sentiment']

plt.plot(sentiment_all['sentiment'], linestyle='--', color='b')
