from Code.GlobalParams import *
import pandas as pd
import datetime as dt
import textract
import os


def read_nasdaq(file_name=None):
    if file_name is None:
        nasdaq_news_file = text_data_path + 'NASDAQ_News/nasdaq_news_20190604.csv'
    else:
        nasdaq_news_file = text_data_path + f'NASDAQ_News/{file_name}.csv'
    nasdaq_news = pd.read_csv(nasdaq_news_file)
    return nasdaq_news


# Read Text data
def read_analyst_reports():
    reports_dir = data_dir + 'AnalystReports/'
    all_reps = [file for file in os.listdir(reports_dir) if file.endswith('pdf')]

    reps = list()

    for i, file_name in enumerate(all_reps):
        # reps_temp = list()
        print(i)
        print(file_name)
        date = dt.datetime.strptime(file_name[0:8], '%d%m%Y')
        analyst_comp = file_name[8:11]
        ratingcurr = file_name[11:14]
        ratingprev = file_name[16:19]
        title = file_name[19:].split('.pdf')[0]
        text = textract.process(reports_dir + f"{file_name}").decode('utf_8')
        with open(reports_dir + f'{date.date()}_{title}.csv', 'w') as text_file:
            text_file.write(text)
        reps_temp = [date, analyst_comp, ratingcurr, ratingprev, title, text]
        reps.append(reps_temp)

    reps_df = pd.DataFrame(reps, columns=['Date', 'AnalystComp', 'Rating_curr', 'Rating_prev', 'Title', 'Text'])
    return reps_df
