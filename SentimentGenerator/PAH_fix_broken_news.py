from Code.GlobalParams import *
import pandas as pd

nasdaq_score = pd.read_csv(outdata_path + 'svm_sentiment_score_clean.csv', index_col='Unnamed: 0')
text_info = nasdaq_score.loc[:, nasdaq_score.columns^['Pos', 'Neutral', 'Neg']]
score = nasdaq_score.loc[:, ['Pos', 'Neutral', 'Neg']]
score.dropna(axis=0, inplace=True)

drop_idx = [144871, 806619, 868452, 898592, 968881, 1070525]

text_info.drop(drop_idx, axis=0, inplace=True)
text_info.reset_index(drop=True, inplace=True)

nasdaq_score_clean = pd.concat([text_info, score], axis=1)
nasdaq_score_clean.index.name = 'idx'
nasdaq_score_clean.to_csv(outdata_path + 'svm_sentiment_score_clean.csv')
