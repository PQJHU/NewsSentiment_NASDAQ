# Read Packages
from Code.GlobalParams import *
import pandas as pd
import pickle
import numpy as np
import sklearn as skl
import os
from Code.SentimentGenerator.CAL_Lemmatization import docs_lemma_nltk
from Code.SentimentGenerator.RID_ReadData import read_nasdaq, read_analyst_reports
from nltk import tokenize as token

import sys
import textract
import datetime as dt
import re

"""
Read the trained model and predict the polarity for each sentence and aggregate to each article
"""


def sentiment_score_count(article):
    article = article.strip()
    docs = token.sent_tokenize(article)  # split article into sentences
    # docs is a list with each sentence as string
    lemmatized_docs = docs_lemma_nltk(docs=docs)
    docs_clean = [doc for doc in lemmatized_docs if len(doc.split()) >= 10]

    # Fit optimal trained model
    trained_model = list(piper.predict(docs_clean))
    pos_score = trained_model.count(1)
    neu_score = trained_model.count(0)
    neg_score = trained_model.count(-1)
    return pos_score, neu_score, neg_score


# def calculate_all_score(articles_df, article_col_name):
#     # Fit Model
#     articles = articles_df[article_col_name]
#     # article = articles[0]
#     score = list()
#     for num, article in enumerate(articles):
#         print(num)
#         pos, neu, neg = sentiment_score_count(article)
#         score.append([pos, neu, neg])
#
#     score = pd.DataFrame(score, columns=['Pos', 'Neutral', 'Neg'])
#     articles_df = pd.concat([articles_df, score], axis=1)
#     articles_df.drop(labels=[article_col_name], axis=1, inplace=True)
#     return articles_df

def calculate_article_score(num, article):
    try:
        print(num)
        pos, neu, neg = sentiment_score_count(article)
        score = [pos, neu, neg]
        return score

    except Exception as e:
        print(e)
        return [num]


def calculate_all_score(articles_df, content_col_name):
    # Fit Model
    # articles_df = articles_df.loc[144870:144880]
    articles = articles_df[content_col_name]
    # article = articles[0]
    score = list()
    indcs = list()
    # score_list = [calculate_article_score(num, article) for num, article in enumerate(articles)]

    for idx in articles.index:
        print(idx)
        article = articles.loc[idx]
        try:
            pos, neu, neg = sentiment_score_count(article)
            score.append([pos, neu, neg])
            indcs.append(idx)
        except Exception as e:
            print(e)
            with open(outdata_path + 'suspect_article_url.txt', 'a+') as f:
                f.write(f"{idx}: {articles_df.loc[idx]['article_link']}\n")
            articles.drop(idx, axis=0, inplace=True)
            articles_df.drop(idx, axis=0, inplace=True)

    score = pd.DataFrame(score, columns=['Pos', 'Neutral', 'Neg'], index=indcs)
    articles_df = pd.concat([articles_df, score], axis=1)
    # articles_df.drop(labels=[content_col_name], axis=1, inplace=True)
    return articles_df

# reps_df = read_analyst_reports()
# full_df.drop(labels=['Text'], axis=1, inplace=True)

with open(outmodel_path + 'LinearClfModel.pkl', "rb") as model_obj:
    piper = pickle.load(model_obj)

nasdaq_df = read_nasdaq()
nasdaq_full_df = calculate_all_score(articles_df=nasdaq_df, content_col_name='article_content')
nasdaq_full_df.to_csv(outdata_path + 'nasdaq_news/LinearClf_Polarity.csv')
