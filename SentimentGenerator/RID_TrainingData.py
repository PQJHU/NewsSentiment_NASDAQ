"""
Read Phrase bank data and EU News Headline Annotation

Tokenize and lemmatize the training data

Available to choose the classification criterion

"""
from Code.GlobalParams import *
import os
import pandas as pd
from Code.SentimentGenerator.CAL_Lemmatization import docs_lemma_spacy, docs_lemma_nltk

phrase_bank_dir = data_dir + 'PhraseBank/'
eu_headline_dir = data_dir + 'semeval-2017-task-5-subtask-2/'


def read_phrase_bank(agg_percent, refresh, lemma_model='nltk'):
    # all_files = [file for file in os.listdir(phrase_bank_dir) if file.startswith('Sentences')]
    if refresh or not os.path.exists(phrase_bank_dir + f'Sentences_{agg_percent}Agree.csv'):
        if agg_percent in ('50', '66', '75', 'All'):
            phrase_bank_data = pd.read_csv(phrase_bank_dir + f'Sentences_{agg_percent}Agree.txt', sep='@', header=None,
                                           encoding='ISO-8859-2')
        else:
            raise KeyError("Agg Percentage shall be in ['50', '66', '75', 'All']")

        phrase_bank_data.columns = ['Phrase', 'Polarity']
        phrase_bank_data.loc[phrase_bank_data['Polarity'] == 'positive', 'Polarity'] = 1
        phrase_bank_data.loc[phrase_bank_data['Polarity'] == 'negative', 'Polarity'] = -1
        phrase_bank_data.loc[phrase_bank_data['Polarity'] == 'neutral', 'Polarity'] = 0

        stats = phrase_bank_data['Polarity'].value_counts()

        print(f'# Positive Sentence: {stats[1]}')
        print(f'# Negative Sentence: {stats[-1]}')
        print(f'# Neutral Sentence: {stats[0]}')
        if lemma_model == 'spacy':
            phrase_bank_data['Phrase'] = docs_lemma_spacy(docs=phrase_bank_data['Phrase'].values)
        elif lemma_model == 'nltk':
            phrase_bank_data['Phrase'] = docs_lemma_nltk(docs=phrase_bank_data['Phrase'].values)
        else:
            raise KeyError("Input has to be 'nltk' or 'spacy'! ")
        phrase_bank_data.to_csv(phrase_bank_dir + f'Sentences_{agg_percent}Agree.csv')
    else:
        phrase_bank_data = pd.read_csv(phrase_bank_dir + f'Sentences_{agg_percent}Agree.csv', index_col=0)
    # docs = phrase_bank_data['Phrase'].values
    # sentence = docs[0]
    # sentence = sentence.encode('utf-8').strip()
    return phrase_bank_data


def read_eu_head_line(threshold=0.34, refresh=True, lemma_model='nltk'):
    if refresh or not os.path.exists(eu_headline_dir + f'headline_data_{threshold}.csv'):
        train_data = pd.read_json(eu_headline_dir + 'Headline_Trainingdata.json')
        eu_headline_data = train_data[['title', 'sentiment']]
        eu_headline_data.loc[eu_headline_data['sentiment'] > threshold, ['sentiment']] = 1
        eu_headline_data.loc[eu_headline_data['sentiment'] < -threshold, ['sentiment']] = -1
        eu_headline_data.loc[
            (eu_headline_data['sentiment'] <= threshold) & (eu_headline_data['sentiment'] >= -threshold), [
                'sentiment']] = 0

        eu_headline_data.columns = ['Phrase', 'Polarity']

        stats = eu_headline_data['Polarity'].value_counts()

        print(f'# Positive Sentence: {stats[1]}')
        print(f'# Negative Sentence: {stats[-1]}')
        print(f'# Neutral Sentence: {stats[0]}')

        if lemma_model == 'spacy':
            eu_headline_data['Phrase'] = docs_lemma_spacy(docs=eu_headline_data['Phrase'].values)
        elif lemma_model == 'nltk':
            eu_headline_data['Phrase'] = docs_lemma_nltk(docs=eu_headline_data['Phrase'].values)
        else:
            raise KeyError("Input has to be 'nltk' or 'spacy'! ")

        eu_headline_data['Phrase'] = docs_lemma_spacy(docs=eu_headline_data['Phrase'].values)
        eu_headline_data.to_csv(eu_headline_dir + f'headline_data_{threshold}.csv')

    else:
        eu_headline_data = pd.read_csv(eu_headline_dir + f'headline_data_{threshold}.csv', index_col=0)

    return eu_headline_data
