import pandas as pd
from Code.SentimentGenerator.RID_TrainingData import read_phrase_bank, read_eu_head_line
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import numpy as np
import pickle
from Code.GlobalParams import *

"""
Train Linear-SVC and LSTM model on Malo et al(2014) Phrasebank dataset and EU News Headline Annotation
"""

# Read training dataset
seed = np.random.seed(1124)
sw = stopwords.words('english')

phrase_bank =       read_phrase_bank(agg_percent='66', refresh=False, lemma_model='nltk')
eu_headline = read_eu_head_line(threshold=0.34, refresh=False, lemma_model='nltk')
train_data = pd.concat([phrase_bank, eu_headline], axis=0, ignore_index=True)


def balanced_subsample(pre_balance_x, pre_balance_y, subsample_size_coef=1.0, balance_seed=None):
    """
    Create a balanced subsample with upsampling.
    :param pre_balance_x: Sample X, pre-balance
    :param pre_balance_y: Sample y, pre-balance
    :param subsample_size_coef: Choosing the coefficient to define max number of oversampling
    :param balance_seed: for random module
    :return: DataFrame, size = (max_elems * subsample_size_coef * 3, 2), includes pos, neutral, negative sentiment selected_sample
    """
    X_col = pre_balance_x.name
    y_col = pre_balance_y.name
    # Create a DataFrame to contain the samples of each fold to do the oversampling
    pre_balance_data = pd.concat([pre_balance_x, pre_balance_y], axis=1)
    max_elems = pre_balance_y.value_counts().values.max()
    sample_out_data = pd.DataFrame()

    for val in pre_balance_y.unique():
        selected_sample = pre_balance_data.loc[pre_balance_data[y_col] == val, :]
        # print(selected_sample.sentiment.unique())
        num_dup = int(max_elems * subsample_size_coef - len(selected_sample))
        # print(num_dup)
        dup_samples = selected_sample.sample(n=num_dup, random_state=balance_seed, axis=0, replace=True)
        # print(len(dup_samples))
        print(dup_samples)
        sample_out_data = pd.concat([sample_out_data, dup_samples], axis=0, ignore_index=True)

    balanced_data = pd.concat([pre_balance_data, sample_out_data], axis=0, ignore_index=True)
    # count_ = balanced_data.sentiment.value_counts()  # check 3 sentiments have the same size after oversampling
    X_balanced = balanced_data[X_col]
    y_balanced = balanced_data[y_col]
    return X_balanced, y_balanced


# Define Hyperparameters

def linear_models_calibration():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(n_jobs=-1, verbose=1, early_stopping=True, n_iter_no_change=10, random_state=seed))
    ])

    hyper_parameters = {
        # vect parameters
        'vect__stop_words': (None, 'english'),
        'vect__max_df': (0.75, 0.9),
        'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'vect__max_features': (None, 10000, 50000),

        'tfidf__sublinear_tf': (True, False),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),

        'clf__loss': ('hinge', 'modified_huber', 'squared_hinge'),
        'clf__penalty': ('none', 'l2', 'l1', 'elasticnet'),
        'clf__alpha': (0.00005, 0.0001, 0.0005, 0.001, 0.005),
        'clf__max_iter': (500, 1000),
    }
    return pipeline, hyper_parameters


def nonlinear_models_calibration():
    pipeline = Pipeline([
        ('vect')
    ])
    pass


X_train, y_train = balanced_subsample(pre_balance_x=train_data['Phrase'], pre_balance_y=train_data['Polarity'],
                                      balance_seed=seed)
pipeline, hyperparams = linear_models_calibration()
grid_search = GridSearchCV(pipeline, hyperparams, cv=10, n_jobs=-1, iid=True, refit=True, verbose=1)
grid_search.fit(X=X_train, y=y_train)

best_parameters = grid_search.best_estimator_.get_params()
best_estimator = grid_search.best_estimator_

for param_name in sorted(best_parameters.keys()):
    print(f'{param_name}: {best_parameters[param_name]}')

with open(outmodel_path + 'grid_search.pkl', 'wb') as grid_file:
    pickle.dump(grid_search, grid_file)


with open(outmodel_path + 'LinearClfModel.pkl', 'wb') as model_file:
    pickle.dump(best_estimator, model_file)

with open(outmodel_path + 'LinearClfModel.pkl', 'rb') as mf:
    best_model_load = pickle.load(mf)


from sklearn.metrics.classification import classification_report, confusion_matrix

X_origin = train_data['Phrase']
y_origin = train_data['Polarity']
best_pred_original = best_model_load.predict(X_origin)

clf_report = classification_report(y_true=y_origin, y_pred=best_pred_original)
print(clf_report)

conf_matrix = confusion_matrix(y_true=y_origin, y_pred=best_pred_original, labels=[-1, 0 , 1])



"""
              precision    recall  f1-score   support
        -1.0       0.98      1.00      0.99      3180
         0.0       0.99      0.98      0.98      3180
         1.0       0.98      0.98      0.98      3180
    accuracy                           0.98      9540
   macro avg       0.98      0.98      0.98      9540
weighted avg       0.98      0.98      0.98      9540

"""
