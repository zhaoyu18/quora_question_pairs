
# coding: utf-8


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import itertools as it
import pickle
import glob
import os
import string
from scipy import sparse
import nltk
import spacy
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, make_scorer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize
import eli5



df_train = pd.read_csv('./../../input/train.csv', dtype={'question1': np.str, 'question2': np.str})
df_train['test_id'] = -1
df_test = pd.read_csv('./../../input/test.csv', dtype={'question1': np.str, 'question2': np.str})
df_test['id'] = -1
df_test['qid1'] = -1
df_test['qid2'] = -1
df_test['is_duplicate'] = -1
df = pd.concat([df_train, df_test])
df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')
df['uid'] = np.arange(df.shape[0])
df = df.set_index(['uid'])
del(df_train, df_test)



ix_train = np.where(df['id'] >= 0)[0]
ix_test = np.where(df['id'] == -1)[0]
ix_is_dup = np.where(df['is_duplicate'] == 1)[0]
ix_not_dup = np.where(df['is_duplicate'] == 0)[0]



df['len1'] = df['question1'].str.len().astype(np.float32)
df['len2'] = df['question2'].str.len().astype(np.float32)
df['abs_diff_len1_len2'] = np.abs(df['len1'] - df['len2'])



max_in_dup = df.loc[ix_is_dup]['abs_diff_len1_len2'].max()
max_in_not_dups = df.loc[ix_not_dup]['abs_diff_len1_len2'].max()
std_in_dups = df.loc[ix_is_dup]['abs_diff_len1_len2'].std()
replace_value = max_in_dup + 2*std_in_dups

df['abs_diff_len1_len2'] = df['abs_diff_len1_len2'].apply(lambda x: x if x < replace_value else replace_value)
df['log_abs_diff_len1_len2'] = np.log(df['abs_diff_len1_len2'] + 1)
df['ratio_len1_len2'] = df['len1'].apply(lambda x: x if x > 0.0 else 1.0)/df['len2'].apply(lambda x: x if x > 0.0 else 1.0)

max_in_dup = df.loc[ix_is_dup]['ratio_len1_len2'].max()
max_in_not_dups = df.loc[ix_not_dup]['ratio_len1_len2'].max()
std_in_dups = df.loc[ix_is_dup]['ratio_len1_len2'].std()
replace_value = max_in_dup + 2*std_in_dups

df['ratio_len1_len2'] = df['ratio_len1_len2'].apply(lambda x: x if x < replace_value else replace_value)
df['log_ratio_len1_len2'] = np.log(df['ratio_len1_len2'] + 1)



cv_char = CountVectorizer(ngram_range=(1, 3), analyzer='char')
ch_freq = np.array(cv_char.fit_transform(df['question1'].tolist() + df['question2'].tolist()).sum(axis=0))[0, :]


unigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 1])
ix_unigrams = np.sort(list(unigrams.values()))

bigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 2])
ix_bigrams = np.sort(list(bigrams.values()))

trigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 3])
ix_trigrams = np.sort(list(trigrams.values()))


m_q1 = cv_char.transform(df['question1'].values)
m_q2 = cv_char.transform(df['question2'].values)

v_num = (m_q1[:, ix_unigrams] > 0).minimum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
v_den = (m_q1[:, ix_unigrams] > 0).maximum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['unigram_jaccard'] = v_score

v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
v_den = m_q1[:, ix_unigrams].sum(axis=1) + m_q2[:, ix_unigrams].sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['unigram_all_jaccard'] = v_score

v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
v_den = m_q1[:, ix_unigrams].maximum(m_q2[:, ix_unigrams]).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['unigram_all_jaccard_max'] = v_score


v_num = (m_q1[:, ix_bigrams] > 0).minimum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
v_den = (m_q1[:, ix_bigrams] > 0).maximum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['bigram_jaccard'] = v_score

print( 'Number of right speakers on the right:', (df['bigram_jaccard'] > 1).sum())
print( 'Number of outsiders on the left: ', (df['bigram_jaccard'] < -1.47).sum())
df.loc[df['bigram_jaccard'] < -1.478751, 'bigram_jaccard'] = -1.478751
df.loc[df['bigram_jaccard'] > 1.0, 'bigram_jaccard'] = 1.0

v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
v_den = m_q1[:, ix_bigrams].sum(axis=1) + m_q2[:, ix_bigrams].sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['bigram_all_jaccard'] = v_score

v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
v_den = m_q1[:, ix_bigrams].maximum(m_q2[:, ix_bigrams]).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['bigram_all_jaccard_max'] = v_score


m_q1 = m_q1[:, ix_trigrams]
m_q2 = m_q2[:, ix_trigrams]

v_num = (m_q1 > 0).minimum((m_q2 > 0)).sum(axis=1)
v_den = (m_q1 > 0).maximum((m_q2 > 0)).sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['trigram_jaccard'] = v_score

v_num = m_q1.minimum(m_q2).sum(axis=1)
v_den = m_q1.sum(axis=1) + m_q2.sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['trigram_all_jaccard'] = v_score

v_num = m_q1.minimum(m_q2).sum(axis=1)
v_den = m_q1.maximum(m_q2).sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
df['trigram_all_jaccard_max'] = v_score


tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] * np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
v_num[np.where(v_den == 0)] = 1
v_den[np.where(v_den == 0)] = 1
v_score = 1 - v_num/v_den
df['trigram_tfidf_cosine'] = v_score

tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])
df['trigram_tfidf_l2_euclidean'] = v_score

tft = TfidfTransformer(
    norm='l1', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])
df['trigram_tfidf_l1_euclidean'] = v_score

tft = TfidfTransformer(
    norm='l2', 
    use_idf=False, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])
df['trigram_tf_l2_euclidean'] = v_score



data = {
    'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in tqdm_notebook(folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']), 
                                         total=n_splits):
    # {'en__l1_ratio': 0.0001, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=100, 
        shuffle=True, 
        n_jobs=-1,
        l1_ratio=0.0001,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])

def func(w):
    return (mp*data['y_test_pred'].shape[0] -
            np.sum( w[0]*data['y_test_pred'] / (w[0]*data['y_test_pred'] + (1 - w[0]) * (1 - data['y_test_pred']))) )**2

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_q1_q2_tf_oof'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_q1_q2_tf_oof'] = data['y_train_pred']
df.loc[ix_test, 'm_q1_q2_tf_oof'] = data['y_test_pred_fixed']

del(data)
del(unigrams, bigrams, trigrams)


svd = TruncatedSVD(n_components=100)
m_svd = svd.fit_transform(sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf))))

df['m_q1_q2_tf_svd0'] = m_svd[:, 0]
df['m_q1_q2_tf_svd1'] = m_svd[:, 0]

data={
    'X_train': m_svd[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': m_svd[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in tqdm_notebook(folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']), 
                                         total=n_splits):
    # {'en__l1_ratio': 0.5, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=100, 
        shuffle=True, 
        n_jobs=-1,
        l1_ratio=0.5,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])

def func(w):
    return ( mp*data['y_test_pred'].shape[0] - 
             np.sum(w[0]*data['y_test_pred'] / (w[0]*data['y_test_pred'] + (1 - w[0]) * (1 - data['y_test_pred']))) )**2

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_q1_q2_tf_svd100_oof'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_q1_q2_tf_svd100_oof'] = data['y_train_pred']
df.loc[ix_test, 'm_q1_q2_tf_svd100_oof'] = data['y_test_pred_fixed']

del(data)
del(m_q1, m_q2, m_svd)


m_diff_q1_q2 = m_q1_tf - m_q2_tf

data={
    'X_train': m_diff_q1_q2[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': m_diff_q1_q2[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in tqdm_notebook(folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']), 
                                         total=n_splits):
    # {'en__l1_ratio': 0.01, 'en__alpha': 0.001}
    model = SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=100, 
        shuffle=True, 
        n_jobs=-1,
        l1_ratio=0.01,
        alpha=0.001,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

df['m_diff_q1_q2_tf_oof'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_diff_q1_q2_tf_oof'] = data['y_train_pred']
df.loc[ix_test, 'm_diff_q1_q2_tf_oof'] = data['y_test_pred']

del(data)
del(m_diff_q1_q2)


svd = TruncatedSVD(n_components=100)
m_svd = svd.fit_transform(sparse.csc_matrix(sparse.vstack((m_q1_tf, m_q2_tf))))
del(m_q1_tf, m_q2_tf)

m_svd_q1 = m_svd[:m_svd.shape[0]//2, :]
m_svd_q2 = m_svd[m_svd.shape[0]//2:, :]
del(m_svd)

df['m_vstack_svd_q1_q1_euclidean'] = ((m_svd_q1 - m_svd_q2)**2).mean(axis=1)

num = (m_svd_q1*m_svd_q2).sum(axis=1)
den = np.sqrt((m_svd_q1**2).sum(axis=1))*np.sqrt((m_svd_q2**2).sum(axis=1))
num[np.where(den == 0)] = 0
den[np.where(den == 0)] = 1
df['m_vstack_svd_q1_q1_cosine'] = 1 - num/den


m_svd = m_svd_q1*m_svd_q2

data={
    'X_train': m_svd[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': m_svd[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}
del(m_svd)

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in tqdm_notebook(folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']), 
                                         total=n_splits):
    # {'en__l1_ratio': 1, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=100, 
        shuffle=True, 
        n_jobs=-1,
        l1_ratio=1.0,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])

def func(w):
    return ( mp*data['y_test_pred'].shape[0] - 
             np.sum(w[0]*data['y_test_pred'] / (w[0]*data['y_test_pred'] + (1 - w[0]) * (1 - data['y_test_pred']))) )**2

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_vstack_svd_mult_q1_q2_oof'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_vstack_svd_mult_q1_q2_oof'] = data['y_train_pred']
df.loc[ix_test, 'm_vstack_svd_mult_q1_q2_oof'] = data['y_test_pred_fixed']

del(data)


m_svd = np.abs(m_svd_q1 - m_svd_q2)

data={
    'X_train': m_svd[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': m_svd[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}
del(m_svd, m_svd_q1, m_svd_q2)

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in tqdm_notebook(folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']), 
                                         total=n_splits):
    # {'en__l1_ratio': 0.01, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=100, 
        shuffle=True, 
        n_jobs=-1,
        l1_ratio=0.01,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])

def func(w):
    return (
        mp*data['y_test_pred'].shape[0] - 
        np.sum(w[0]*data['y_test_pred'] / (w[0]*data['y_test_pred'] + (1 - w[0]) * (1 - data['y_test_pred'])))
    )**2

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_vstack_svd_absdiff_q1_q2_oof'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_vstack_svd_absdiff_q1_q2_oof'] = data['y_train_pred']
df.loc[ix_test, 'm_vstack_svd_absdiff_q1_q2_oof'] = data['y_test_pred']

del(data)


nlp = spacy.load('en')
SYMBOLS = set(' '.join(string.punctuation).split(' ') + ['...', '“', '”', '\'ve'])

q1 = []

for doc in nlp.pipe(df['question1'].fillna(""), n_threads=16, batch_size=10000):
    q1.append([c.lemma_ for c in doc if c.lemma_ not in SYMBOLS])

q2 = []

for doc in nlp.pipe(df['question2'].fillna(""), n_threads=16, batch_size=10000):
    q2.append([c.lemma_ for c in doc if c.lemma_ not in SYMBOLS])
    
cv_words = CountVectorizer(ngram_range=(1, 1), analyzer='word')
w_freq = np.array(cv_words.fit_transform(
    [' '.join(s) for s in q1] + [' '.join(s) for s in q2]).sum(axis=0))[0, :]

m_q1 = cv_words.transform([' '.join(s) for s in q1])
m_q2 = cv_words.transform([' '.join(s) for s in q2])


tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] *         np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
v_num[np.where(v_den == 0)] = 1
v_den[np.where(v_den == 0)] = 1

v_score = 1 - v_num/v_den

df['1wl_tfidf_cosine'] = v_score


tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['1wl_tfidf_l2_euclidean'] = v_score


tft = TfidfTransformer(
    norm='l2', 
    use_idf=False, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['1wl_tf_l2_euclidean'] = v_score


tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

data={
    'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}
del(m_q1_tf, m_q2_tf)

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in tqdm_notebook(folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']), 
                                         total=n_splits):
    # {'en__l1_ratio': 0.0001, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=100, 
        shuffle=True, 
        n_jobs=-1,
        l1_ratio=0.0001,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])

def func(w):
    return (
        mp*data['y_test_pred'].shape[0] - 
        np.sum(w[0]*data['y_test_pred'] / (w[0]*data['y_test_pred'] + (1 - w[0]) * (1 - data['y_test_pred'])))
    )**2

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_w1l_tfidf_oof'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_w1l_tfidf_oof'] = data['y_train_pred']
df.loc[ix_test, 'm_w1l_tfidf_oof'] = data['y_test_pred_fixed']
del(data)


df[:404290].to_csv('train_feature_oof.csv', index=False)
df[404290:].to_csv('test_feature_oof.csv', index=False)


