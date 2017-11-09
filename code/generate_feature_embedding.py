
# coding: utf-8

import pandas as pd
import numpy as np
import gensim
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from gensim.models.wrappers import FastText


df_train = pd.read_csv('../../input/train.csv')
df_test = pd.read_csv('../../input/test.csv')
len_train = df_train.shape[0]


df_feat = pd.DataFrame()
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)


# google news wmd
model = gensim.models.KeyedVectors.load_word2vec_format('../../corpora/GoogleNews-vectors-negative300.bin', binary=True)

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

df_feat['google_news_wmd'] = df_data.apply(lambda row: wmd(row['question1'], row['question2']), axis=1)

norm_model = gensim.models.KeyedVectors.load_word2vec_format('../../corpora/GoogleNews-vectors-negative300.bin', binary=True)
norm_model.init_sims(replace=True)

def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)

df_feat['google_news_norm_wmd'] = df_data.apply(lambda row: norm_wmd(row['question1'], row['question2']), axis=1)


# glove wmd
model = gensim.models.KeyedVectors.load_word2vec_format('../../corpora/glove_model.txt', binary=False)

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

df_feat['glove_wmd'] = df_data.apply(lambda row: wmd(row['question1'], row['question2']), axis=1)

norm_model = gensim.models.KeyedVectors.load_word2vec_format('../../corpora/glove_model.txt', binary=False)
norm_model.init_sims(replace=True)

def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)

df_feat['glove_norm_wmd'] = df_data.apply(lambda row: norm_wmd(row['question1'], row['question2']), axis=1)


# google news w2v distance
model = gensim.models.KeyedVectors.load_word2vec_format('../../corpora/GoogleNews-vectors-negative300.bin', binary=True)
stop_words = stopwords.words('english')

def sent2vec(s):
    words = str(s).lower()
    words = nltk.word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

question1_vectors = np.zeros((df_data.shape[0], 300))
for i, q in enumerate(df_data.question1.values):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((df_data.shape[0], 300))
for i, q in enumerate(df_data.question2.values):
    question2_vectors[i, :] = sent2vec(q)

df_feat['google_news_cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['google_news_cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['google_news_jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['google_news_canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['google_news_euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['google_news_minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['google_news_braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['google_news_skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
df_feat['google_news_skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
df_feat['google_news_kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
df_feat['google_news_kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


# glove w2v distance
model = gensim.models.KeyedVectors.load_word2vec_format('../../corpora/glove_model.txt', binary=False)
def sent2vec(s):
    words = str(s).lower()
    words = nltk.word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

question1_vectors = np.zeros((df_data.shape[0], 300))
for i, q in enumerate(df_data.question1.values):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((df_data.shape[0], 300))
for i, q in enumerate(df_data.question2.values):
    question2_vectors[i, :] = sent2vec(q)

df_feat['glove_cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['glove_cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['glove_jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['glove_canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['glove_euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['glove_minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['glove_braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['glove_skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
df_feat['glove_skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
df_feat['glove_kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
df_feat['glove_kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


# fasttext w2v distance
model = FastText.load_word2vec_format('../../corpora/wiki.en.vec')
def sent2vec(s):
    words = str(s).lower()
    words = nltk.word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

question1_vectors = np.zeros((df_data.shape[0], 300))
for i, q in enumerate(df_data.question1.values):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((df_data.shape[0], 300))
for i, q in enumerate(df_data.question2.values):
    question2_vectors[i, :] = sent2vec(q)

df_feat['fasttext_cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['fasttext_cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['fasttext_jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['fasttext_canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['fasttext_euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['fasttext_minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['fasttext_braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_feat['fasttext_skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
df_feat['fasttext_skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
df_feat['fasttext_kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
df_feat['fasttext_kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


df_feat[:len_train].to_csv('train_feature_embedding.csv', index=False)
df_feat[len_train:].to_csv('test_feature_embedding.csv', index=False)

