
# coding: utf-8


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords


df_train = pd.read_csv('../../input/train.csv')
df_test = pd.read_csv('../../input/test.csv')


len_train = df_train.shape[0]


df_feat = pd.DataFrame()
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)


# get question nouns
df_data['question1_nouns'] = df_data.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
df_data['question2_nouns'] = df_data.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])


# generate noun match features
noun_match_count = []
noun_match_ratio = []

for index, row in df_data.iterrows():
    count = sum([1 for w in row.question1_nouns if w in row.question2_nouns])
    ratio = count / (len(row.question1_nouns) + len(row.question2_nouns))
    noun_match_count.append(count)
    noun_match_ratio.append(ratio)


df_feat['noun_match_count'] = noun_match_count
df_feat['noun_match_ratio'] = noun_match_ratio


stops = set(stopwords.words('english'))
def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


df_feat['word_match_ratio'] = df_data.apply(word_match_share, axis=1)


# Unigrams without stops
def get_unigrams(que):
    return [word for word in nltk.word_tokenize(que.lower()) if word not in stops]
df_data["unigrams_nostop_q1"] = df_data['question1'].apply(lambda x: get_unigrams(str(x)))
df_data["unigrams_nostop_q2"] = df_data['question2'].apply(lambda x: get_unigrams(str(x)))

def get_word_match_nostop_unigrams_count(row):
    return len(set(row["unigrams_nostop_q1"]).intersection(set(row["unigrams_nostop_q2"])))

def get_word_match_nostop_unigrams_ratio(row):
    return float(row["unigrams_nostop_match_count"])/max(len(set(row["unigrams_nostop_q1"]).union(set(row["unigrams_nostop_q2"]))),1)

unigrams_nostop_match_count = []
unigrams_nostop_match_ratio = []

for index, row in df_data.iterrows():
    count = len(set(row["unigrams_nostop_q1"]).intersection(set(row["unigrams_nostop_q2"])))
    ratio = float(count) / max(len(set(row["unigrams_nostop_q1"]).union(set(row["unigrams_nostop_q2"]))),1)
    unigrams_nostop_match_count.append(count)
    unigrams_nostop_match_ratio.append(ratio)

df_feat["unigrams_nostop_match_count"] = unigrams_nostop_match_count
df_feat["unigrams_nostop_match_ratio"] = unigrams_nostop_match_ratio


# Bigrams without stops
def get_bigrams(que):
    return [i for i in nltk.ngrams(que, 2)]
df_data["bigrams_nostop_q1"] = df_data["unigrams_nostop_q1"].apply(lambda x: get_bigrams(x))
df_data["bigrams_nostop_q2"] = df_data["unigrams_nostop_q2"].apply(lambda x: get_bigrams(x))

bigrams_nostop_match_count = []
bigrams_nostop_match_ratio = []

for index, row in df_data.iterrows():
    count = len(set(row["bigrams_nostop_q1"]).intersection(set(row["bigrams_nostop_q2"])))
    ratio = float(count) / max(len(set(row["bigrams_nostop_q1"]).union(set(row["bigrams_nostop_q2"]))),1)
    bigrams_nostop_match_count.append(count)
    bigrams_nostop_match_ratio.append(ratio)

df_feat["bigrams_nostop_match_count"] = bigrams_nostop_match_count
df_feat["bigrams_nostop_match_ratio"] = bigrams_nostop_match_ratio


# Trigrams without stops
def get_trigrams(que):
    return [i for i in nltk.ngrams(que, 3)]
df_data["trigrams_nostop_q1"] = df_data["unigrams_nostop_q1"].apply(lambda x: get_trigrams(x))
df_data["trigrams_nostop_q2"] = df_data["unigrams_nostop_q2"].apply(lambda x: get_trigrams(x)) 

trigrams_nostop_match_count = []
trigrams_nostop_match_ratio = []

for index, row in df_data.iterrows():
    count = len(set(row["trigrams_nostop_q1"]).intersection(set(row["trigrams_nostop_q2"])))
    ratio = float(count) / max(len(set(row["trigrams_nostop_q1"]).union(set(row["trigrams_nostop_q2"]))),1)
    trigrams_nostop_match_count.append(count)
    trigrams_nostop_match_ratio.append(ratio)

df_feat["trigrams_nostop_match_count"] = trigrams_nostop_match_count
df_feat["trigrams_nostop_match_ratio"] = trigrams_nostop_match_ratio


df_data[:len_train].to_csv('train_feature_match.csv', index=False)
df_data[len_train:].to_csv('test_feature_match.csv', index=False)




