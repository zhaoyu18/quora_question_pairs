
# coding: utf-8


import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords



df_train = pd.read_csv('../../input/train.csv')
df_test  = pd.read_csv('../../input/test.csv')


def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


def add_word_count(df_feature, df, word):
    df_feature['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
    df_feature['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
    df_feature[word + '_both'] = df_feature['q1_' + word] * df_feature['q2_' + word]


stops = set(stopwords.words("english"))

def word_shares(row):
    q1_list = str(row['question1']).lower().split()
    q1 = set(q1_list)
    q1words = q1.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0:0:0:0'

    q2_list = str(row['question2']).lower().split()
    q2 = set(q2_list)
    q2words = q2.difference(stops)
    if len(q2words) == 0:
        return '0:0:0:0:0:0:0:0'

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q2_weights

    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)) #count share
    R31 = len(q1stops) / len(q1words) #stops in q1
    R32 = len(q2stops) / len(q2words) #stops in q2
    Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))
    Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)

df = pd.concat([df_train, df_test])
df['word_shares'] = df.apply(word_shares, axis=1, raw=True)


df_feature = pd.DataFrame()

df_feature['word_match_ratio_2']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
df_feature['word_match_ratio_2_root'] = np.sqrt(df_feature['word_match_ratio_2'])
df_feature['tfidf_word_match_ratio_2'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
df_feature['shared_count_2']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

df_feature['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
df_feature['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
df_feature['shared_2gram']     = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
df_feature['word_match_cosine']= df['word_shares'].apply(lambda x: float(x.split(':')[6]))
df_feature['words_hamming']    = df['word_shares'].apply(lambda x: float(x.split(':')[7]))
df_feature['diff_stops_r']     = df_feature['stops1_ratio'] - df_feature['stops2_ratio']

df_feature['caps_count_q1'] = df['question1'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
df_feature['caps_count_q2'] = df['question2'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
df_feature['diff_caps'] = df_feature['caps_count_q1'] - df_feature['caps_count_q2']

df['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
df['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))

df['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
df['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))

df_feature['avg_world_len1'] = df['len_char_q1'] / df['len_word_q1']
df_feature['avg_world_len2'] = df['len_char_q2'] / df['len_word_q2']
df_feature['diff_avg_word'] = df_feature['avg_world_len1'] - df_feature['avg_world_len2']

add_word_count(df_feature, df,'how')
add_word_count(df_feature, df,'what')
add_word_count(df_feature, df,'which')
add_word_count(df_feature, df,'who')
add_word_count(df_feature, df,'where')
add_word_count(df_feature, df,'when')
add_word_count(df_feature, df,'why')


df_feature[:df_train.shape[0]].to_csv('train_feature_match_2.csv', index=False)
df_feature[df_train.shape[0]:].to_csv('test_feature_match_2.csv', index=False)




