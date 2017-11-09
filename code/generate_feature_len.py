
# coding: utf-8


import pandas as pd
import numpy as np
import nltk


df_train = pd.read_csv('../../input/train.csv')
df_test = pd.read_csv('../../input/test.csv')
len_train = df_train.shape[0]


df_feat = pd.DataFrame()
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)


# generate len features
df_feat['char_len1'] = df_data.question1.map(lambda x: len(str(x)))
df_feat['char_len2'] = df_data.question2.map(lambda x: len(str(x)))
df_feat['word_len1'] = df_data.question1.map(lambda x: len(str(x).split()))
df_feat['word_len2'] = df_data.question2.map(lambda x: len(str(x).split()))


df_feat['char_len_diff_ratio'] = df_feat.apply(
    lambda row: abs(row.char_len1-row.char_len2)/(row.char_len1+row.char_len2), axis=1)
df_feat['word_len_diff_ratio'] = df_feat.apply(
    lambda row: abs(row.word_len1-row.word_len2)/(row.word_len1+row.word_len2), axis=1)


df_feat[:len_train].to_csv('train_feature_len.csv', index=False)
df_feat[len_train:].to_csv('test_feature_len.csv', index=False)




