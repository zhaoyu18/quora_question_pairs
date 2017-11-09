
# coding: utf-8

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

df_train = pd.read_csv('../../input/train.csv')
df_test = pd.read_csv('../../input/test.csv')
len_train = df_train.shape[0]

df_feat = pd.DataFrame()
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)

df_feat['fuzz_qratio'] = df_data.apply(lambda row: fuzz.QRatio(str(row['question1']), str(row['question2'])), axis=1)
df_feat['fuzz_WRatio'] = df_data.apply(lambda row: fuzz.WRatio(str(row['question1']), str(row['question2'])), axis=1)
df_feat['fuzz_partial_ratio'] = df_data.apply(lambda row: fuzz.partial_ratio(str(row['question1']), str(row['question2'])), axis=1)
df_feat['fuzz_partial_token_set_ratio'] = df_data.apply(lambda row: fuzz.partial_token_set_ratio(str(row['question1']), str(row['question2'])), axis=1)
df_feat['fuzz_partial_token_sort_ratio'] = df_data.apply(lambda row: fuzz.partial_token_sort_ratio(str(row['question1']), str(row['question2'])), axis=1)
df_feat['fuzz_token_set_ratio'] = df_data.apply(lambda row: fuzz.token_set_ratio(str(row['question1']), str(row['question2'])), axis=1)
df_feat['fuzz_token_sort_ratio'] = df_data.apply(lambda row: fuzz.token_sort_ratio(str(row['question1']), str(row['question2'])), axis=1)

df_feat[:len_train].to_csv('train_feature_fuzz.csv', index=False)
df_feat[len_train:].to_csv('test_feature_fuzz.csv', index=False)
