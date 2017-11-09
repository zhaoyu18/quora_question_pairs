
# coding: utf-8

import pandas as pd
import numpy as np
import difflib

df_train = pd.read_csv('../../input/train.csv')
df_test = pd.read_csv('../../input/test.csv')

len_train = df_train.shape[0]

df_feat = pd.DataFrame()
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)

def diff_ratios(row):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(row.question1).lower(), str(row.question2).lower())
    return seq.ratio()

df_feat['diff_ratios'] = df_data.apply(diff_ratios, axis=1)

df_feat[:len_train].to_csv('train_feature_difflib.csv', index=False)
df_feat[len_train:].to_csv('test_feature_difflib.csv', index=False)
