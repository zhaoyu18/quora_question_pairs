
# coding: utf-8


import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.corpus import stopwords


train_orig =  pd.read_csv('../../input/train.csv', header=0)
test_orig =  pd.read_csv('../../input/test.csv', header=0)

ques = pd.concat([train_orig[['question1', 'question2']], test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')


stops = set(stopwords.words("english"))
def word_match_share(q1, q2, stops=None):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stops:
            q1words[word] = 1
    for word in q2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


q_dict = defaultdict(dict)
for i in range(ques.shape[0]):
        wm = word_match_share(ques.question1[i], ques.question2[i], stops=stops)
        q_dict[ques.question1[i]][ques.question2[i]] = wm
        q_dict[ques.question2[i]][ques.question1[i]] = wm


def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))
def q1_q2_wm_ratio(row):
    q1 = q_dict[row['question1']]
    q2 = q_dict[row['question2']]
    inter_keys = set(q1.keys()).intersection(set(q2.keys()))
    if(len(inter_keys) == 0): return 0.
    inter_wm = 0.
    total_wm = 0.
    for q,wm in q1.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    for q,wm in q2.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    if(total_wm == 0.): return 0.
    return inter_wm/total_wm


train_orig['q1_q2_wm_ratio'] = train_orig.apply(q1_q2_wm_ratio, axis=1, raw=True)
test_orig['q1_q2_wm_ratio'] = test_orig.apply(q1_q2_wm_ratio, axis=1, raw=True)

train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

train_feat = train_orig[['q1_q2_intersect', 'q1_q2_wm_ratio']]
test_feat = test_orig[['q1_q2_intersect', 'q1_q2_wm_ratio']]

train_feat.to_csv('train_feature_graph_intersect.csv', index=False)
test_feat.to_csv('test_feature_graph_intersect.csv', index=False)

