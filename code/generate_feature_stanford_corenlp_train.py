
# coding: utf-8


import pickle
import numpy as np
import pandas as pd
from dependency_parse import get_pos_dep
from semantic_similarity import align
from features import *
import gzip


len_train = pd.read_csv('../../input/train.csv').shape[0]


with gzip.open('stanford_corenlp_train.nlp', 'rb') as handle:
    rows = []
    for i in range(len_train):
        try:
            d = pickle.load(handle)
            id = d['id']
#             print(id)
            is_duplicate = d['is_duplicate']
            #print(d['q1']['raw'])
            #print(d['q2']['raw'])
            S = get_pos_dep(d['q1']['toks'], d['q1']['deps'])
            T = get_pos_dep(d['q2']['toks'], d['q2']['deps'])
            #print(S)
            #print(T)
            A = align(S,T)
            #print(A)
            #print('Len(S):',len(S))
            #print('Len(T):', len(T))
            #print('Len(A):', len(A))
            # Semantic Similarity Features
            S_sem_sim = percentage_semantic_similarity_one(S, A)
            T_sem_sim = percentage_semantic_similarity_one(T, A)
            sem_sim = percentage_semantic_similarity_both(S, T, A)
            # Noun Features
            S_unmatch_n, T_unmatch_n = number_unmatched(S, T, A, 'n', inferred_pos=True)
            S_unmatch_n_p, T_unmatch_n_p = percent_unmatched(S, T, A, 'n', inferred_pos=True)
            # Adjective Features
            S_unmatch_a, T_unmatch_a = number_unmatched(S, T, A, 'a', inferred_pos=True)
            S_unmatch_a_p, T_unmatch_a_p = percent_unmatched(S, T, A, 'a', inferred_pos=True)
            # Verb Features
            S_unmatch_v, T_unmatch_v = number_unmatched(S, T, A, 'v', inferred_pos=True)
            S_unmatch_v_p, T_unmatch_v_p = percent_unmatched(S, T, A, 'v', inferred_pos=True)
            # Personal Pronoun Feature
            S_unmatch_pp, T_unmatch_pp = number_unmatched(S, T, A, 'PRP', inferred_pos=False)
            S_unmatch_pp_p, T_unmatch_pp_p = percent_unmatched(S, T, A, 'PRP', inferred_pos=False)
            # WH-Pronoun Feature
            S_unmatch_wp, T_unmatch_wp = number_unmatched(S, T, A, 'WP', inferred_pos=False)
            S_unmatch_wp_p, T_unmatch_wp_p = percent_unmatched(S, T, A, 'WP', inferred_pos=False)
            # Numbers Feature
            S_unmatch_num, T_unmatch_num = number_unmatched(S, T, A, 'CD', inferred_pos=False)
            S_unmatch_num_p, T_unmatch_num_p = percent_unmatched(S, T, A, 'CD', inferred_pos=False)
            # NER Feature
            S_unmatch_ner, T_unmatch_ner = ner_unmatched(S, T)
            # Length Difference Feature
            len_dif = len_difference(S, T)
            len_dif_p = len_difference_p(S, T)
            # Put all features in a row
            features_row = [id,
                            S_sem_sim, T_sem_sim, sem_sim,
                            S_unmatch_n, T_unmatch_n, S_unmatch_n_p, T_unmatch_n_p,
                            S_unmatch_a, T_unmatch_a, S_unmatch_a_p, T_unmatch_a_p,
                            S_unmatch_v, T_unmatch_v, S_unmatch_v_p, T_unmatch_v_p,
                            S_unmatch_pp, T_unmatch_pp, S_unmatch_pp_p, T_unmatch_pp_p,
                            S_unmatch_wp, T_unmatch_wp, S_unmatch_wp_p, T_unmatch_wp_p,
                            S_unmatch_num, T_unmatch_num, S_unmatch_num_p, T_unmatch_num_p,
                            S_unmatch_ner, T_unmatch_ner,
                            len_dif, len_dif_p,
                            is_duplicate]
            rows.append(features_row)
        except EOFError:
            break

    columns = ['id',
               'S_sem_sim', 'T_sem_sim', 'sem_sim',
               'S_unmatch_n','T_unmatch_n','S_unmatch_n_p','T_unmatch_n_p',
               'S_unmatch_a','T_unmatch_a','S_unmatch_a_p','T_unmatch_a_p',
               'S_unmatch_v','T_unmatch_v','S_unmatch_v_p','T_unmatch_v_p',
               'S_unmatch_pp','T_unmatch_pp','S_unmatch_pp_p','T_unmatch_pp_p',
               'S_unmatch_wp','T_unmatch_wp','S_unmatch_wp_p','T_unmatch_wp_p',
               'S_unmatch_num', 'T_unmatch_num', 'S_unmatch_num_p', 'T_unmatch_num_p',
               'S_unmatch_ner', 'T_unmatch_ner',
               'len_dif', 'len_dif_p',
               'is_duplicate']
    df = pd.DataFrame(np.array(rows), columns=columns)


df[df.columns[1:-1]].to_csv('train_feature_stanfordcorenlp.csv', index=False)

