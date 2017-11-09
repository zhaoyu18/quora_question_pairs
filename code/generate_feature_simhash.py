
# coding: utf-8


import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from simhash import Simhash


df_train = pd.read_csv('../../input/train.csv')
df_test = pd.read_csv('../../input/test.csv')
len_train = df_train.shape[0]
df_feat = pd.DataFrame()
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)


def tokenize(sequence):
    words = word_tokenize(sequence)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words

def clean_sequence(sequence):
    tokens = tokenize(sequence)
    return ' '.join(tokens)

def get_word_ngrams(sequence, n=3):
    tokens = tokenize(sequence)
    return [' '.join(ngram) for ngram in ngrams(tokens, n)]

def get_character_ngrams(sequence, n=3):
    sequence = clean_sequence(sequence)
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]


def caluclate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))

def get_word_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return caluclate_simhash_distance(q1, q2)

def get_word_2gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_char_2gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_word_3gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)

def get_char_3gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)


df_data['questions'] = df_data['question1'] + '_split_tag_' + df_data['question2']


df_feat['simhash_tokenize_distance'] = df_data['questions'].apply(get_word_distance)
df_feat['simhash_word_2gram_distance'] = df_data['questions'].apply(get_word_2gram_distance)
df_feat['simhash_char_2gram_distance'] = df_data['questions'].apply(get_char_2gram_distance)
df_feat['simhash_word_3gram_distance'] = df_data['questions'].apply(get_word_3gram_distance)
df_feat['simhash_char_3gram_distance'] =df_data['questions'].apply(get_char_3gram_distance)


df_feat[:len_train].to_csv('train_feature_simhash.csv', index=False)
df_feat[len_train:].to_csv('test_feature_simhash.csv', index=False)




