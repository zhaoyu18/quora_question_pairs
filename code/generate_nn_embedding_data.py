
# coding: utf-8


import numpy as np
import pandas as pd
import gensim
from gensim.models.wrappers import FastText
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import _save, _load, SaveData


BASE_DIR = '../../input/'
EMBEDDING_FILE = '../../corpora/glove_model.txt'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300


print('Indexing word vectors')
# glove
word2vec = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=False)
# google news
# word2vec = gensim.models.KeyedVectors.load_word2vec_format('../../corpora/GoogleNews-vectors-negative300.bin', binary=True)
# fast text
# word2vec = FastText.load_word2vec_format('../../corpora/wiki.en.vec')
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
train[['question1', 'question2']] = train[['question1', 'question2']].astype(str)
test[['question1', 'question2']] = test[['question1', 'question2']].astype(str)


print('Processing text dataset')

texts_1 = [] 
texts_2 = []
labels = []
def get_text(row):
    global texts_1, texts_2, labels
    texts_1.append(row.question1)
    texts_2.append(row.question2)
    labels.append(int(row.is_duplicate))
train.apply(get_text, axis=1)
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
def get_test_text(row):
    global test_texts_1, test_texts_2, test_ids
    test_texts_1.append(row.question1)
    test_texts_2.append(row.question2)
    test_ids.append(row.test_id)
test.apply(get_test_text, axis=1)
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)


print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


save_data = SaveData()
save_data.data_1 = data_1
save_data.data_2 = data_2
save_data.labels = labels
save_data.test_data_1 = test_data_1
save_data.test_data_2 = test_data_2
save_data.test_ids = test_ids
save_data.embedding_matrix = embedding_matrix
save_data.nb_words = nb_words
_save('nn_glove_embedding_data.pkl', save_data)




