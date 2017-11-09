import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
np.random.seed(1)

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys

# normalize feature data with z score: all_data = (all_data - all_data.mean())/all_data.std()

def build_model(
    feature_num,
    nb_words,
    embedding_dim,
    embedding_matrix,
    max_sequence_length,
    rate_drop_lstm,
    rate_drop_dense,
    lstm_num,
    dense_num,
    act
):
    
    input_3 = Input(shape=(feature_num,))
    z1 = Dense(feature_num, kernel_initializer="he_normal")(input_3)
    embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)
    lstm_layer = LSTM(lstm_num, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(dense_num, activation=act, name='sentence_vector')(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    merged = concatenate([merged, z1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    merged = Dense(dense_num, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    merged = Dense(dense_num//2, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)
    
    model = Model(inputs=[sequence_1_input, sequence_2_input, input_3], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    return model