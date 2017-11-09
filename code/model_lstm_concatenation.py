
# coding: utf-8


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
from sklearn.cross_validation import KFold
from utils import _save, _load, SaveData
import sys


save_data = _load('./nn_glove_embedding_data.pkl')
data_1 = save_data.data_1
data_2 = save_data.data_2
labels = save_data.labels
test_data_1 = save_data.test_data_1
test_data_2 = save_data.test_data_2
test_ids = save_data.test_ids
embedding_matrix = save_data.embedding_matrix
nb_words = save_data.nb_words


def build_model(nb_words,
                embedding_dim,
                embedding_matrix,
                max_sequence_length,
                rate_drop_lstm,
                rate_drop_dense,
                lstm_num,
                dense_num,
                act
               ):
    
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
#     merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(dense_num, activation=act)(merged)
#     merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
#     merged = Dense(dense_num//2, activation=act)(merged)
# #     merged = Dropout(rate_drop_dense)(merged)
#     merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)
    
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    return model


num_lstm = 400
num_dense = 150
rate_drop_dense = 0.1
rate_drop_lstm = 0.15
re_weight = False
act = 'relu'


nfolds = 5
folds = KFold(data_1.shape[0], n_folds = nfolds, shuffle = True, random_state = 2017)
pred_results = []

for curr_fold, (idx_train, idx_val) in enumerate(folds):

    data_1_train = data_1[idx_train]
    data_2_train = data_2[idx_train]
    labels_train = labels[idx_train]

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = labels[idx_val]

    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels_val==0] = 1.309028344

    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

    model = build_model(nb_words,
                        embedding_matrix.shape[1],
                        embedding_matrix,
                        data_1.shape[1],
                        rate_drop_lstm,
                        rate_drop_dense,
                        num_dense,
                        num_lstm,
                        act
                       )
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = 'lstm_concatenation_%ddense_%.2fdropoutdense_%.2fdropoutlstm_%dfold_%dcurfold.h5'%(num_dense, rate_drop_dense, rate_drop_lstm, nfolds, curr_fold)
    model_checkpoint = ModelCheckpoint(bst_model_path,
                                       save_best_only=True,
                                       save_weights_only=True)
    
    print('   ')
    print(bst_model_path, "curr_fold:", curr_fold)
    
    hist = model.fit([data_1_train, data_2_train],
                     labels_train, 
                     validation_data=([data_1_val, data_2_val], labels_val, weight_val),
                     epochs=200,
                     batch_size=512,
                     shuffle=True, 
                     class_weight=class_weight,
                     callbacks=[early_stopping, model_checkpoint],
                     verbose = 2)

#     break
    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
    
    preds = model.predict([test_data_1, test_data_2], batch_size=2048, verbose=2)
    pred_results.append(preds)


res = (pred_results[0] + pred_results[1] + pred_results[2] +
       pred_results[3] + pred_results[4]) / 5
submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':res.ravel()})
a = 0.174264424749 / 0.369197853026
b = (1 - 0.174264424749) / (1 - 0.369197853026)
submission.to_csv('lstm_concatenation_prediction.csv', index=False)


submission.is_duplicate = submission.is_duplicate.apply(lambda x: a * x / (a * x + b * (1 - x)))
submission.to_csv('lstm_concatenation_prediction_trans.csv', index=False)

train_stacking_pred = np.zeros(data_1.shape[0]).astype(np.float32)
for curr_fold, (idx_train, idx_val) in enumerate(folds):

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = labels[idx_val]

    model = build_model(embedding_matrix, shape, settings)
    
    model.load_weights('lstm_concatenation_%ddense_%.2fdropoutdense_%.2fdropoutlstm_%dfold_%dcurfold.h5'%(num_dense, rate_drop_dense, rate_drop_lstm, nfolds, curr_fold))
    
    pred = model.predict([data_1_val, data_2_val], batch_size=2048, verbose=1).ravel()
    train_stacking_pred[idx_val] = pred

df = pd.DataFrame()
df['train_stacking_pred'] = train_stacking_pred
df['is_duplicate'] = labels
_save('train_lstm_concatenation_stacking_pred.pkl', train_stacking_pred)