from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model

import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from utils import _save, _load, SaveData

np.random.seed(1)
re_weight = False # whether to re-weight classes to fit the 17.5% share in test set

save_data = _load('./nn_glove_embedding_data.pkl')
data_1 = save_data.data_1
data_2 = save_data.data_2
labels = save_data.labels
test_data_1 = save_data.test_data_1
test_data_2 = save_data.test_data_2
test_ids = save_data.test_ids
embedding_matrix = save_data.embedding_matrix
nb_words = save_data.nb_words

def build_model(emb_matrix, max_sequence_length):
    
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=max_sequence_length,
        trainable=False
    )
    
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(max_sequence_length,))
    seq2 = Input(shape=(max_sequence_length,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    # Add the magic features
    magic_input = Input(shape=(5,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
    # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
    distance_input = Input(shape=(20,))
    distance_dense = BatchNormalization()(distance_input)
    distance_dense = Dense(128, activation='relu')(distance_dense)

    # Merge the Magic and distance features with the difference layer
    # merge = concatenate([diff, mul, magic_dense, distance_dense])
    merge = concatenate([diff, mul])

    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model

nfolds = 5
folds = KFold(data_1.shape[0], n_folds = nfolds, shuffle = True, random_state = 2017)
pred_results = []
re_weight = True

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
    
    model = build_model(embedding_matrix,
                        data_1.shape[1],
                       )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    bst_model_path = '1d_cnn_orig_fold{}.h5'.format(curr_fold)
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
submission.to_csv('1d_cnn_glove_prediction.csv', index=False)

submission.is_duplicate = submission.is_duplicate.apply(lambda x: a * x / (a * x + b * (1 - x)))
submission.to_csv('1d_cnn_glove_prediction_trans.csv', index=False)

train_stacking_pred = np.zeros(data_1.shape[0]).astype(np.float32)
for curr_fold, (idx_train, idx_val) in enumerate(folds):

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = labels[idx_val]

    model = build_model(embedding_matrix, shape, settings)
    
    model.load_weights('1d_cnn_orig_fold{}.h5'.format(curr_fold))
    
    pred = model.predict([data_1_val, data_2_val], batch_size=2048, verbose=1).ravel()
    train_stacking_pred[idx_val] = pred

df = pd.DataFrame()
df['train_stacking_pred'] = train_stacking_pred
df['is_duplicate'] = labels
_save('train_1d_cnn_stacking_pred.pkl', train_stacking_pred)