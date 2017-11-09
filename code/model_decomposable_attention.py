
# coding: utf-8


from keras.layers import InputSpec, Layer, Input, Dense, merge
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers import Bidirectional, GRU, LSTM, SpatialDropout1D
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import ELU
import keras.backend as K
from keras.models import Sequential, Model, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from utils import _save, _load, SaveData

np.random.seed(1)
re_weight = False


save_data = _load('./nn_glove_embedding_data.pkl')
data_1 = save_data.data_1
data_2 = save_data.data_2
labels = save_data.labels
test_data_1 = save_data.test_data_1
test_data_2 = save_data.test_data_2
test_ids = save_data.test_ids
embedding_matrix = save_data.embedding_matrix
nb_words = save_data.nb_words


class _StaticEmbedding(object):
    def __init__(self, vectors, max_length, nr_out, nr_tune=1000, dropout=0.0):
        self.nr_out = nr_out
        self.max_length = max_length
        self.dropout = dropout
        self.embed = Embedding(
                        vectors.shape[0],
                        vectors.shape[1],
                        input_length=max_length,
                        weights=[vectors],
                        name='embed',
                        trainable=False)
        self.tune = Embedding(
                        nr_tune,
                        nr_out,
                        input_length=max_length,
                        weights=None,
                        name='tune',
                        trainable=True)
        self.mod_ids = Lambda(lambda sent: sent % (nr_tune-1)+1,
                              output_shape=(self.max_length,))

        self.project = TimeDistributed(
                            Dense(
                                nr_out,
                                activation=None,
                                bias=False,
                                name='project'))

    def __call__(self, sentence):
        def get_output_shape(shapes):
            print(shapes)
            return shapes[0]
        mod_sent = self.mod_ids(sentence)
        tuning = self.tune(mod_sent)
        tuning = SpatialDropout1D(self.dropout)(tuning)
        #tuning = merge([tuning, mod_sent],
        #    mode=lambda AB: AB[0] * (K.clip(K.cast(AB[1], 'float32'), 0, 1)),
        #    output_shape=(self.max_length, self.nr_out))
        pretrained = self.project(self.embed(sentence))
        vectors = merge([pretrained, tuning], mode='sum')
        return vectors

class _BiRNNEncoding(object):
    def __init__(self, max_length, nr_out, dropout=0.0):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(nr_out, return_sequences=True,
                                         dropout_W=dropout, dropout_U=dropout),
                                         input_shape=(max_length, nr_out)))
        self.model.add(TimeDistributed(Dense(nr_out, activation='relu', init='he_normal')))
        self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, sentence):
        return self.model(sentence)

class _Attention(object):
    def __init__(self, max_length, nr_hidden, dropout=0.0, L2=0.0, activation='relu'):
        self.max_length = max_length
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(nr_hidden,)))
        self.model.add(
            Dense(nr_hidden, name='attend1',
                init='he_normal', W_regularizer=l2(L2),
                input_shape=(nr_hidden,), activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nr_hidden, name='attend2',
            init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model = TimeDistributed(self.model)

    def __call__(self, sent1, sent2):
        def _outer(AB):
            att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
            return K.permute_dimensions(att_ji,(0, 2, 1))
        return merge(
                [self.model(sent1), self.model(sent2)],
                mode=_outer,
                output_shape=(self.max_length, self.max_length))

class _SoftAlignment(object):
    def __init__(self, max_length, nr_hidden):
        self.max_length = max_length
        self.nr_hidden = nr_hidden

    def __call__(self, sentence, attention, transpose=False):
        def _normalize_attention(attmat):
            att = attmat[0]
            mat = attmat[1]
            if transpose:
                att = K.permute_dimensions(att,(0, 2, 1))
            # 3d softmax
            e = K.exp(att - K.max(att, axis=-1, keepdims=True))
            s = K.sum(e, axis=-1, keepdims=True)
            sm_att = e / s
            return K.batch_dot(sm_att, mat)
        return merge([attention, sentence], mode=_normalize_attention,
                      output_shape=(self.max_length, self.nr_hidden)) # Shape: (i, n)

class _Comparison(object):
    def __init__(self, words, nr_hidden, L2=0.0, dropout=0.0):
        self.words = words
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(nr_hidden*2,)))
        self.model.add(Dense(nr_hidden, name='compare1',
            init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nr_hidden, name='compare2',
                        W_regularizer=l2(L2), init='he_normal'))
        self.model.add(Activation('relu'))
        self.model = TimeDistributed(self.model)

    def __call__(self, sent, align, **kwargs):
        result = self.model(merge([sent, align], mode='concat')) # Shape: (i, n)
#         avged = GlobalAveragePooling1D()(result, mask=self.words)
        avged = GlobalAveragePooling1D()(result)
#         maxed = GlobalMaxPooling1D()(result, mask=self.words)
        maxed = GlobalMaxPooling1D()(result)
        merged = merge([avged, maxed])
        result = BatchNormalization()(merged)
        return result

class _Entailment(object):
    def __init__(self, nr_hidden, nr_out, dropout=0.0, L2=0.0):
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(nr_hidden*2,)))
        self.model.add(Dense(nr_hidden, name='entail1',
            init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nr_hidden, name='entail2',
            init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1, name='entail_out', activation='sigmoid',
                        W_regularizer=l2(L2), init='zero'))

    def __call__(self, feats1, feats2):
        features = merge([feats1, feats2], mode='concat')
        return self.model(features)

class _GlobalSumPooling1D(Layer):
    '''Global sum pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''
    def __init__(self, **kwargs):
        super(_GlobalSumPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if mask is not None:
            return K.sum(x * K.clip(mask, 0, 1), axis=1)
        else:
            return K.sum(x, axis=1)

def build_model(vectors, shape, settings):
    '''Compile the model.'''
    max_length, nr_hidden, nr_class = shape
    # Declare inputs.
    ids1 = Input(shape=(max_length,), dtype='int32', name='words1')
    ids2 = Input(shape=(max_length,), dtype='int32', name='words2')

    # Construct operations, which we'll chain together.
    embed = _StaticEmbedding(vectors, max_length, nr_hidden, dropout=0.2, nr_tune=5000)
    if settings['gru_encode']:
        encode = _BiRNNEncoding(max_length, nr_hidden, dropout=settings['dropout'])
    attend = _Attention(max_length, nr_hidden, dropout=settings['dropout'])
    align = _SoftAlignment(max_length, nr_hidden)
    compare = _Comparison(max_length, nr_hidden, dropout=settings['dropout'])
    entail = _Entailment(nr_hidden, nr_class, dropout=settings['dropout'])

    # Declare the model as a computational graph.
    sent1 = embed(ids1) # Shape: (i, n)
    sent2 = embed(ids2) # Shape: (j, n)

    if settings['gru_encode']:
        sent1 = encode(sent1)
        sent2 = encode(sent2)

    attention = attend(sent1, sent2)  # Shape: (i, j)

    align1 = align(sent2, attention)
    align2 = align(sent1, attention, transpose=True)

    feats1 = compare(sent1, align1)
    feats2 = compare(sent2, align2)

    scores = entail(feats1, feats2)

    # Now that we have the input/output, we can construct the Model object...
    model = Model(input=[ids1, ids2], output=[scores])

    # ...Compile it...
    model.compile(
        optimizer=Adam(lr=settings['lr']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    # ...And return it for training.
    return model


num_dense = 150
rate_drop = 0.1
nfolds = 5

shape = (40, num_dense, 1)
settings = {'lr': 0.001, 'dropout': rate_drop, 'gru_encode':False}


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

    model = build_model(embedding_matrix, shape, settings)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    bst_model_path = 'decomposable_attention_%dfold_%dcurfold.h5'%(nfolds, curr_fold)
    model_checkpoint = ModelCheckpoint(bst_model_path,
                                       save_best_only=True,
                                       save_weights_only=True)

    print(bst_model_path, "curr_fold:", curr_fold)

    hist = model.fit([data_1_train, data_2_train],
                     labels_train, 
                     validation_data=([data_1_val, data_2_val], labels_val, weight_val),
                     epochs=200,
                     batch_size=512,
                     shuffle=True, 
                     class_weight=class_weight,
                     callbacks=[early_stopping, model_checkpoint],
                     verbose=2)

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
submission.to_csv('decomposable_attention_prediction.csv', index=False)


submission.is_duplicate = submission.is_duplicate.apply(lambda x: a * x / (a * x + b * (1 - x)))
submission.to_csv('decomposable_attention_prediction_trans.csv', index=False)


train_stacking_pred = np.zeros(data_1.shape[0]).astype(np.float32)
for curr_fold, (idx_train, idx_val) in enumerate(folds):

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = labels[idx_val]

    model = build_model(embedding_matrix, shape, settings)
    
    model.load_weights('decomposable_attention_%dfold_%dcurfold.h5'%(nfolds, curr_fold))
    
    pred = model.predict([data_1_val, data_2_val], batch_size=2048, verbose=1).ravel()
    train_stacking_pred[idx_val] = pred

df = pd.DataFrame()
df['train_stacking_pred'] = train_stacking_pred
df['is_duplicate'] = labels
_save('train_decomposable_attention_stacking_pred.pkl', train_stacking_pred)

