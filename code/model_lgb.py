
# coding: utf-8

import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.cross_validation import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
from utils import _load, _save
from sklearn.metrics import log_loss
import lightgbm as lgb
seed=1024
np.random.seed(seed)

train_difflib = pd.read_csv('train_feature_difflib.csv')
train_embedding = pd.read_csv('train_feature_embedding.csv')
train_fuzz = pd.read_csv('train_feature_fuzz.csv')
train_len = pd.read_csv('train_feature_len.csv')
train_match = pd.read_csv('train_feature_match.csv')
train_match_2 = pd.read_csv('train_feature_match_2.csv')
train_oof = pd.read_csv('train_feature_oof.csv')
train_simhash = pd.read_csv('train_feature_simhash.csv')

train_graph_clique = pd.read_csv('train_feature_graph_clique.csv')
train_graph_pagerank = pd.read_csv('train_feature_graph_pagerank.csv')
train_graph_freq = pd.read_csv('train_feature_graph_question_freq.csv')
train_graph_intersect = pd.read_csv('train_feature_graph_intersect.csv')

train = pd.concat([
        train_difflib,
        train_embedding,
        train_fuzz,
        train_len,
        train_match,
        train_match_2,
        train_oof,
        train_simhash,
        
        train_graph_clique,
        train_graph_pagerank,
        train_graph_freq,
        train_graph_intersect,
    ], axis=1)

test_difflib = pd.read_csv('test_feature_difflib.csv')
test_embedding = pd.read_csv('test_feature_embedding.csv')
test_fuzz = pd.read_csv('test_feature_fuzz.csv')
test_len = pd.read_csv('test_feature_len.csv')
test_match = pd.read_csv('test_feature_match.csv')
test_match_2 = pd.read_csv('test_feature_match_2.csv')
test_oof = pd.read_csv('test_feature_oof.csv')
test_simhash = pd.read_csv('test_feature_simhash.csv')

test_graph_clique = pd.read_csv('test_feature_graph_clique.csv')
test_graph_pagerank = pd.read_csv('test_feature_graph_pagerank.csv')
test_graph_freq = pd.read_csv('test_feature_graph_question_freq.csv')
test_graph_intersect = pd.read_csv('test_feature_graph_intersect.csv')

test = pd.concat([
        test_difflib,
        test_embedding,
        test_fuzz,
        test_len,
        test_match,
        test_match_2,
        test_oof,
        test_simhash,
        
        test_graph_clique,
        test_graph_pagerank,
        test_graph_freq,
        test_graph_intersect,
    ], axis=1)


y = pd.read_csv('../../input/train.csv')['is_duplicate']


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 47,
    'learning_rate': 0.02,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'save_binary': True,
    'min_data_in_leaf': 100,
    'max_bin': 1023,
}

n_fold = 5
kf = KFold(n=train_the1owl.shape[0], n_folds=n_fold, shuffle=True, random_state=2017)

n = 0
for index_train, index_eval in kf:

    x_train, x_eval = train.iloc[index_train], train.iloc[index_eval]
    y_train, y_eval = y[index_train], y[index_eval]
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_eval, y_eval, reference=lgb_train)
    
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=[lgb_eval],
                    verbose_eval=100,
                    early_stopping_rounds=500)
    
    print('start predicting on test...')
    testpreds = gbm.predict(test.values, num_iteration=gbm.best_iteration)
    if n > 0:
        totalpreds = totalpreds + testpreds
    else:
        totalpreds = testpreds
    gbm.save_model('lgb_model_fold_{}.txt'.format(n), num_iteration=gbm.best_iteration)
    n += 1

totalpreds = totalpreds / n
test_id = pd.read_csv('../../input/test.csv')['test_id']
sub = pd.DataFrame()
sub['test_id'] = test_id
sub['is_duplicate'] = pd.Series(totalpreds)
sub.to_csv('lgb_prediction.csv', index=False)


a = 0.174264424749 / 0.369197853026
b = (1 - 0.174264424749) / (1 - 0.369197853026)
trans = sub.is_duplicate.apply(lambda x: a * x / (a * x + b * (1 - x)))
sub['is_duplicate'] = trans
sub.to_csv('lgb_prediction_trans.csv', index=False)


train_stacking_pred = np.empty(train.shape[0])
for n, (index_train, index_eval) in enumerate(kf):
    gbm = lgb.Booster(model_file="./lgb_model_fold_{}.txt".format(n))
    x_eval = train.iloc[index_eval].values
    train_stacking_pred[index_eval] = gbm.predict(x_eval, num_iteration=gbm.best_iteration)

_save('train_lgb_stacking_pred.pkl', train_stacking_pred)

