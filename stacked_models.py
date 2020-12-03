# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:21:08 2019

@author: xiaodanli
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from keras.utils import np_utils
from deep_learning_models import stacked_LSTM, CNN, CNN_LSTM, LSTM_with_dropout, basic_LSTM, Bidirectional_LSTM, DNN, transformResult, stacked_LSTM_lookahead, CNN_lookahead, CNN_LSTM_lookahead, LSTM_with_dropout_lookahead, basic_LSTM_lookahead, Bidirectional_LSTM_lookahead, DNN_lookahead
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
#import catboost as cb

def stacking(train_x, train_y, test_x, test_y, epochs, batch_size, lookahead=False):

    """ stacking

    input: train_x, train_y, test

    output: test的预测值

    clfs: 5个一级分类器

    dataset_blend_train: 一级分类器的prediction, 二级分类器的train_x

    dataset_blend_test: 二级分类器的test

    """
    labels = pd.Series(test_y['label'])
    labels = labels.convert_objects(convert_numeric=True)
    y_test_onehot = np_utils.to_categorical(labels)
    
    labels = pd.Series(train_y['label'])
    labels = labels.convert_objects(convert_numeric=True)
    y_train_onehot = np_utils.to_categorical(labels)        
    
    ### adjust the dataset dimension
    # reshape X to be [samples, time steps, features]
    X_test_LSTM = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    
    #epochs = 13
    #batch_size = 250
    
    ### Stacked LSTM
    sLSTM_params = {
        'batch_size' : batch_size,
        'epochs' : epochs
    }
    
    ### input for CNN
    X_train_CNN = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    X_test_CNN = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    
    ### CNN
    CNN_params = {
        'batch_size' : 5,
        'epochs' : 109 #15
    }
    
    ### CNN LSTM
    CNN_LSTM_params = {
        'batch_size' : batch_size,
        'epochs' : epochs # 15
    }
    dropout = 0.2
    
    ### LSTM_with_dropout
    LSTM_dropout_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
    
    ### LSTM
    LSTM_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
    
    ### bidirectional LSTM
    biLSTM_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
    
    ### DNN
    DNN_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
        
    d_train_lgb = lgb.Dataset(train_x, label=train_y)
    lgb_params = {"max_depth": 25, "learning_rate" : 0.1, "num_leaves": 300,  "n_estimators": 200}
    
    clf_sLSTM= None
    clf_cnn= None
    clf_cnn_lstm= None
    clf_lstm_dropout = None
    clf_LSTM= None
    clf_biLSTM= None
    clf_dnn = None
    LGB = None
    
    if lookahead:
        print("lookahead is applied")
        clf_sLSTM= stacked_LSTM_lookahead(X_test_LSTM, y_test_onehot)
        clf_cnn= CNN_lookahead(X_train_CNN, y_test_onehot)
        clf_cnn_lstm= CNN_LSTM_lookahead(X_train_CNN, y_test_onehot)
        clf_lstm_dropout = LSTM_with_dropout_lookahead(X_test_LSTM, y_test_onehot, dropout)
        clf_LSTM= basic_LSTM_lookahead(X_test_LSTM, y_test_onehot)
        clf_biLSTM= Bidirectional_LSTM_lookahead(X_test_LSTM, y_test_onehot)
        clf_dnn = DNN_lookahead(train_x, y_train_onehot)
    else:
        print("traditional optimizer is applied")
        clf_sLSTM= stacked_LSTM(X_test_LSTM, y_test_onehot)
        clf_cnn= CNN(X_train_CNN, y_test_onehot)
        clf_cnn_lstm= CNN_LSTM(X_train_CNN, y_test_onehot)
        clf_lstm_dropout = LSTM_with_dropout(X_test_LSTM, y_test_onehot, dropout)
        clf_LSTM= basic_LSTM(X_test_LSTM, y_test_onehot)
        clf_biLSTM= Bidirectional_LSTM(X_test_LSTM, y_test_onehot)
        clf_dnn = DNN(train_x, y_train_onehot)
        LGB = lgb.train(lgb_params, d_train_lgb)        
        
    # 5个一级分类器
    clfs = [#SVC(C = 3, kernel="rbf"),
            clf_sLSTM,
            clf_cnn_lstm,
            clf_dnn,
            OneVsRestClassifier(LinearSVC(random_state=0)),
            RandomForestClassifier(n_estimators=800, max_features="log2", max_depth=25, min_samples_leaf=2, min_samples_split=2, bootstrap=True, n_jobs=-1, random_state=1),
            xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.16, max_delta_step=0, max_depth=10, min_child_weight=3, missing=None, n_estimators=200, n_jobs=1, nthread=None, objective='multi:softprob', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=None, subsample=1, verbosity=1),
            ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy', max_depth=30, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=2, min_samples_split=5,min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False),
            #CatBoostClassifier(eval_metric="MultiClass", depth=10, iterations= 300, l2_leaf_reg= 9, learning_rate= 0.15)
            ]

    # 二级分类器的train_x, test
    dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)), dtype=np.int)
    dataset_blend_test = np.zeros((test_x.shape[0], len(clfs)), dtype=np.int)
    # 5个分类器进行8_folds预测

    n_folds = 9 # 8

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    for i, clf in enumerate(clfs):

        dataset_blend_test_j = np.zeros((test_x.shape[0], n_folds))  # 每个分类器的单次fold预测结果

        for j,(train_index,test_index) in enumerate(skf.split(train_x, train_y)):
            tr_x = train_x[train_index]
            tr_y = train_y.values[train_index]
            if i == 0:
                # reshape X to be [samples, time steps, features]
                tr_x = np.reshape(tr_x, (tr_x.shape[0], 1, tr_x.shape[1]))
                tr_y = np_utils.to_categorical(tr_y)
                clf.fit(tr_x, tr_y, sLSTM_params['batch_size'], sLSTM_params['epochs'])
                tmp = train_x[test_index]
                tmp_1 = np.reshape(tmp, (tmp.shape[0], 1, tmp.shape[1]))
                sLSTM_prediction = clf.predict(tmp_1)
                dataset_blend_train[test_index, i] = transformResult(sLSTM_prediction)
                X_test_LSTM = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
                dataset_blend_test_j[:, j] = transformResult(clf.predict(X_test_LSTM))
            elif i == 1:
                tr_x = np.reshape(tr_x, (tr_x.shape[0], tr_x.shape[1], 1))
                tr_y = np_utils.to_categorical(tr_y)
                clf.fit(tr_x, tr_y, CNN_LSTM_params['batch_size'], CNN_LSTM_params['epochs'])
                tmp = train_x[test_index]
                tmp_2 = np.reshape(tmp, (tmp.shape[0], tmp.shape[1], 1))
                CNN_LSTM_prediction = clf.predict(tmp_2)
                dataset_blend_train[test_index, i] = transformResult(CNN_LSTM_prediction)  
                X_test_CNN_LSTM = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
                dataset_blend_test_j[:, j] = transformResult(clf.predict(X_test_CNN_LSTM))
            elif i == 2:
                tr_y = np_utils.to_categorical(tr_y)
                clf.fit(tr_x, tr_y, DNN_params['batch_size'], DNN_params['epochs'])
                dataset_blend_train[test_index, i] = transformResult(clf.predict(train_x[test_index]))
                dataset_blend_test_j[:, j] = transformResult(clf.predict(test_x))
            elif i == 7:
                clf.fit(tr_x, tr_y)
                pre_train = clf.predict(train_x[test_index])
                dataset_blend_train[test_index, i] = pre_train.flatten()
                prediction = clf.predict(test_x)
                dataset_blend_test_j[:, j] = prediction.flatten()
            else:
                clf.fit(tr_x, tr_y)
                dataset_blend_train[test_index, i] = clf.predict(train_x[test_index])
                dataset_blend_test_j[:, j] = clf.predict(test_x)

        dataset_blend_test[:, i] = dataset_blend_test_j.sum(axis=1) // (n_folds//2 + 1)

    # 二级分类器进行预测
    #clf = LogisticRegression(penalty="l2", tol=1e-6, C=1.0, random_state=1, n_jobs=-1, multi_class='multinomial', solver='newton-cg')
    model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, 
                              colsample_bynode=1, colsample_bytree=1, gamma=0, 
                              learning_rate=0.16, max_delta_step=0, max_depth=10, 
                              min_child_weight=3, missing=None, n_estimators=200, 
                              n_jobs=1, nthread=None, objective='multi:softprob', 
                              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
                              seed=None, silent=None, subsample=1, verbosity=1)
    
    X_train, y_train = dataset_blend_train, train_y
    X_test, y_test = dataset_blend_test, test_y
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(X_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True)
    #model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)
    
    lookahead_label = ' with RMSprop'
    if lookahead: 
        lookahead_label = ' with Lookahead'
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    ax.legend()
    pyplot.ylabel('mLog Loss')
    pyplot.title('mLog Loss' + lookahead_label)
    plt.savefig('../result/mLog Loss for stacked model.png', dpi=300)
    pyplot.show()
    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['merror'], label='Train')
    ax.plot(x_axis, results['validation_1']['merror'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification mError')
    pyplot.title('Classification mError' + lookahead_label)
    plt.savefig('../result/Classification mError for stacked model.png', dpi=300)
    pyplot.show()
    
    return predictions, dataset_blend_train, dataset_blend_test