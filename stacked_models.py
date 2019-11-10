# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:21:08 2019

@author: xiaodanli
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from deep_learning_models import stacked_LSTM, CNN, CNN_LSTM, LSTM_with_dropout, basic_LSTM, Bidirectional_LSTM, DNN, transformResult, getAccuracyMulti
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def stacking(train_x, train_y, test_x, test_y, epochs, batch_size):

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
        'epochs' : epochs #15
    }
    
    clf_sLSTM= stacked_LSTM(X_test_LSTM, y_test_onehot)
    
    ### input for CNN
    X_train_CNN = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    X_test_CNN = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    
    ### CNN
    CNN_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
    
    clf_cnn= CNN(X_train_CNN, y_test_onehot)

    ### CNN LSTM
    CNN_LSTM_params = {
        'batch_size' : batch_size,
        'epochs' : epochs # 15
    }
    dropout = 0.2
    clf_cnn_lstm= CNN_LSTM(X_train_CNN, y_test_onehot, dropout)
    
    ### LSTM_with_dropout
    LSTM_dropout_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
    clf_lstm_dropout = LSTM_with_dropout(X_test_LSTM, y_test_onehot, dropout)
    
    ### LSTM
    LSTM_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
    
    clf_LSTM= basic_LSTM(X_test_LSTM, y_test_onehot)
    
    ### bidirectional LSTM
    biLSTM_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
    clf_biLSTM= Bidirectional_LSTM(X_test_LSTM, y_test_onehot)
    
    ### DNN
    DNN_params = {
        'batch_size' : batch_size,
        'epochs' : epochs #15
    }
    clf_dnn = DNN(train_x, y_train_onehot, dropout)
    # 5个一级分类器
    clfs = [#SVC(C = 3, kernel="rbf"),
            OneVsRestClassifier(LinearSVC(random_state=0)),
            RandomForestClassifier(n_estimators=100, max_features="log2", max_depth=10, min_samples_leaf=1, bootstrap=True, n_jobs=-1, random_state=1),
            KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
            xgb.XGBClassifier(n_estimators=100, objective="multi:softmax", gamma=1, max_depth=10, subsample=0.8, nthread=-1, seed=1, num_class=3),
            ExtraTreesClassifier(n_estimators=100, criterion="gini", max_features="log2", max_depth=10, min_samples_split=2, min_samples_leaf=1,bootstrap=True, n_jobs=-1, random_state=1),
            clf_sLSTM,
            clf_cnn_lstm,
            clf_dnn
            ]

    # 二级分类器的train_x, test
    dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)), dtype=np.int)
    dataset_blend_test = np.zeros((test_x.shape[0], len(clfs)), dtype=np.int)
    # 5个分类器进行8_folds预测

    n_folds = 8 # 8

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    for i, clf in enumerate(clfs):

        dataset_blend_test_j = np.zeros((test_x.shape[0], n_folds))  # 每个分类器的单次fold预测结果

        for j,(train_index,test_index) in enumerate(skf.split(train_x, train_y)):
            tr_x = train_x[train_index]
            tr_y = train_y.values[train_index]
            if i == 5:
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
            elif i == 6:
                tr_x = np.reshape(tr_x, (tr_x.shape[0], tr_x.shape[1], 1))
                tr_y = np_utils.to_categorical(tr_y)
                clf.fit(tr_x, tr_y, CNN_LSTM_params['batch_size'], CNN_LSTM_params['epochs'])
                tmp = train_x[test_index]
                tmp_2 = np.reshape(tmp, (tmp.shape[0], tmp.shape[1], 1))
                CNN_LSTM_prediction = clf.predict(tmp_2)
                dataset_blend_train[test_index, i] = transformResult(CNN_LSTM_prediction)  
                X_test_CNN_LSTM = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
                dataset_blend_test_j[:, j] = transformResult(clf.predict(X_test_CNN_LSTM))
            elif i == 7:
                tr_y = np_utils.to_categorical(tr_y)
                clf.fit(tr_x, tr_y, DNN_params['batch_size'], DNN_params['epochs'])
                dataset_blend_train[test_index, i] = transformResult(clf.predict(train_x[test_index]))
                dataset_blend_test_j[:, j] = transformResult(clf.predict(test_x))
            else:
                clf.fit(tr_x, tr_y)
                dataset_blend_train[test_index, i] = clf.predict(train_x[test_index])
                dataset_blend_test_j[:, j] = clf.predict(test_x)

        dataset_blend_test[:, i] = dataset_blend_test_j.sum(axis=1) // (n_folds//2 + 1)

    # 二级分类器进行预测

    clf = LogisticRegression(penalty="l2", tol=1e-6, C=1.0, random_state=1, n_jobs=-1, multi_class='multinomial', solver='newton-cg')

    clf.fit(dataset_blend_train, train_y)
    
    prediction = clf.predict(dataset_blend_test)
    
    return prediction, dataset_blend_train, dataset_blend_test

if __name__ == "__main__":
    df = pd.read_csv('all_features_with_label.csv', sep=',',header=None) 
    df_feature = df.iloc[:, 1:]
    X = df_feature.values
    label = df.iloc[:,0]
    y = label.values
    cols = list(df)
    columns_names = ["label"]
    for i in range(0, 1817):
        columns_names.append("Column " + str(i))
    df.columns = columns_names
    """
        DATA EXPLORATION ANALYSIS
    """
    #correlation matrix
    #k = 15 #number of variables for heatmap
    #corrmat = df.corr()
    #cols = corrmat.nlargest(k, 'label')['label'].index
    #cm = np.corrcoef(df[cols].values.T)
    #sns.set(font_scale=1.25)
    #hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    #plt.show()
    
    #scatterplot
    #sns.set()
    #sns.pairplot(df[cols], size = 2.5)
    #plt.show()
    
    #missing data
    total = df_feature.isnull().sum().sort_values(ascending=False)
    percent = (df_feature.isnull().sum()/df_feature.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    #print(missing_data.head(20))
    
    threshold = 0.80
    corr_matrix = df[df['label'].notnull()].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
    #df_removed = df.drop(columns = to_drop)
    df_removed = df.drop(to_drop, axis=1)
    
    X = df_removed.drop(['label'], axis=1)
    y = df_removed[['label']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    best_acc = 0
    best_epo = 0
    best_size = 0
    
    for epochs in range(10, 16, 1):
        for batch_size in range(200, 500, 50):
            prediction, dataset_blend_train, dataset_blend_test = stacking(X_train, y_train, X_test, y_test, epochs, batch_size)
            accuracy, precision, recall, F1, fbeta = getAccuracyMulti(prediction, y_test)
            print(epochs)
            print(batch_size)
            print(accuracy)    
            print(precision)    
            print(recall)    
            print(F1)    
            print(fbeta)
            if accuracy > best_acc:
                best_acc = accuracy
                best_epo = epochs
                best_size = batch_size
      
    print("final output")    
    print(best_acc)
    print(best_epo)
    print(best_size)