# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:07:27 2020

@author: xiaodanli
"""

#https://machinelearningmastery.com/implementing-stacking-scratch-python/
#https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/

# stacked generalization with linear meta model on blobs dataset
import numpy as np
from keras.utils import np_utils
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from keras.utils import to_categorical
from numpy import dstack
from keras.models import Sequential
from keras.layers import Dense
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from deep_learning_models import stacked_LSTM, CNN, CNN_LSTM, LSTM_with_dropout, basic_LSTM, Bidirectional_LSTM, DNN, transformResult

### Stacked LSTM
sLSTM_params = {
    'batch_size' : 123,
    'epochs' : 146 #15
}

### CNN
CNN_params = {
    'batch_size' : 5,
    'epochs' : 109
}

### LSTM
LSTM_params = {
    'batch_size' : 350,
    'epochs' : 15
}

### DNN
DNN_params = {
    'batch_size' : 350,
    'epochs' : 15
}
 
# fit model on dataset
def fit_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=trainX.shape[1], activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model
 
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, trainx, trainy):
    trainedModels = list()
    stackX = None
    for mn, clf in members:
        if mn == "slstm":
            tr_x = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
            tr_y = np_utils.to_categorical(trainy)
            clf.fit(tr_x, tr_y, sLSTM_params['batch_size'], sLSTM_params['epochs'])
            yhat = clf.predict(tr_x)
        elif mn == "lstm":
            tr_x = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
            tr_y = np_utils.to_categorical(trainy)
            clf.fit(tr_x, tr_y, LSTM_params['batch_size'], LSTM_params['epochs'])
            yhat = clf.predict(tr_x)
        elif mn == "dnn":
            tr_y = np_utils.to_categorical(trainy)
            clf.fit(trainx, tr_y, DNN_params['batch_size'], DNN_params['epochs'])
            yhat = clf.predict(trainx)
        elif mn == "cnn":
            tr_x = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], 1))
            tr_y = np_utils.to_categorical(trainy)
            clf.fit(tr_x, tr_y, CNN_params['batch_size'], CNN_params['epochs'])
            yhat = clf.predict(tr_x)
        else:
            clf.fit(trainx, trainy)
            yhat = to_categorical(clf.predict(trainx))
        print(mn)
        print(yhat.shape)
        trainedModels.append((mn, clf))
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX, trainedModels
 
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, trainx, trainy):
    # create dataset using ensemble
    stackedX, trainedModels = stacked_dataset(members, trainx, trainy)
    # fit standalone model
    model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, 
                              colsample_bytree=1, gamma=0, learning_rate=0.16, max_delta_step=0, max_depth=10, 
                              min_child_weight=3, missing=None, n_estimators=200, n_jobs=1, nthread=None, 
                              objective='multi:softprob', random_state=0, reg_alpha=0, reg_lambda=1, 
                              scale_pos_weight=1, seed=None, silent=None, subsample=1, verbosity=1)
    model.fit(stackedX, trainy)
    return model, trainedModels
 
# make a prediction with the stacked model
def stacked_prediction(trainedModels, model, trainx):
    # create dataset using ensemble
    stackX = None
    for mn, clf in trainedModels:
        if mn == "slstm":
            tr_x = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
            yhat = clf.predict(tr_x)
        elif mn == "lstm":
            tr_x = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
            yhat = clf.predict(tr_x)
        elif mn == "dnn":
            yhat = clf.predict(trainx)
        elif mn == "cnn":
            tr_x = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], 1))
            yhat = clf.predict(tr_x)
        else:
            yhat = to_categorical(clf.predict(trainx))
        print(mn)
        print(yhat.shape)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    # make a prediction
    yhat = model.predict(stackX)
    return yhat

df_removed = pd.read_csv('../result/process_data.csv') 
X = df_removed.drop(['label'], axis=1)
y = df_removed[['label']]

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(trainX)
X_test = scaler.transform(testX)

"""
# fit and save models
n_members = 5
models = list()
for i in range(n_members):
    trainy_enc = to_categorical(trainy)
    # fit model
    model = fit_model(X_train, trainy_enc)
    models.append(('model_' + str(i + 1), model))
"""

labels = pd.Series(testy['label'])
labels = labels.convert_objects(convert_numeric=True)
y_test_onehot = np_utils.to_categorical(labels)

labels = pd.Series(trainy['label'])
labels = labels.convert_objects(convert_numeric=True)
y_train_onehot = np_utils.to_categorical(labels)        

### adjust the dataset dimension
# reshape X to be [samples, time steps, features]
X_test_LSTM = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

### input for CNN
X_train_CNN = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_CNN = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


clf_sLSTM= stacked_LSTM(X_test_LSTM, y_test_onehot)
clf_cnn= CNN(X_train_CNN, y_test_onehot)
clf_LSTM= basic_LSTM(X_test_LSTM, y_test_onehot)
clf_dnn = DNN(X_train, y_train_onehot)
    
clfs = [
        ("slstm", clf_sLSTM),
        ("lstm", clf_LSTM),
        ("dnn", clf_dnn),
        ("cnn", clf_cnn),
        ("svc", OneVsRestClassifier(LinearSVC(random_state=0))),
        ("rf", RandomForestClassifier(n_estimators=800, max_features="log2", max_depth=25, min_samples_leaf=2, min_samples_split=2, bootstrap=True, n_jobs=-1, random_state=1)),
        ("xgb", xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, 
                          colsample_bytree=1, gamma=0, learning_rate=0.16, max_delta_step=0, 
                          max_depth=10, min_child_weight=3, missing=None, n_estimators=200, 
                          n_jobs=1, nthread=None, objective='multi:softprob', random_state=0, 
                          reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=None, subsample=1, verbosity=1)),
        ("extraTree", ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy', max_depth=30, 
                                           max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                           min_impurity_split=None, min_samples_leaf=2, min_samples_split=5,
                                           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=None, 
                                           oob_score=False, random_state=None, verbose=0, warm_start=False))
        ]
"""       
# evaluate standalone models on test dataset
for mn, model in models:
	testy_enc = to_categorical(testy)
	_, acc = model.evaluate(X_test, testy_enc, verbose=0)
	print('Model Accuracy: %.3f' % acc)
"""
# fit stacked model using the ensemble
model, trainedModels = fit_stacked_model(clfs, X_train, trainy)
# evaluate model on test set
yhat = stacked_prediction(trainedModels, model, X_test)
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)