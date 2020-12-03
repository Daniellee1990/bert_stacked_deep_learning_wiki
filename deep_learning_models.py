#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:23:54 2018

@author: lixiaodan
https://github.com/bojone/keras_lookahead
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import itertools
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import log_loss
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam
from keras.utils import plot_model
from lookahead import Lookahead

def stacked_LSTM(X_test, y_test):
    ## CNN LSTM
    model = Sequential()
    model.add(LSTM(42, return_sequences=True, input_shape=(1, X_test.shape[2]))) # 32 # 50
    model.add(LSTM(348, return_sequences=True)) 

    model.add(Flatten())
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(3, activation='sigmoid'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adagrad(lr=0.0001141303940684644)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    plot_model(model, to_file='../result/stacked_lstm.png', show_shapes=True, show_layer_names=True) # plot my model
    return model

def stacked_LSTM_lookahead(X_test, y_test):
    ## CNN LSTM
    model = Sequential()
    model.add(LSTM(91, return_sequences=True, input_shape=(1, X_test.shape[2]))) # 32 # 50
    model.add(LSTM(97, return_sequences=True)) 

    model.add(Flatten())
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(3, activation='softmax'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adagrad(lr=4.313502169608129e-05)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
    lookahead.inject(model) # 插入到模型中
    
    plot_model(model, to_file='../result/stacked_lstm.png', show_shapes=True, show_layer_names=True) # plot my model
    return model
    
def CNN(X_train, y_test):
    ## CNN
    model = Sequential()
    model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=19, kernel_size=[9], 
                     padding='same', activation='relu', name='layer_conv1'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(kernel_size=[26], filters=12,
                     padding='same', activation='relu', name='layer_conv2'))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(3, activation='softmax'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adagrad(lr=0.0004104646460561052)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def CNN_lookahead(X_train, y_test):
    ## CNN
    model = Sequential()
    #model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=5, kernel_size=1, activation='relu'))
    model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=8, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
    lookahead.inject(model)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    #print(model.summary())
    #history = model.fit(X_train, y_train, batch_size, epochs)
    return model

def CNN_LSTM(X_train, y_test):
    model = Sequential()
    model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=14, kernel_size=[18], 
                     padding='same', activation='relu', name='layer_conv1'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(kernel_size=[30], filters=25,
                     padding='same', activation='relu', name='layer_conv2'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    model.add(LSTM(359, return_sequences=True))
    model.add(LSTM(359, return_sequences=True)) 
    model.add(LSTM(359, return_sequences=True)) 

    model.add(Flatten())
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(3, activation='softmax'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adagrad(lr=0.00010997321170422143)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    plot_model(model, to_file='../result/cnn_lstm.png', show_shapes=True, show_layer_names=True) # plot my model
    return model

def CNN_LSTM_lookahead(X_train, y_test):
    model = Sequential()
    model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=14, kernel_size=[18], 
                     padding='same', activation='relu', name='layer_conv1'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(kernel_size=[30], filters=25,
                     padding='same', activation='relu', name='layer_conv2'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    model.add(LSTM(359, return_sequences=True))
    model.add(LSTM(359, return_sequences=True)) 
    model.add(LSTM(359, return_sequences=True)) 

    model.add(Flatten())
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(3, activation='softmax'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adagrad(lr=0.00010997321170422143)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
    lookahead.inject(model)
    
    plot_model(model, to_file='../result/cnn_lstm.png', show_shapes=True, show_layer_names=True) # plot my model
    return model

def LSTM_with_dropout(X_train, y_test, dropout_rate):
    ## stacked LSTM
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True)) #33
    #model.add(LSTM(32, return_sequences=True, input_shape=(1, x_test.shape[2])))
    model.add(Dropout(dropout_rate))
    #model.add(LSTM(100))
    #model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    #print(model.summary())
    #history = model.fit(X_train, y_train, batch_size, epochs)
    return model

def LSTM_with_dropout_lookahead(X_train, y_test, dropout_rate):
    ## stacked LSTM
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True)) #33
    #model.add(LSTM(32, return_sequences=True, input_shape=(1, x_test.shape[2])))
    model.add(Dropout(dropout_rate))
    #model.add(LSTM(100))
    #model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
    lookahead.inject(model)
    #print(model.summary())
    #history = model.fit(X_train, y_train, batch_size, epochs)
    return model

def basic_LSTM(X_train, y_test):
    model = Sequential()
    model.add(LSTM(332, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True)) # 33
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    optimizer = Adagrad(lr=0.00014848706050504123)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #print(model.summary())
    #history = model.fit(X_train, y_train, batch_size, epochs)
    return model

def basic_LSTM_lookahead(X_train, y_test):
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True)) # 33
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
    lookahead.inject(model)
    #print(model.summary())
    #history = model.fit(X_train, y_train, batch_size, epochs)
    return model

def Bidirectional_LSTM(X_train, y_test):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model

def Bidirectional_LSTM_lookahead(X_train, y_test):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
    lookahead.inject(model)
    return model

# define baseline model
def DNN(X_train, Y_train):
	# create model
    model = Sequential()
    model.add(Dense(210, input_dim=X_train.shape[1], init='normal', activation='relu')) # 20
    model.add(Dropout(0.412660494242836))
    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    model.add(Dense(185, activation='relu'))
    model.add(Dropout(0.412660494242836))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(3, activation='sigmoid'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adagrad(lr=0.0019214161130842798)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    plot_model(model, to_file='../result/DNN.png', show_shapes=True, show_layer_names=True) # plot my model
    return model

# define baseline model
def DNN_lookahead(X_train, Y_train):
	# create model
    model = Sequential()
    model.add(Dense(495, input_dim=X_train.shape[1], init='normal', activation='relu')) # 20
    model.add(Dropout(0.11368248736379467))
    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    model.add(Dense(443, activation='relu'))
    model.add(Dropout(0.11368248736379467))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(3, activation='softmax'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adagrad(lr=0.00028755003327360373)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
    lookahead.inject(model)
    
    plot_model(model, to_file='../result/DNN.png', show_shapes=True, show_layer_names=True) # plot my model
    return model

def getAccuracy(prediction, y_test): ### prediction and y_test are both encoded.
    sample_size = prediction.shape[0]
    col_num = prediction.shape[1]
    correct_num = 0
    wrong_num = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(sample_size):
        cur_row = prediction[i,:]
        max = 0
        max_id = 0
        res_id = 0 
        for j in range(col_num):
            if cur_row[j] > max:
                max = cur_row[j]
                max_id = j
        for k in range(col_num):
            if y_test[i, k] == 1:
                res_id = k
                break
        if res_id == max_id:
            #print("result id")
            #print(res_id)
            correct_num = correct_num + 1
            if res_id == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            wrong_num = wrong_num + 1
            if res_id == 0:
                fn = fn + 1
            else:
                fp = fp + 1
    accuracy = float(correct_num) / sample_size
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    TNR = float(tn) / (tn + fp)
    F = 2 * float(precision * recall) / (precision + recall) 
    return accuracy, precision, recall, TNR, F

def getAccuracyMulti(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    F1 = metrics.f1_score(y_true, y_pred, average='weighted') 
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    fbeta = metrics.fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    return accuracy, precision, recall, F1, fbeta

def transformResult(result):
    sample_size = result.shape[0]
    col_num = result.shape[1]
    result_list = list()
    for i in range(sample_size):
        cur_row = result[i,:]
        max = 0
        max_id = 0
        for j in range(col_num):
            if cur_row[j] > max:
                max = cur_row[j]
                max_id = j
        result_list.append(max_id)
    return result_list

def plotTrainingAccuracy(history):
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
def plotTrainingLoss(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
def plotRoc(y_predict, y_label):
    # roc for unigram
    fprUni, tprUni, _ = roc_curve(y_predict, y_label)
    roc_aucUni = auc(fprUni, tprUni)
    
    plt.figure()
    lw = 2
    plt.plot(fprUni, tprUni, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_aucUni)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def transform(pred):
    pr_size = len(pred)
    res = np.zeros((pr_size, 3))
    for i in range(pr_size):
        curLabel = int(pred[i])
        if curLabel == 0:
            res[i, 0] = 1
        elif curLabel == 1:
            res[i, 1] = 1
        else:
            res[i, 2] = 1
    return res