# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:03:14 2020

@author: xiaodanli
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from plot_precision_recall import plot_average_precision
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import stacked_models
from deep_learning_models import getAccuracyMulti, plot_confusion_matrix, transform

df_removed = pd.read_csv('../result/process_data.csv') 
X = df_removed.drop(['label'], axis=1)
y = df_removed[['label']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

useLookahead = False
prediction, dataset_blend_train, dataset_blend_test = stacked_models.stacking(X_train, y_train, X_test, y_test, 15, 350, useLookahead)

accuracy, precision, recall, F1, fbeta = getAccuracyMulti(prediction, y_test)
print(accuracy)    
print(precision)    
print(recall)    
print(F1)    
print(fbeta)

f = open('../result/metrics.txt', 'w')
print(accuracy, file = f)    
print(precision, file = f)    
print(recall, file = f)    
print(F1, file = f)    
print(fbeta, file = f)

added_str = ' with Adagrad'

if useLookahead:
    added_str = ' with lookahead optimizer'

cnf_matrix_1 = confusion_matrix(y_test, prediction)
plt.figure()
class_names = np.array([['Good'], ['Medium'], ['Low']])
plot_confusion_matrix(cnf_matrix_1, classes=class_names, normalize=True,title='Confusion matrix for stacked model' + added_str)
plt.savefig('../result/Confusion matrix for stacked model accuracy.png', dpi=300)
plt.show()

y_test_t = transform(y_test['label'].tolist())
prediction_t = transform(prediction)
plot_average_precision(y_test_t, prediction_t, ' of Stacked model' + added_str)