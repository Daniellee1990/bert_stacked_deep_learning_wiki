# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:33:04 2020

@author: xiaodanli
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from plot_precision_recall import plot_average_precision
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import bert_elmo_tri_cnn_feature_extractor
import stacked_models
from deep_learning_models import getAccuracyMulti, plot_confusion_matrix, transform
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

import timeit

bert_elmo_tri_cnn_feature_extractor.get_feature_label(-1)
df = pd.read_csv('../data/all_features_with_label.csv', sep=',',header=None) 
df_feature = df.iloc[:, 1:]
X = df_feature.values
label = df.iloc[:,0]
y = label.values
cols = list(df)
columns_names = ["label"]
for i in range(0, len(cols) - 1):
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

df_removed.to_csv('../result/process_data.csv', index=False)

X = df_removed.drop(['label'], axis=1)
y = df_removed[['label']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n = 10
time_sum = 0
time_sum_lookahead = 0

"""
for i in range(n):
    print("no lookahead")
    print(i)
    start = timeit.default_timer()
    prediction, dataset_blend_train, dataset_blend_test = stacked_models.stacking(X_train, y_train, X_test, y_test, 15, 350)
    stop = timeit.default_timer()
    time_sum += stop - start

for i in range(n):
    print("with lookahead")
    print(i)
    start = timeit.default_timer()
    prediction, dataset_blend_train, dataset_blend_test = stacked_models.stacking(X_train, y_train, X_test, y_test, 15, 350, True)
    stop = timeit.default_timer()
    time_sum_lookahead += stop - start

mean_time_sum = time_sum / 10.0
print("Average time is:")
print(mean_time_sum)

mean_time_sum_la = time_sum_lookahead / 10.0
print("Average time with lookahead is:")
print(mean_time_sum_la)
"""

useLookahead = True
prediction, dataset_blend_train, dataset_blend_test = stacked_models.stacking(X_train, y_train, X_test, y_test, 15, 350, useLookahead)

accuracy, precision, recall, F1, fbeta = getAccuracyMulti(prediction, y_test)
print(accuracy)    
print(precision)    
print(recall)    
print(F1)    
print(fbeta)

added_str = ''

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