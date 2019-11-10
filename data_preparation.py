#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:42:11 2019

@author: lixiaodan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def data_preparation(path):
    df = pd.read_csv(path, sep=',',header=None) 
    df_feature = df.iloc[:, 1:]
    X = df_feature.values
    label = df.iloc[:,0]
    y = label.values
    cols = list(df)
    columns_names = ["label"]
    for i in range(0, 1817):
        columns_names.append("Column " + str(i))
    df.columns = columns_names
    
    #missing data
    total = df_feature.isnull().sum().sort_values(ascending=False)
    percent = (df_feature.isnull().sum()/df_feature.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    threshold = 0.80
    corr_matrix = df[df['label'].notnull()].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
    df_removed = df.drop(to_drop, axis=1)
    
    X = df_removed.drop(['label'], axis=1)
    y = df_removed[['label']]
    y_encode = pd.get_dummies(y.label)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encode, test_size=0.33, random_state=42)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train.values, y_test.values

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_preparation('all_features_with_label.csv')
