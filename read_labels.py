# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:12:13 2019

@author: xiaodanli
"""

import pandas

def read_wiki_graph_history_features(file):
    df = pandas.read_csv(file)
    labels = df[['plain_text_index', 'main_category']]
    plain_text_index = df.plain_text_index
    df = df.drop(['main_category', 'sub_category', 'f1', 'name'], axis=1)
    return df, labels, plain_text_index

if __name__ == '__main__':
    df, labels, plain_text_index = read_wiki_graph_history_features('wiki_feature.csv')
