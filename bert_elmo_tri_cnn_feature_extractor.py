# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:32:07 2019

@author: xiaodanli
"""
import json
import statistics
import numpy as np
import pandas as pd
from keras.utils import np_utils
from bert_serving.client import BertClient
import read_labels
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
#import xlnet
from numpy import array

"""
def triple_cnns_extractor(lens, wikiLines, y):
    K.clear_session()      
    len_mean = statistics.mean(lens)
    unique_labels = set(y)
    output_classes = len(unique_labels)
    trainY = np_utils.to_categorical(y)
    tokenizer = triple_cnn_extractor.create_tokenizer(wikiLines)
    ### extract features from triple cnn
    length = int(round(len_mean))
    vocab_size = len(tokenizer.word_index) + 1
    trainX = triple_cnn_extractor.encode_text(tokenizer, wikiLines, length)
    model, merged = triple_cnn_extractor.define_model(length, vocab_size, output_classes)
    model.fit([trainX,trainX,trainX], trainY, epochs=7, batch_size=15)
    
    dense1_layer_model = triple_cnn_extractor.Model(inputs=model.input,
                                         outputs=model.get_layer('Dense_1').output)
    return dense1_layer_model, trainX
"""

def elmo_vectors(x):
  K.clear_session()
  elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

def elmo_feature_extractor(source_text):
    train = pd.DataFrame({'clean_txt':source_text})
    list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]
    # Extract ELMo embeddings
    elmo_train = [elmo_vectors(x['clean_txt']) for x in list_train]
    total_elmo_feature = np.vstack(elmo_train)
    return total_elmo_feature

def bert_feature_extractor(source_text):
    K.clear_session()
    bc = BertClient(check_version=False)
    wiki_content_bert_features = bc.encode(source_text)
    return wiki_content_bert_features

def changeLabel(cate):
    if cate == 'FA' or cate == 'A':
        return 0
    elif cate == 'GA' or cate == 'B':
        return 1
    elif cate == 'Start' or cate == 'Stub':
        return 2

def getLabels(labels):
    labels["main_category_code"] = labels["main_category"].apply(lambda x: changeLabel(x))      
    unique_labels = labels.main_category_code.unique()
    id2label = dict(zip(labels.plain_text_index, labels.main_category_code))
    
    data = json.load(open("C:/Users/xiaodanli/Desktop/bert_clean_code/data/wiki_word2content.json"))
    wikiLines = list()
    max_len = 0
    lens = list()
    y = list()
    matched_fileID = list()
    i = 0
    for key, value in data.items():
        fileID = int(key.split("/")[-1])
        if fileID not in id2label.keys():
            continue
        matched_fileID.append(fileID)
        wikiLines.append(value)
        lens.append(len(value))
        curLabel = id2label[fileID]
        y.append(curLabel)
        if max_len < len(value):
            max_len = len(value)
        i = i + 1
    
    keys = id2label.keys()
    unmatched = keys - set(matched_fileID)
    y_array = np.array(y)
    
    print("Number of labels")
    print(i)
    
    return y_array, unmatched, wikiLines

def get_feature_label(tested_length):
    ### get graph based feature and edit history based features
    graph_history_features, labels, plain_text_index = read_labels.read_wiki_graph_history_features('wiki_feature.csv')
    y_labels, unmatched, wikiLines = getLabels(labels, tested_length)
    
    #sample_size = 2000
    graph_history_features = graph_history_features[graph_history_features.plain_text_index != list(unmatched)[0]]
    graph_history_features = graph_history_features.head(tested_length).reset_index(drop=True)
    graph_history_features_matrix = graph_history_features.as_matrix()
    graph_history_features_with_label = np.column_stack([y_labels, graph_history_features_matrix])
    np.savetxt("../result/graph_history_extracted_all_features.csv", graph_history_features_with_label, delimiter=",")
    print("Graph history feature extraction is done")
    
    """
    y = y[:sample_size]
    graph_history_features_matrix = graph_history_features_matrix[:sample_size,:]
    wikiLines = wikiLines[:sample_size]
    """
    
    """
    dense1_layer_model, trainX = triple_cnns_extractor(lens, wikiLines, y)
    triple_cnn_features = dense1_layer_model.predict([trainX,trainX,trainX])
    print("triple cnn features extraction is done")
    """
    
    #### extract from elmo
    source_text = wikiLines[0:tested_length]
    source_text = [' '.join(t[0:150]) for t in source_text]
    
    total_elmo_feature = elmo_feature_extractor(source_text)
    elmo_with_label = np.column_stack([y_labels, total_elmo_feature])
    np.savetxt("../result/elmo_extracted_all_features.csv", elmo_with_label, delimiter=",")
    print("elmo features extraction is done")
    
    #### extract features from bert
    wiki_content_bert_features = bert_feature_extractor(source_text)
    bert_with_label = np.column_stack([y_labels, wiki_content_bert_features])
    np.savetxt("../result/bert_extracted_all_features.csv", bert_with_label, delimiter=",")
    print("bert features extraction is done")
    
    ### extract features from xlnet
    #uni_xlnet, bi_xlnet = [xlnet.xlnet_extractor(x) for x in source_text[0:20]]
    #extracted_all_features = np.concatenate((total_elmo_feature, wiki_content_bert_features, triple_cnn_features, graph_history_features_matrix),axis=1)
    #extracted_all_features.to_csv('all_features.csv')
    
    extracted_all_features_no_cnns = np.concatenate((total_elmo_feature, wiki_content_bert_features, graph_history_features_matrix),axis=1)
    np.savetxt("../result/extracted_all_features_no_cnns.csv", extracted_all_features_no_cnns, delimiter=",")
    
    feature_label = np.column_stack([y_labels, extracted_all_features_no_cnns])
    np.savetxt("../result/all_features_with_label.csv", feature_label, delimiter=",")
    
if __name__ == "__main__":
    #tested_length = -1
    #get_feature_label(tested_length)
    
    graph_history_features, labels, plain_text_index = read_labels.read_wiki_graph_history_features('wiki_feature.csv')
    y_labels, unmatched, wikiLines = getLabels(labels, -1)