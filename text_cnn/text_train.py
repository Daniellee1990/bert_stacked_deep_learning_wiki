# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:26:09 2020

@author: xiaodanli
"""

from __future__ import print_function
import  tensorflow as tf
import numpy as np
import text_model
import loader
from sklearn import metrics
import os
import time
import train_word2vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas

import sys
sys.path.append("..")
from bert_elmo_tri_cnn_feature_extractor import getLabels
from deep_learning_models import getAccuracyMulti, plot_confusion_matrix

def evaluate(sess, x_, y_, model):
    data_len = len(x_)
    batch_eval = loader.batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0, model)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def feed_data(x_batch, y_batch, keep_prob, model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob:keep_prob
    }
    return feed_dict

def read_wiki_graph_history_features(file):
    df = pandas.read_csv(file)
    labels = df[['plain_text_index', 'main_category']]
    plain_text_index = df.plain_text_index
    df = df.drop(['main_category', 'sub_category', 'f1', 'name'], axis=1)
    return df, labels, plain_text_index

def TexCNN_train_evaluate():
    train_word2vec.train_word2vec()
    print('Configuring CNN model...')
    config = text_model.TextConfig()
    loader.build_vocab(config.vocab_filename, config.vocab_size)

    #read vocab and categories
    categories,cat_to_id = loader.read_category()
    words,word_to_id = loader.read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    # trans vector file to numpy file
    if not os.path.exists(config.vector_word_npz):
        loader.export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_trianing = loader.get_training_word2vec_vectors(config.vector_word_npz)
    
    graph_history_features, labels, plain_text_index = read_wiki_graph_history_features('../../data/wiki_feature.csv')
    y_labels, unmatched, wikiLines = getLabels(labels)
    model = text_model.TextCNN(config)
    
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/textcnn'
    save_dir = './checkpoints/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    print("Loading training and validation data...")
    start_time = time.time()
    X, y = loader.process_file(y_labels, wikiLines, word_to_id, cat_to_id, max_length=600)
    x_t, x_test, y_t, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_t, y_t, test_size=0.33, random_state=42)
    
    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    best_val_accuracy = 0
    last_improved = 0  # record global_step at best_val_accuracy
    require_improvement = 1000  # break training if not having improvement over 1000 iter
    flag=False

    for epoch in range(config.num_epochs):
        batch_train = loader.batch_iter(x_train, y_train, config.batch_size)
        start = time.time()
        print('Epoch:', epoch + 1)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.keep_prob, model)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                    merged_summary, model.loss,
                                                                                    model.acc], feed_dict=feed_dict)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                val_loss, val_accuracy = evaluate(session, x_val, y_val, model)
                writer.add_summary(train_summaries, global_step)

                # If improved, save the model
                if val_accuracy > best_val_accuracy:
                    saver.save(session, save_path)
                    best_val_accuracy = val_accuracy
                    last_improved=global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                        global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                        (end - start) / config.print_per_batch,improved_str))
                start = time.time()

            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break
        if flag:
            break
        config.lr *= config.lr_decay
    
    print("Loading test data...")
    t1=time.time()

    session=tf.Session()
    session.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    saver.restore(sess=session,save_path=save_path)

    print('Testing...')
    test_loss,test_accuracy = evaluate(session,x_test,y_test,model)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(test_loss, test_accuracy))

    batch_size=config.batch_size
    data_len=len(x_test)
    num_batch=int((data_len-1)/batch_size)+1
    y_test_cls=np.argmax(y_test,1)
    y_pred_cls=np.zeros(shape=len(x_test),dtype=np.int32)

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        feed_dict={
            model.input_x:x_test[start_id:end_id],
            model.keep_prob:1.0,
        }
        y_pred_cls[start_id:end_id]=session.run(model.y_pred_cls,feed_dict=feed_dict)

    #evaluate
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    
    accuracy, precision, recall, F1, fbeta = getAccuracyMulti(y_pred_cls, y_test_cls)
    print(accuracy)    
    print(precision)    
    print(recall)    
    print(F1)    
    print(fbeta)

    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)
    
    text_cnn_matrix = confusion_matrix(y_test_cls, y_pred_cls)
    plt.figure()
    class_names = np.array([['High'], ['Medium'], ['Low']])
    plot_confusion_matrix(text_cnn_matrix, classes=class_names, normalize=True,title='Confusion matrix for TextCNN')
    plt.savefig('./result/Confusion matrix for TextCNN.png', dpi=300)
    plt.show()

    print("Time usage:%.3f seconds...\n"%(time.time() - t1))

if __name__ == '__main__':
    TexCNN_train_evaluate()