#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:13:00 2019

@author: lixiaodan
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from keras import backend as K
from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

import data_preparation as dp

print(tf.__version__)
print(skopt.__version__)

dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_filter1 = Integer(low=4, high=32, name='num_filter1')
dim_num_kernel_size1 = Integer(low=3, high=11, name='num_kernel_size1')
dim_activation = Categorical(categories=['relu', 'sigmoid', 'softmax'],
                             name='activation')
dim_num_filter2 = Integer(low=3, high=11, name='num_kernel_size2')
dim_num_kernel_size2 = Integer(low=3, high=5, name='num_kernel_size2')

dimensions = [dim_learning_rate,
              dim_num_filter1,
              dim_num_kernel_size1,
              dim_num_filter2,
              dim_num_kernel_size2,
              dim_activation]

def log_dir_name(learning_rate, num_filter1,
                 num_kernel_size1, num_filter2,
                 num_kernel_size2, activation):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_filter1,
                       num_kernel_size1,
                       num_filter2,
                       num_kernel_size2,
                       activation)

    return log_dir

x_train, x_test, y_train, y_test = dp.data_preparation('all_features_with_label.csv')
input_shape = (x_train.shape[1], 1)
# Number of classes
num_classes = 3
    
def create_model(learning_rate, num_filter1,
                 num_kernel_size1, num_filter2,
                 num_kernel_size2, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    
    model = Sequential()
    model.add(Conv1D(input_shape=input_shape, filters=num_filter1, kernel_size=num_kernel_size1, 
                     padding='same', activation=activation, name='layer_conv1'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(kernel_size=num_kernel_size2, filters=num_filter2,
                     padding='same', activation=activation, name='layer_conv2'))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

path_best_model = '19_best_model.keras'
best_accuracy = 0.0

default_parameters = [1e-5, 4, 3, 4, 3, 'relu']

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_filter1,
             num_kernel_size1, num_filter2,
             num_kernel_size2, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_filter1:', num_filter1)
    print('num_kernel_size1:', num_kernel_size1)
    print('num_filter2:', num_filter2)
    print('num_kernel_size2:', num_kernel_size2)
    print('activation:', activation)
    print()
    
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_filter1 = num_filter1,
                         num_kernel_size1=num_kernel_size1,
                         num_filter2 = num_filter2,
                         num_kernel_size2=num_kernel_size2,
                         activation=activation)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_filter1,
                           num_kernel_size1, num_filter2,
                           num_kernel_size2, activation)
    
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)
   
    # Use Keras to train the model.
    x_train, x_test, y_train, y_test = dp.data_preparation('all_features_with_label.csv')
    
    #x_train = data.train.images
    #y_train = data.train.labels
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    validation_data = (x_test, y_test)
    
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=20,
                        batch_size=128,
                        validation_data=validation_data,
                        callbacks=[callback_log])

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)
        
        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

fitness(x=default_parameters)

search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=11,
                            x0=default_parameters)

with open('out.txt', 'w') as f:
    print('best result:', search_result.x, file=f)
