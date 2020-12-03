# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:29:53 2020

@author: xiaodanli
"""

import torch


class Config(object):
    """Base configuration class."""

    name = "TextCNN"

    cuda = True

    epochs = 300
    batch_size = 64
    shuffle = True

    learning_rate = 0.001
    learning_momentum = 0.9
    weight_decay = 0.0001
    dropout = 0.5

    embed_dim = 128
    kernel_num = 100
    kernel_sizes = "3,4,5"

    def __init__(self):
        if self.cuda:
            self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

    def dump(self):
        """Display Configurations."""

        print("Configuration:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("\t{:30} = {}".format(a, getattr(self, a)))
        print()