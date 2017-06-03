from datetime import datetime
import numpy as np
import copy

class SGD:
    def __init__(self, syn0, syn1):
        # Hyperparameters
        self.__learing_rate = 0.0001
        
        # get a copy of weight
        self.__syn0 = copy.deepcopy(syn0)
        self.__syn1 = copy.deepcopy(syn1)

    def get_weight(self):
        return (self.__syn0, self.__syn1)

    def train(self, t, l1_grad, l2_grad):
        self.__syn0 += self.__learing_rate * l1_grad
        self.__syn1 += self.__learing_rate * l2_grad
