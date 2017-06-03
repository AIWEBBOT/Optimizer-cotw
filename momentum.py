from datetime import datetime
import numpy as np
import copy

class Momentum:
    def __init__(self, syn0, syn1):
        # Hyperparameters
        self.__m = 0.9
        self.__learing_rate = 0.0001

        # initialize momentum
        self.__m_syn0 = np.zeros_like(syn0)
        self.__m_syn1 = np.zeros_like(syn1)

        # get a copy of weight
        self.__syn0 = copy.deepcopy(syn0)
        self.__syn1 = copy.deepcopy(syn1)

    def get_weight(self):
        return (self.__syn0, self.__syn1)

    def train(self, t, l1_grad, l2_grad):
        # momentum
        self.__m_syn1 = self.__m * self.__m_syn1 + self.__learing_rate * l2_grad
        self.__m_syn0 = self.__m * self.__m_syn0 + self.__learing_rate * l1_grad

        # apply
        self.__syn1 += self.__m_syn1
        self.__syn0 += self.__m_syn0
