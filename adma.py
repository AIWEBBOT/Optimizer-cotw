from datetime import datetime
import numpy as np
import copy

class Adma:
    def __init__(self, syn0, syn1):
        # Hyperparameters
        self.__b1 = 0.9
        self.__b2 = 0.999
        self.__learing_rate = 0.001
        self.__epsilon = 1e-8

        # initialize momentum
        self.__m_syn0 = np.zeros_like(syn0)
        self.__m_syn1 = np.zeros_like(syn1)
        self.__v_syn0 = np.zeros_like(syn0)
        self.__v_syn1 = np.zeros_like(syn1)

        # get a copy of weight
        self.__syn0 = copy.deepcopy(syn0)
        self.__syn1 = copy.deepcopy(syn1)

    def get_weight(self):
        return (self.__syn0, self.__syn1)

    def train(self, t, l1_grad, l2_grad):
        # first momentum
        self.__m_syn1 = self.__b1 * self.__m_syn1 + (1 - self.__b1) * l2_grad
        self.__m_syn0 = self.__b1 * self.__m_syn0 + (1 - self.__b1) * l1_grad

        # second momentum
        self.__v_syn1 = self.__b2 * self.__v_syn1 + (1 - self.__b2) * l2_grad**2
        self.__v_syn0 = self.__b2 * self.__v_syn0 + (1 - self.__b2) * l1_grad**2

        # bias corrected first momentum
        self.__mbc_syn0 = self.__m_syn0 / (1 - self.__b1**t)
        self.__mbc_syn1 = self.__m_syn1 / (1 - self.__b1**t)

        # bias corrected second momentum
        self.__vbc_syn0 = self.__v_syn0 / (1 - self.__b2**t)
        self.__vbc_syn1 = self.__v_syn1 / (1 - self.__b2**t)

        # Apply momentum correction
        self.__syn1 += self.__learing_rate * self.__mbc_syn1 / (np.sqrt(self.__vbc_syn1) + self.__epsilon)
        self.__syn0 += self.__learing_rate * self.__mbc_syn0 / (np.sqrt(self.__vbc_syn0) + self.__epsilon)
