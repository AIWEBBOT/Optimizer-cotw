from datetime import datetime
import numpy as np
import copy

class Math:
    @staticmethod
    def sigmoid(x, deriv=False):
        '''
        Sigmoïd function
        :param x: np.array
        :param deriv: derivate wanted ?
        :return:
        '''
        if deriv:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x, deriv=False):
        '''
        Rectifier function
        :param x: np.array
        :param deriv: derivate wanted ?
        :return:
        '''
        if deriv:
            return np.ones_like(x) * (x > 0)

        return x * (x > 0)
