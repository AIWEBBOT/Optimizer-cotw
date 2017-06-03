from datetime import datetime
import numpy as np
import copy

class Generator:
    @staticmethod
    def gen_random_batch(batch_size, X, Y):
        '''
        Generator for random batch
        :param batch_size: size or the returned batches
        :param X: X array
        :param Y: Y array
        :return: random batches of the given size
        '''
        while True:
            index = np.arange(X.shape[0])
            np.random.shuffle(index)

            s_X, s_Y = X[index], Y[index]
            for i in range(X.shape[0] // batch_size):
                yield (X[i * batch_size:(i + 1) * batch_size], Y[i * batch_size:(i + 1) * batch_size])

    @staticmethod
    def get_batch(batch_size, X, Y):
        '''
        Generator to split givens arrays in smaller batches
        :param batch_size: size or the returned batches
        :param X: X array
        :param Y: Y array
        :return: random batches of the given size
        '''
        if X.shape[0] % batch_size != 0:
            print("[/!\ Warning /!\] the full set will not be executed because of a poor choice of batch_size")

        for i in range(X.shape[0] // batch_size):
            yield X[i * batch_size:(i + 1) * batch_size], Y[i * batch_size:(i + 1) * batch_size]
