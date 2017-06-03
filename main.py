from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import copy

from utils.dataset import Dataset
from utils.generator import Generator
from utils.math import Math

from adma import Adma
from sgd import SGD
from momentum import Momentum


# Global Hyperparamaters
max_epochs = 2000
batch_size = 60

# Load and prepare data
date, latitude, longitude, magnitude = Dataset.load_from_file("database.csv")
data_size = len(date)
vectorsX, vectorsY = Dataset.vectorize(date, latitude, longitude), magnitude.reshape((data_size, 1))

# Get Batcher
batch_gen = Generator.gen_random_batch(batch_size, vectorsX, vectorsY)

# randomly initialize our weights with mean 0
syn0 = 2 * np.random.standard_normal((vectorsX.shape[1], 32)) / 10
syn1 = 2 * np.random.standard_normal((32, vectorsY.shape[1])) / 10

# Init trainer table and datalog
trainers = [SGD(syn0, syn1), Momentum(syn0, syn1), Adma(syn0, syn1)]
datalog = []

# Train model
x = x = np.arange(1, max_epochs)
for t in x:
    # Get Batch
    batch = next(batch_gen)

    for trainer in trainers:
        syn0, syn1 = trainer.get_weight()
        
        # feed forward
        l0 = batch[0]
        l1 = Math.sigmoid(np.dot(l0, syn0))
        l2 = Math.relu(np.dot(l1, syn1))

        # l2 error & grad
        l2_error = batch[1] - l2
        l2_delta = l2_error * Math.relu(l2, deriv=True)
        l2_grad = l1.T.dot(l2_delta)

        # l1 error & grad
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * Math.sigmoid(l1, deriv=True)
        l1_grad = l0.T.dot(l1_delta)

        # Update weigths
        trainer.train(t, l1_grad, l2_grad)

        # log error
        datalog.append(np.sqrt(np.mean(l2_error**2)))

#reshape and plot datalog
datalog = np.array(datalog).reshape([len(datalog) // 3, 3])
datalog = np.swapaxes(datalog, 0, 1)
for i in range(3):
    plt.semilogy(x, datalog[i], label = trainers[i].__class__.__name__)
plt.xlabel('epochs')
plt.ylabel('avg rms error')
plt.legend()
plt.show()
