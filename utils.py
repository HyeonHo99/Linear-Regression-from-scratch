import numpy as np
import os
from optim.Optimizer import *


def RMSE(h, y):
    if len(h.shape) > 1:
        h = h.squeeze()
    se = np.square(h - y)
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse


def optimizer(optim_name):
    if optim_name == 'SGD':
        optim = SGD()
    else:
        raise NotImplementedError
    return optim


def load_data(data_name):
    path = os.path.join('./data', data_name)

    if data_name == 'Diabetes':
        train_x, train_y = DiabetesData(path, 'train.csv')
        test_x, test_y = DiabetesData(path, 'test.csv')
    elif data_name == 'Wave':
        train_x, train_y = WaveData(path, 'train.csv')
        test_x, test_y = WaveData(path, 'test.csv')
    else:
        raise NotImplementedError

    return (train_x, train_y), (test_x, test_y)


def DiabetesData(path, filename):
    return load_reg_data(path, filename, target_at_front=False, normalize=True)


def WaveData(path, filename):
    return load_reg_data(path, filename, target_at_front=False, normalize=True)


def load_reg_data(path, filename, target_at_front, normalize=False, shuffle=False):
    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    data = lines[1:]

    data = np.array([[float(f) for f in d] for d in data], dtype=np.float32)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0]
    else:
        x, y = data[:, :-1], data[:, -1]

    num_data = x.shape[0]
    if normalize:
        mins = np.expand_dims(np.min(x, axis=0), 0).repeat(num_data, 0)
        maxs = np.expand_dims(np.max(x, axis=0), 0).repeat(num_data, 0)
        x = (x - mins) / maxs

    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x = np.concatenate((x, bias), axis=1)

    if shuffle:
        perm = np.random.permutation(num_data)
        x = x[perm]
        y = y[perm]

    return x, y
