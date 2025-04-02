import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Normalization

def back_prop():
    # dc   dc   da   dz     dc   dc    dc        dc   dz
    # -- = -- * -- * --  ;  -- = --  ; --      = -- * --
    # dw   da   dz   dw     db   dz    da(back)  dz   da(prev)

    for i in range(len(layers)-1, 0 , -1):
        if i == len(layers)-1:
            dc_dz = (cache[i] - y_train) / dataCount
        else:
            da_dz = cache[i] * (1 - cache[i])
            dc_dz = dc_da_back * da_dz     
        dz_dw = cache[i-1].T
        dc_dw = np.matmul(dc_dz, dz_dw)
        dc_db = np.sum(dc_dz, axis=1, keepdims=True)
        dc_da_back = np.matmul(weights[i].T, dc_dz)
        weights[i] -= (alpha * dc_dw)
        bias[i] -= (alpha*dc_db)

def forward_prop(x_train):
    inp = x_train
    for layer in range(1, len(weights)):
        if layer == len(weights)-1:
            inp = np.matmul(weights[layer], inp) + bias[layer]
        else:
            inp = sigmoid(np.matmul(weights[layer], inp) + bias[layer])
        cache[layer] = inp
    return inp

def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

def pre_process(df):
    df = df.drop(columns=['customer name', 'customer e-mail', 'country'])
    x_train = df[['age','annual Salary', 'credit card debt', 'net worth']]
    y_train = df[['car purchase amount']].to_numpy()
    x_train = x_train.to_numpy()
    norm_layer_x = Normalization(axis=-1)
    norm_layer_y = Normalization(axis=-1)
    norm_layer_x.adapt(x_train)
    norm_layer_y.adapt(y_train)
    return norm_layer_x(x_train).numpy(), norm_layer_y(y_train).numpy()

df = pd.read_csv('car_purchasing.csv', encoding='ISO-8859-1')
x_train, y_train = pre_process(df)
y_train = y_train.T
dataCount, dataFeat = x_train.shape
x_train = x_train.T
layers = [4,6,6,1]
weights = []
bias = []
cache = {}
weights.append(0)
bias.append(0)
cache[0] = x_train
alpha = 1.0e-2
epoch = 400

for i in range(1,len(layers)):
    weights.append(np.random.randn(layers[i], layers[i-1]))
    bias.append(np.random.randn(layers[i], 1))

for i in range(epoch):
    y_pred = forward_prop(x_train)
    cost = np.sum((y_pred.reshape(-1,1)-y_train.reshape(-1,1)) ** 2) / (2 * dataCount)
    back_prop()
    print(cost)






