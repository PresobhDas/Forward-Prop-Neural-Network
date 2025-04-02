import tensorflow as tf
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot(history):
    fig, ax1 = plt.subplots(1, constrained_layout=True, figsize=(12,4))
    ax1.plot(history)
    plt.show()

def pre_process(df):
    df = df.drop(columns=['customer name', 'customer e-mail', 'country'])
    x_train = df[['age','annual Salary', 'credit card debt', 'net worth']]
    y_train = df[['car purchase amount']].to_numpy()
    x_train = x_train.to_numpy()
    norm_layer_x = Normalization(axis=-1)
    norm_layer_y = Normalization(axis=-1)
    norm_layer_x.adapt(x_train)
    norm_layer_y.adapt(y_train)
    return norm_layer_x(x_train).numpy(), norm_layer_y(y_train).numpy(), norm_layer_x, norm_layer_y

df = pd.read_csv('car_purchasing.csv', encoding='ISO-8859-1')
x_train, y_train, norm_layer_x, norm_layer_y = pre_process(df)
dataCount, featCount = x_train.shape
model = Sequential(
                    [
                        tf.keras.Input(shape=(4,)), 
                        Dense(10, activation='relu', name = 'layer1'),
                        Dense(10, activation='relu', name = 'layer2'),
                        Dense(1, name = 'layer3')
                    ], name = 'Das'
                    )
model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
)

history = model.fit(x_train, y_train, epochs = 100)
plot(history.history['loss'])



