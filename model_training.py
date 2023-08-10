import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_prep import load_data

X_train, y_train, X_test, y_test, X_val, y_val = load_data("diabetes_prediction_dataset.csv")

model = Sequential(
    [
        Dense(8, activation = 'relu', name = "L1"),
        Dense(4, activation = 'relu', name = "L2"),
        Dense(2, activation = 'linear', name = "L3")
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy']
)

model.fit(
    dd1_X_train,dd1_y_train,
    epochs=10, validation_data=(dd1_X_test,dd1_y_test)
)