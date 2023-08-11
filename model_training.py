import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_prep import load_data

X_train, y_train, X_test, y_test, X_val, y_val = load_data("diabetes_prediction_dataset.csv")

#here's the helper code that will help me construct model architectures for model training optimization
num_layers = 2
min_num_nodes = 3
max_num_nodes = 7

node_options = list(range(min_num_nodes, max_num_nodes+1))

layer_possibilities = num_layers*[node_options]

layer_node_permutations = list(itertools.product(*layer_possibilities))
print(layer_node_permutations)


'''
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
    X_train,y_train,
    epochs=10, validation_data=(X_test,y_test)
)
'''