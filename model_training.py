import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_prep import load_data

X_train, y_train, X_test, y_test, X_val, y_val = load_data("diabetes_prediction_dataset.csv")

#here's the helper code that will help me construct model architectures for model training optimization
def get_models(num_layers: int,
              min_num_nodes: int,
              max_num_nodes: int,
              node_step_size: int,
              hidden_layer_activation: str = 'relu',
              num_nodes_at_output: int = 2,
              output_layer_activation: str = 'linear') -> list:
    
    node_options = list(range(min_num_nodes, max_num_nodes+1))
    layer_possibilities = num_layers*[node_options]
    layer_node_permutations = list(itertools.product(*layer_possibilities))
    
    #list that holds all the models that will be created based on the nodes and layers permutations
    models = []

    #forloops that combine layers and nodes to create the various model architectures. This is the model level where we setup the model and the input layer
    for permutation in layer_node_permutations:
    
        model_name = ''
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(8,)))
    	
    	#this is the layer level where for each layer we add the layers and the corresponding nodes
        for nodes_in_layer in permutation:
        
            model.add(tf.keras.layers.Dense(nodes_in_layer, activation = 'relu'))
            model_name += f'dense{nodes_in_layer}_'
        
        #finally we have the output layer and we append the model in the list
        model.add(tf.keras.layers.Dense(2, activation = 'linear'))
        model._name = model_name[:-1]
    
        models.append(model)
    
    return models

models = get_models(num_layers = 2,
                   min_num_nodes=3,
                   max_num_nodes=7,
                   node_step_size=1,
                   hidden_layer_activation='relu',
                   num_nodes_at_output=2,
                   output_layer_activation='linear')

for model in models:
	print(model.summary())