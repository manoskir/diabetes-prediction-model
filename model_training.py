import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_prep import load_data

X_train, y_train, X_test, y_test, X_val, y_val = load_data("diabetes_prediction_dataset.csv")

#here's the helper code that will help me construct model architectures for model training optimization
def get_models(min_num_layers: int,
               max_num_layers: int,
               min_num_nodes: int,
               max_num_nodes: int,
               node_step_size: int,
               hidden_layer_activation: str = 'relu',
               num_nodes_at_output: int = 2,
               output_layer_activation: str = 'linear') -> list:

    node_options = list(range(min_num_nodes, max_num_nodes+1, node_step_size))
    layer_options = list(range(min_num_layers, max_num_layers+1))
    #print(layer_options)
    '''
    in this part of the function I put together the possible architectures starting from the layer and going up to the node options
    for instance an 1 layer architecture can have 3,4,5,6,7 layera or a 2 layer architecture can have the potential combinations of two 3,4,5,6,7 arrays and so on
    this for loop puts together those combinations
    '''
    layer_possibilities = []

    for num in layer_options:
        layer_possibilities.append(num*[node_options])

    #print(layer_possibilities)

    '''
    in this part of the function the actual node permutations are computed using the combinations produced above this is why I use the product function
    i.e. for a 1 layer architecture I have (3,) (4,)...(7,) but for a 2 layer architecture I have to find the product of two (3,4,5,6,7) arrays and so on
    '''
    layer_node_permutations = []

    for item in layer_possibilities:
        layer_node_permutations.append(list(itertools.product(*item)))
    
    #print(layer_node_permutations)

    models = []

    #the first for loop is just there so that I can pick each of the potential architectures
    for permutation in layer_node_permutations:
    
        #at this level of the for loop I pick each potential architecture and create just the model (e.g. a (3,) is an 1 layer - 3 nodes while a (3,6) is a two layer where one has 3 and the other 6 nodes
        for architecture in permutation:
            model_name = ''
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(8,)))
        
            #at this level I construct the layers with the nodes
            for nodes_in_layer in architecture:
                model.add(tf.keras.layers.Dense(nodes_in_layer, activation = hidden_layer_activation))
                model_name += f'dense{nodes_in_layer}_'
            
            model.add(tf.keras.layers.Dense(num_nodes_at_output, activation = output_layer_activation))
            model._name = model_name[:-1]
        
            models.append(model)
    
    return models

models = get_models(min_num_layers = 1,
                    max_num_layers=3,
                    min_num_nodes=3,
                    max_num_nodes=7,
                    node_step_size=1,
                    hidden_layer_activation='relu',
                    num_nodes_at_output=2,
                    output_layer_activation='linear')

#function that compiles the model, runs the model training and then returns a data frame with the training results
def optimize(models: list,
            X_train: np.array,
            y_train: np.array,
            X_test: np.array,
            y_test: np.array,
            epochs: int = 10,
            verbose: int = 0) -> pd.DataFrame:
    
    results = []
    
    #helper function nested under the main function that runs the model training
    def train(model: tf.keras.Sequential) -> dict:
        
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(0.01),
            metrics=['accuracy']
        )
        
        model_history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose = verbose)
        
        return {
            'model_name': model.name,
            'test_accuracy': model_history.history['val_accuracy'][-1]
        }
    
    for model in models:
        
        try:
            print(model.name, end='...')
            res = train(model=model)
            results.append(res)
        except Exception as e:
            print(f'{model.name} --> {str(e)}')
            
    return pd.DataFrame(results)

optimization_results = optimize(
    models = models,
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test,
    epochs = 10,
    verbose = 1)

print(optimization_results.sort_values(by='test_accuracy', ascending = False))