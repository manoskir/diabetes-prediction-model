import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(dataset):
	
	#loading data in str format from csv
	raw_data = np.loadtxt(dataset, delimiter = ",", dtype = str)

	#removing header row
	diabetes_data = np.delete(raw_data, 0, 0)

	#replacing male and female values with 0 and 1 and removing the other values
	diabetes_data[diabetes_data == 'Female'] = 0
	diabetes_data[diabetes_data == 'Male'] = 1
	diabetes_data = np.delete(diabetes_data, np.where(diabetes_data[:,0]=='Other'), axis = 0)

	#mapping out the smoking status column to new values. No Info - 0, current - 1, ever - 2, former - 3, never - 4, not current - 5
	diabetes_data[diabetes_data == 'No Info'] = 0
	diabetes_data[diabetes_data == 'current'] = 1
	diabetes_data[diabetes_data == 'ever'] = 2
	diabetes_data[diabetes_data == 'former'] = 3
	diabetes_data[diabetes_data == 'never'] = 4
	diabetes_data[diabetes_data == 'not current'] = 5

	#transforming the type of numpy data from str to float
	diabetes_data = diabetes_data.astype(np.float)

	#defining the X array
	diabetes_data_X = diabetes_data[:,:8]

	#defining the y array (labels)
	diabetes_data_y = diabetes_data[:,8:]

	#splitting between train and a temporary set to further split it into test and validation
	X_train, X_temp, y_train, y_temp = train_test_split(
	    diabetes_data_X, diabetes_data_y, test_size = 0.33, random_state = 42)

	X_test, X_val, y_test, y_val = train_test_split(
	    X_temp, y_temp, test_size = 0.5, random_state = 42)

	return(X_train, y_train, X_test, y_test, X_val, y_val)

X_train, y_train, X_test, y_test, X_val, y_val = load_data("diabetes_prediction_dataset.csv")

print(X_train[:3])
print(X_train.shape)
print(y_train[:3])
print(y_train.shape)
print(X_test[:3])
print(X_test.shape)
print(y_test[:3])
print(y_test.shape)
print(X_val[:3])
print(X_val.shape)
print(y_val[:3])
print(y_val.shape)