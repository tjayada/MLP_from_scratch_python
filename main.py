import math
import random
from model import *
from model_helper import *
from training import train

from data import load_logic_gates, load_housing_data, new_load_housing_data, load_mnist_data, new_load_mnist_data



def main():
	#X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
	
	#X,y = load_logic_gates("and")
	#train_data, test_data = new_load_mnist_data(train_test_split=0.2)

	#train_data, test_data = load_mnist_data(train_test_split=0.001)
	train_data, _ = new_load_housing_data()
	X,y = train_data
	
	# just some provisional hyper parameters
	epochs = 40
	batch_size = 100

	#model = DenseModel(layer_config=[512, 256, 128, 10], activation_functions=[sigmoid, sigmoid, sigmoid, relu], classification=True, error_function=mean_squared_error)
	model = DenseModel(layer_config=[50, 25, 12, 1], activation_functions=[sigmoid, sigmoid, sigmoid, relu], classification=False,error_function=mean_squared_error)
	#print(model([[1,2,3]], [[1]]))

	model = train(model, X=X, y=y, epochs=epochs, show_epochs=1, accuracy_measure=r_2)


	return model
 		




if __name__ == '__main__':
	trained_model = main()
