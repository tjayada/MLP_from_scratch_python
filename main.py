import math
import random
from model import *
from model_helper import *
from training import train

from data import load_logic_gates, load_housing_data, load_mnist_data, new_load_mnist_data



def main():
	#X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
	
	#X,y = load_logic_gates("and")
	#train_data, test_data = new_load_mnist_data(train_test_split=0.2)

	train_data, _ = load_housing_data()
	X,y = train_data
	
	# just some provisional hyper parameters
	epochs = 20
	batch_size = 100

	model = DenseModel([20, 10, 5, 1], [relu, relu, relu, relu], mean_squared_error)

	#print(model([[1,2,3]], [[1]]))

	model = train(model, X=X, y=y, epochs=epochs, show_epochs=1)


	return model
 		




if __name__ == '__main__':
	trained_model = main()
