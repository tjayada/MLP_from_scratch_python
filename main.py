import math
import random
from model import *
from model_helper import *
from training import train

from data import load_logic_gates, load_housing_data, new_load_housing_data, load_mnist_data, new_load_mnist_data



def main():
	epochs = 10
	loss_average = False
	batch_average = False
	batch_size = 32
	learning_rate = 0.001


	#X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
	
	#X,y = load_logic_gates("and")
	#train_data, test_data, _ = new_load_mnist_data(train_test_split=0.01, validation=0.8)

	#train_data, test_data = load_mnist_data(train_test_split=0.001)
	#train_data, _ = new_load_housing_data()
	train_data, test_data, val_data = new_load_housing_data(train_test_split=0.6 , validation=0.5, batch_size=batch_size)
	#X,y = train_data
	
	# just some provisional hyper parameters
	

	#model = DenseModel(layer_config=[100, 200, 10], activation_functions=[relu, relu, linear], classification=True, error_function=cross_entropy)
	model = DenseModel(layer_config=[10, 20, 10, 1], activation_functions=[sigmoid, relu, relu, relu], classification=False, error_function=mean_squared_error, learning_rate=learning_rate, batch_average=batch_average, loss_average=loss_average)

	#print(model([[1,2,3], [4,5,6], [1,2,3]], [[1], [2], [1]]))
	#print(model(X[0], [[1]]))
	#print(X[0])
	#print(y[0])
	#model = train(model, X=X, y=y, epochs=epochs, show_epochs=1, accuracy_measure=r_2)
	model = train(model=model, train_data=train_data, test_data=test_data, val_data=val_data,epochs=epochs, show_epochs=1, accuracy_measure=r_2)
	#model = train(model, , y=y, epochs=epochs, show_epochs=1, accuracy_measure=hit_or_miss)
	#model = train(model=model, train_data=train_data, test_data=test_data, epochs=epochs, show_epochs=1, accuracy_measure=hit_or_miss)


	return model
 		




if __name__ == '__main__':
	trained_model = main()
