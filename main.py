from model import *
from model_helper import *
from training import train
from data import load_logic_gates, new_load_housing_data, new_load_mnist_data


def run_logic_gates():
	"""logic gate example with pre-set hyper-parameters and network architecture"""
	epochs = 100
	loss_average = False
	batch_average = False
	batch_size = 1
	learning_rate = 1

	train_data = load_logic_gates(logic="xor")
	model = DenseModel(layer_config=[10, 2], activation_functions=[sigmoid, sigmoid], classification=True, error_function=cross_entropy, learning_rate=learning_rate, batch_average=batch_average, loss_average=loss_average)
	model = train(model=model, train_data=train_data, test_data=False, val_data=False, epochs=epochs, show_epochs=1, accuracy_measure=hit_or_miss)
	
def run_california_housing():
	"""california housing regression example with pre-set hyper-parameters and network architecture"""
	epochs = 20
	loss_average = False
	batch_average = False
	batch_size = 16
	learning_rate = 0.001

	train_data, test_data, val_data = new_load_housing_data(train_test_split=0.6 , validation=0.5, batch_size=batch_size)
	model = DenseModel(layer_config=[10, 20, 10, 1], activation_functions=[sigmoid, relu, relu, relu], classification=False, error_function=mean_squared_error, learning_rate=learning_rate, batch_average=batch_average, loss_average=loss_average)
	model = train(model=model, train_data=train_data, test_data=test_data, val_data=val_data,epochs=epochs, show_epochs=1, accuracy_measure=r_2)
	
def run_mnist():
	"""mnist classification example with pre-set hyper-parameters and network architecture"""
	epochs = 1
	loss_average = False
	batch_average = False
	batch_size = 32
	learning_rate = 0.1

	train_data, test_data, val_data = new_load_mnist_data(train_test_split=0.6, validation=0.5, batch_size=batch_size)
	model = DenseModel(layer_config=[200, 100, 10], activation_functions=[relu, relu, soft_cross], classification=True, error_function=cross_entropy, learning_rate=learning_rate, batch_average=batch_average, loss_average=loss_average)
	model = train(model=model, train_data=train_data, test_data=test_data, val_data=val_data,epochs=epochs, show_epochs=1, accuracy_measure=hit_or_miss)
	
def main():
	# run_logic_gates()
	run_california_housing()
	# run_mnist()
	
if __name__ == '__main__':
	main()