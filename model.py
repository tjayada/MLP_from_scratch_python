import math 
import random
random.seed(42)
from model_helper import *


class DenseLayer(object):
	"""single dense layer used in fully connected networks created with DenseModel class"""
	def __init__(self, size, activation_function, learning_rate, batch_average):
		super(DenseLayer, self).__init__()
		self.output_size 			= size
		self.activation_function	= activation_function
		self.learning_rate			= learning_rate
		self.batch_average			= batch_average
		self.inputs 				= None
		self.targets 				= None
		self.weights 				= None
		self.bias 					= None
		self.output					= None
		self.activation				= None
		self.built					= False


	def __call__(self, inputs, targets):
		"""forward pass by dot-product with input and weight matrix through activation function"""
		self.inputs 		= inputs.copy()
		self.targets 		= targets.copy()
		# first time called the input size sets the size of the weight matrix
		# weigths and biases are set using the xavier uniform distribution 
		if not self.built:
			inputs_size 	= len(inputs) if not isinstance(inputs[0], list) else len(inputs[0])
			self.weights 	= [ [random.uniform(- math.sqrt( 6 / (inputs_size + self.output_size )), math.sqrt( 6 / (inputs_size + self.output_size ))) for _ in range(self.output_size)] for _ in range(inputs_size) ]
			self.bias 		= [ random.uniform(- math.sqrt( 6 / (inputs_size + self.output_size )), math.sqrt( 6 / (inputs_size + self.output_size ))) for _ in range(self.output_size) ]
			self.built 		= True

		self.activation 		= mat_mat_dot_product(self.inputs.copy(), self.weights.copy())
		self.activation 		= mat_plus_vec(self.activation.copy(), self.bias.copy())
		self.output 		= self.activation_function(self.activation.copy())
		return self.output
	
	def update(self, error_signal):
		"""update the weights and biases of the layer in the DenseModel backprop-step"""
		# differentiate between activation functions, since softmax derivative is a jacobian 
		if not f"{self.activation_function.__name__}_prime" == "soft_cross_prime":
			if f"{self.activation_function.__name__}_prime" == "softmax_prime":
				error_signal = [ mat_mat_dot_product([error_signal[idx]], jacobian_matrix)[0] for idx, jacobian_matrix in enumerate(function_dictionary[f"{self.activation_function.__name__}_prime"](self.activation)) ]
			else:
				error_signal = mat_mat_multiply(error_signal, function_dictionary[f"{self.activation_function.__name__}_prime"](self.activation))
		
		gradient = mat_mat_dot_product(transpose(self.inputs), error_signal)
		error_signal_out = mat_mat_dot_product(error_signal, transpose(self.weights) )
		
		if self.batch_average:
			weight_gradient = mat_scalar_divide(gradient, len(self.targets))
			bias_gradient = sum_up_matrix_by_rows(error_signal)
			bias_gradient = [sum(bias_gradient) / len(bias_gradient)] * len(self.bias)
			bias_gradient = vec_scalar_divide(bias_gradient, len(self.targets))
		else:
			weight_gradient = gradient
			bias_gradient = sum_up_matrix_by_rows(error_signal)
			bias_gradient = [sum(bias_gradient) / len(bias_gradient)] * len(self.bias)
		
		weight_gradient = mat_scalar_multiply(weight_gradient, self.learning_rate)
		bias_gradient = vec_scalar_multiply(bias_gradient, self.learning_rate)
		self.weights = mat_mat_minus(self.weights, weight_gradient)
		self.bias = vec_minus_vec(self.bias, bias_gradient)

		return error_signal_out
	


class DenseModel(object):
	"""implements a fully connected model using (multiple) instances of the DenseLayer classes"""
	def __init__(self, layer_config, activation_functions, classification = False, error_function = mean_squared_error, learning_rate=0.01, batch_average=False, loss_average=False):
		super(DenseModel, self).__init__()
		self.layer_config 			= layer_config
		self.activation_functions	= activation_functions
		self.classification 		= classification
		self.error_function			= error_function
		self.learning_rate			= learning_rate
		self.batch_average			= batch_average
		self.loss_average			= loss_average
		self.error_function_derivative = function_dictionary[f"{error_function.__name__}_prime"]
		self.inputs 				= None
		self.targets 				= None
		self.layers 				= []
		self.output					= None
		self.prediction				= None
		self.activation				= None
		self.built					= False
		self.activation_functions_derivative = [ function_dictionary[f"{self.activation_functions[i].__name__}_prime"] for i in range(len(self.activation_functions)) ]

	def __call__(self, inputs, targets):
		"""forward pass through network by iterating over layers"""
		self.inputs 		= inputs
		self.targets 		= targets
		# first time called the layers are build using the input sizes for the weight matrices
		if not self.built:
			for i in range(len(self.layer_config)):
				self.layers.append(DenseLayer(self.layer_config[i], self.activation_functions[i], learning_rate=self.learning_rate, batch_average=self.batch_average))
			self.built 		= True
		
		self.output = self.inputs
		for layer in self.layers:
			self.output = layer(self.output, self.targets)
		# in classification tasks return most probable output
		if self.classification:
			self.prediction = [ get_argmax(elem_x) for elem_x in self.output ]
		
		return self.output

	def backpropagation(self):
		"""update all layers by iterating over layers and calling the update function"""
		# differentiat between activation functions of output layer, since softmax & cross-entropy can be simplified
		if self.activation_functions[-1].__name__ == "soft_cross":
			calculate_loss = soft_cross_prime(self.output, self.targets)
		else:
			calculate_loss = self.error_function_derivative(self.output, self.targets, self.loss_average)
		error_signal = calculate_loss

		for layer in range(len(self.layers))[::-1]:
			error_signal = self.layers[layer].update(error_signal)