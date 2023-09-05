import math 
import random
random.seed(42)
from model_helper import *



class DenseLayer(object):
	"""docstring for DenseLayer"""
	def __init__(self, size, activation_function):
		super(DenseLayer, self).__init__()
		self.output_size 			= size
		self.activation_function	= activation_function
		self.inputs 				= None
		self.targets 				= None
		self.weigths 				= None
		self.bias 					= None
		self.output					= None
		self.activation				= None
		self.built					= False


	def __call__(self, inputs, targets):
		self.inputs 		= inputs
		self.targets 		= targets
		
		if not self.built:
			inputs_size 	= len(inputs) if not isinstance(inputs[0], list) else len(inputs[0])
			#self.weights 	= [ [random.uniform(-0.5, +0.5) for _ in range(inputs_size)] for _ in range(self.output_size) ]
			self.weights 	= [ [random.uniform(-0.5, +0.5) for _ in range(self.output_size)] for _ in range(inputs_size) ]
			self.bias 		= [ random.uniform(-0.5, +0.5) for _ in range(self.output_size) ]
			self.built 		= True

		self.activation 		= mat_mat_dot_product(self.inputs, self.weights)
		self.activation 		= mat_plus_vec(self.activation, self.bias)

		self.output 		= self.activation_function(self.activation)

		return self.output
	
	def update(self, error_signal):
		#delta 	= mat_mat_dot_product(self.inputs, error_signal)
		gradient = mat_mat_dot_product(transpose(error_signal), self.inputs)
		
		weight_gradient = mat_scalar_divide(gradient, len(self.inputs))
		bias_gradient = vec_scalar_divide(sum_up_matrix_by_cols(gradient), len(self.inputs))
		
		self.weights = mat_mat_minus(self.weights, weight_gradient)
		self.bias = vec_minus_vec(self.bias, bias_gradient)

		return gradient
	


class DenseModel(object):
	"""docstring for DenseModel"""
	def __init__(self, layer_config, activation_functions):
		super(DenseModel, self).__init__()
		self.layer_config 			= layer_config
		self.activation_functions	= activation_functions
		self.inputs 				= None
		self.targets 				= None
		self.layers 				= []
		self.output					= None
		self.activation				= None
		self.built					= False
	
	def __call__(self, inputs, targets):
		self.inputs 		= inputs
		self.targets 		= targets
		
		if not self.built:

			for i in range(len(self.layer_config)):
				self.layers.append(DenseLayer(self.layer_config[i], self.activation_functions[i]))
			self.built 		= True
		
		self.output = self.inputs
		for layer in self.layers:
			self.output = layer(self.output, self.targets)
		
		return self.output

	def backpropagation(self):
		calculate_loss = mean_squared_error(self.output, self.targets)
		error_signal = mat_vec_multiplication(relu_function_derivative(self.layers[-1].activation) , calculate_loss)

		for layer in self.layers[::-1]:
			error_signal = layer.update(error_signal)


model = DenseModel([4,3,2], [relu_function, relu_function, relu_function])

zuzu = [[1,-10,20,2], [2,2,2,3], [-3, -1, 5, 5]]
zizi = [ [1, 1], [1,1], [1,1]]


print(model(zuzu, zizi))
print(model.layers[0].weights)
model.backpropagation()
print(model.layers[0].weights)

