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
		self.weights 				= None
		self.bias 					= None
		self.output					= None
		self.activation				= None
		self.built					= False


	def __call__(self, inputs, targets):
		self.inputs 		= inputs.copy()
		self.targets 		= targets.copy()
		
		if not self.built:
			inputs_size 	= len(inputs) if not isinstance(inputs[0], list) else len(inputs[0])
			#self.weights 	= [ [random.uniform(-0.5, +0.5) for _ in range(inputs_size)] for _ in range(self.output_size) ]
			self.weights 	= [ [random.uniform(-0.5, +0.5) for _ in range(self.output_size)] for _ in range(inputs_size) ]
			self.bias 		= [ random.uniform(-0.5, +0.5) for _ in range(self.output_size) ]
			self.built 		= True

		self.activation 		= mat_mat_dot_product(self.inputs.copy(), self.weights.copy())
		self.activation 		= mat_plus_vec(self.activation.copy(), self.bias.copy())

		self.output 		= self.activation_function(self.activation.copy())

		return self.output
	
	def update(self, error_signal):
		#delta 	= mat_mat_dot_product(self.inputs, error_signal)
		#print("error signal : ", error_signal)
		#print("layer inputs : ", self.inputs)
		gradient = mat_mat_dot_product(transpose(self.inputs), error_signal)
		error_signal = mat_mat_dot_product(error_signal, transpose(self.weights) )
		#print("gradient" , gradient)
		
		weight_gradient = mat_scalar_divide(gradient, len(self.inputs))
		weight_gradient = mat_scalar_multiply(weight_gradient, 1)
		
		bias_gradient = vec_scalar_divide(sum_up_matrix_by_cols(gradient), len(self.inputs))
		bias_gradient = vec_scalar_multiply(bias_gradient, 1)

		#print("self.weights" , self.weights)
		#print("weight_gradient" , weight_gradient)
		self.weights = mat_mat_minus(self.weights, weight_gradient)
		#print("new self.weights" , self.weights)
		self.bias = vec_minus_vec(self.bias, bias_gradient)

		return error_signal
	


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
		#print(self.output , self.targets)
		calculate_loss = mean_squared_error_prime(self.output, self.targets)
		#print("calculate_loss" , calculate_loss)
		# loss : 
		#print("self.output: ", relu_function_derivative(self.output))
		error_signal = mat_vec_multiplication(sigmoid_prime(self.output) , calculate_loss)
		#error_signal = self.layers[-1].update(error_signal)
		#print( "error_signal" , error_signal)


		for layer in range(1,len(self.layers)):
			#print(layer)
			#print(self.layers[-(layer)])
			error_signal = self.layers[-layer].update(error_signal)
			#print("error_signal" , error_signal)
			error_signal = mat_vec_multiplication(sigmoid_prime(self.layers[-(layer + 1)].activation), error_signal[0])
			#print("relu_function_derivative(self.layers[-(layer)].activation)" , relu_function_derivative(self.layers[-(layer)].activation))
			#print("error_signal" , error_signal)
			#break
			#error_signal = self.layers[-(layer)].update(error_signal)
			#print(relu_function_derivative(layer.activation))
			#print("\n")
			#print(error_signal)
			#print("\n")
			#print("\n")
			#print(relu_function_derivative(self.layers[-(layer + 1)].activation))
			

inputs = [[0,0],
                   [1,0],
                   [0,1],
                   [1,1]]

# all the logical gates
and_labels = [[0],
                       [0],
                       [0],
                       [1]]

or_labels = [[0],
                      [1],
                      [1],
                      [1]]

not_and_labels = [[1],
                           [1],
                           [1],
                           [0]]

not_or_labels = [[1],
                          [0],
                          [0],
                          [0]]

xor_labels = [[0],
                       [1],
                       [1],
                       [0]]

model = DenseModel([8,1], [sigmoid, sigmoid])

#zuzu = [[1,1,-2,2], [2,2,-4,4] ]#, [-3, -1, 5, 5]]
#zizi = [ [2], [4]]#, [1]]

epoch = 800
for i in range(epoch):
	acc = 0
	for i, x in enumerate(inputs):
		model([x], [and_labels[i]])
		model.backpropagation()
		acc += (round(model.output[0][0]) == and_labels[i][0])
	print(acc / 4)