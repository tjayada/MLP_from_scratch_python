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
		#gradient = []
		#for i in range(len(error_signal)):
		#	gradient.append(mat_mat_dot_product(transpose(self.inputs), error_signal[i]))
		gradient = mat_mat_dot_product(transpose(self.inputs), error_signal)
		
		#error_signal_list = []
		#for i in range(len(error_signal)):
		#	error_signal_list.append(mat_mat_dot_product(error_signal[i], transpose(self.weights) ))
		#print("error signal in update : ", error_signal)
		error_signal = mat_mat_dot_product(error_signal, transpose(self.weights) )
		
		#weights = gradient[0]
		#for i in range(len(error_signal)):
		#gradient = mat_mat_plus(gradient[0], gradient[1])
		
		weight_gradient = mat_scalar_divide(gradient, len(self.inputs))
		weight_gradient = mat_scalar_multiply(weight_gradient, 0.01)
		
		bias_gradient = vec_scalar_divide(sum_up_matrix_by_cols(gradient), len(self.inputs))
		bias_gradient = vec_scalar_multiply(bias_gradient, 0.01)

		#print("self.weights" , self.weights)
		#print("weight_gradient" , weight_gradient)
		self.weights = mat_mat_minus(self.weights, weight_gradient)
		#print("new self.weights" , self.weights)
		self.bias = vec_minus_vec(self.bias, bias_gradient)

		return error_signal
	


class DenseModel(object):
	"""docstring for DenseModel"""
	def __init__(self, layer_config, activation_functions, classification = False, error_function = mean_squared_error):
		super(DenseModel, self).__init__()
		self.layer_config 			= layer_config
		self.activation_functions	= activation_functions
		self.classification 		= classification
		self.error_function			= error_function
		self.error_function_derivative = function_dictionary[f"{error_function.__name__}_prime"]
		self.inputs 				= None
		self.targets 				= None
		self.layers 				= []
		self.output					= None
		self.prediction				= None
		self.activation				= None
		self.built					= False
		

		# make acceptalbe for all activation functions
		self.activation_functions_derivative = [ function_dictionary[f"{self.activation_functions[i].__name__}_prime"] for i in range(len(self.activation_functions)) ]

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
		
		if self.classification:
			self.prediction = [ get_argmax(elem_x) for elem_x in self.output ]
		
		return self.output

	def backpropagation(self):
		#print(self.output , self.targets)
		calculate_loss = self.error_function_derivative(self.output, self.targets)
		#print("calculate_loss" , calculate_loss)

		#print("self.layers[0].weights : ", self.layers[0].weights)
		# loss : 
		#print("self.output: ", relu_function_derivative(self.output))
		error_signal = mat_vec_multiplication(self.activation_functions_derivative[-1](self.output) , calculate_loss)
		
		#print("erste cross derivative acti : " , self.activation_functions_derivative[-1](self.output)[0])
		#print("zweite cross derivative acti : " , self.activation_functions_derivative[-1](self.output)[1])
		#print("alles : " , self.activation_functions_derivative[-1](self.output))
		#error_signal = []
		#for i in range(len(self.activation_functions_derivative[-1](self.output))):
		#	error_signal.append(mat_scalar_multiply(self.activation_functions_derivative[-1](self.output)[i] , calculate_loss))
		
		#print("error_signal" , error_signal)



		for layer in range(1,len(self.layers)):
			#print(layer)
			#print(self.layers[-(layer)])
			error_signal = self.layers[-layer].update(error_signal)
			#print("error_signal" , error_signal)
			#error_signal_list = []
			#for i in range(len(error_signal)):
				#print("acti fucntion len : ", len(self.activation_functions_derivative[-(layer + 1)](self.layers[-(layer + 1)].activation)))
			#	print(self.activation_functions_derivative[-(layer + 1)](self.layers[-(layer + 1)].activation))
				
				#print("acti fucntion 0 len : ", len(self.activation_functions_derivative[-(layer + 1)](self.layers[-(layer + 1)].activation)[0]))
				#print("error signal len : ", len(error_signal[0]))
				#print("error signal 0 len : ", len(error_signal[0][0]))
			#	print(self.layers[-(layer + 1)].weights)

			#	error_signal_list.append(mat_mat_multiply(self.activation_functions_derivative[-(layer + 1)](self.layers[-(layer + 1)].activation), error_signal[i]))
			#error_signal = error_signal_list.copy()
			#print("error_signal : ", error_signal)
			error_signal = mat_vec_multiplication(self.activation_functions_derivative[-(layer + 1)](self.layers[-(layer + 1)].activation), error_signal[0])
			
			
			
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
			
