import math 
import random
random.seed(42)
from model_helper import *



class DenseLayer(object):
	"""docstring for DenseLayer"""
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
		#print("error_signal_before : " , error_signal)
		#print("\n \n")
		
		#print("self.inputs : " , self.inputs)
		#print("\n \n")
		
		#print("self.outputs : " , self.output)
		#print("\n \n")
		#print("error_signal before: " , error_signal)
		#print("acti deriv : ", function_dictionary[f"{self.activation_function.__name__}_prime"](self.inputs))
		if not f"{self.activation_function.__name__}_prime" == "linear_prime":
			#print("hell")
			#print("errort signalo : ", error_signal)
			#print("function_dictionary : ", function_dictionary[f"{self.activation_function.__name__}_prime"](self.inputs))
			error_signal = mat_mat_multiply(error_signal, function_dictionary[f"{self.activation_function.__name__}_prime"](self.activation))
		
		#print("error_signal after acti prime: " , error_signal)
		#print("self input : ", self.inputs)
		gradient = mat_mat_dot_product(transpose(self.inputs), error_signal)
		
		#print("gradient : " , gradient)
		#print("\n \n")

		#error_signal_list = []
		#for i in range(len(error_signal)):
		#	error_signal_list.append(mat_mat_dot_product(error_signal[i], transpose(self.weights) ))
		#print("error signal in update : ", error_signal)
		error_signal_out = mat_mat_dot_product(error_signal, transpose(self.weights) )
		
		#print("error_signal fter dot pro: " , error_signal)
		#print("\n \n")

		#weights = gradient[0]
		#for i in range(len(error_signal)):
		#gradient = mat_mat_plus(gradient[0], gradient[1])
		
		if self.batch_average:# and layer == len_layer:
			#print(len(self.targets))
			weight_gradient = mat_scalar_divide(gradient, len(self.targets))

			#print("error_signal : " , error_signal)
			
			bias_gradient = sum_up_matrix_by_rows(error_signal)
			#print("bias_gradient : " , bias_gradient)
			bias_gradient = [sum(bias_gradient) / len(bias_gradient)] * len(self.bias)
			bias_gradient = vec_scalar_divide(bias_gradient, len(self.targets))

		else:
			weight_gradient = gradient
			bias_gradient = sum_up_matrix_by_rows(error_signal)
			bias_gradient = [sum(bias_gradient) / len(bias_gradient)] * len(self.bias)
		
		#weight_gradient = mat_scalar_divide(gradient, len(self.targets))

		# 1. get all losses individually --> dot product with inputs 
		# --> inputs = 32 x 10 and losses = 32 x 1
		# --> inputs.T x losses = 10 x 1, which is the weight dimensionality of the output layer
		# --> what happens is that all gradients are summed up

		#weight_gradient = gradient
		weight_gradient = mat_scalar_multiply(weight_gradient, self.learning_rate)
		
		#bias_gradient = vec_scalar_divide(sum_up_matrix_by_cols(gradient), len(self.targets))
	
		bias_gradient = vec_scalar_multiply(bias_gradient, self.learning_rate)

		#print("self.weights" , self.weights)
		#print("weight_gradient" , weight_gradient)
		self.weights = mat_mat_minus(self.weights, weight_gradient)
		#print("new self.weights" , self.weights)
		#print("self.bias : " , self.bias)
		#print("bias_gradient :", bias_gradient)
		self.bias = vec_minus_vec(self.bias, bias_gradient)

		return error_signal_out
	


class DenseModel(object):
	"""docstring for DenseModel"""
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
		

		# make acceptalbe for all activation functions
		self.activation_functions_derivative = [ function_dictionary[f"{self.activation_functions[i].__name__}_prime"] for i in range(len(self.activation_functions)) ]

	def __call__(self, inputs, targets):
		self.inputs 		= inputs
		self.targets 		= targets
		
		if not self.built:

			for i in range(len(self.layer_config)):
				self.layers.append(DenseLayer(self.layer_config[i], self.activation_functions[i], learning_rate=self.learning_rate, batch_average=self.batch_average))
			self.built 		= True
		
		self.output = self.inputs
		for layer in self.layers:
			self.output = layer(self.output, self.targets)
		
		if self.classification:
			#print(self.output)
			soft_out = softmax(self.output)
			#print(self.output)
			#self.prediction = [ get_argmax(elem_x) for elem_x in self.output ]
			#print(self.layers[-1].activation)
			self.prediction = [ get_argmax(elem_x) for elem_x in soft_out ]
			#self.output = self.layers[-1].activation
			#print(self.prediction)
		
		return self.output

	def backpropagation(self):
		#print(self.output , self.targets)
		calculate_loss = self.error_function_derivative(self.output, self.targets, self.loss_average)
		#print("calculate_loss" , calculate_loss)

		#print("self.layers[0].weights : ", self.layers[0].weights)
		# loss : 
		#print("self.output: ", relu_function_derivative(self.output))
		#print(self.output)
		#print("\n")
		#print(softmax_prime(self.output))
		#error_signal = [ mat_vec_dot_product(out , calculate_loss) for out in softmax_prime(self.output)]
		#print(self.activation_functions_derivative[-1](self.output))
		#print(error_signal)
		#print("\n")
		#error_signal = mat_mat_multiply(self.activation_functions_derivative[-1](self.output) , error_signal)
		#error_signal = mat_vec_multiplication(self.activation_functions_derivative[-1](self.output) , calculate_loss)
		#error_signal = [ calculate_loss ] * len(self.output)
		error_signal = calculate_loss
		#print(error_signal)
		#print("error_signal : " , error_signal)
		#print("erste cross derivative acti : " , self.activation_functions_derivative[-1](self.output)[0])
		#print("zweite cross derivative acti : " , self.activation_functions_derivative[-1](self.output)[1])
		#print("alles : " , self.activation_functions_derivative[-1](self.output))
		#error_signal = []
		#for i in range(len(self.activation_functions_derivative[-1](self.output))):
		#	error_signal.append(mat_scalar_multiply(self.activation_functions_derivative[-1](self.output)[i] , calculate_loss))
		
		#print("error_signal" , error_signal)



		for layer in range(len(self.layers))[::-1]:
			#print(layer)
			#print(self.layers[layer].activation_function)
			error_signal = self.layers[layer].update(error_signal)
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
			#error_signal = mat_vec_multiplication(self.activation_functions_derivative[-(layer + 1)](self.layers[-(layer + 1)].activation), error_signal[0])
			
			
			
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
			


#m = DenseModel([1,2], [relu, relu], classification=True, error_function=cross_entropy)
#m([[2,3,4]] , [[2,1]])
#m.backpropagation()