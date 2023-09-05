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
		gradient = mat_mat_dot_product(transpose(error_signal), f_2.inputs)
		self.weights = mat_mat_minus(self.weights, gradient)
		#self.bias = 

	

f_1 = DenseLayer(5, relu_function)
zuzu = [[1,-10,20,2], [2,2,2,3], [-3, -1, 5, 5]]
zizi = [ [1, 1], [1,1], [1,1]]

#zuzu = [[1,-10,20]]
#zizi = [ [1, 1, 1]]

out_1 = f_1(zuzu, zizi)
#print("\n")
#print(f_1.weights)
#print("\n")
#print(out_1)
#print("\n")


f_2 = DenseLayer(2, relu_function)
out_2 = f_2(out_1, zizi)
#print(out_2)
#print("\n")

error = mean_squared_error(out_2, zizi)
#print("errororo: ", error)
#print("\n")

#print(relu_function_derivative(f_2.activation))
#print("activ: ", f_2.activation)
#print(f_2.weights)
#print("\n")
#print(f_2.output)
#print(relu_function_derivative(f_2.activation))

# gradient x activations * activation_derivative
# need to get to --> 2 x 5 
# which is the size of the weights 
# altough we need the 2 x 5 excatly batch_size number of times 
# which could be seen as two 1 x 5, which could be batch_size x 5 
# 

# [ 1 x 2 ] x [ 2 x 5 ] x [ 1 x 2 ]

print(f_2.bias)
#print(relu_function_derivative(f_2.activation))
dd = mat_vec_multiplication(relu_function_derivative(f_2.activation) , error)
print(dd)
print(sum_up_matrix_by_cols(dd))
#print(transpose(dd))

#print(mat_mat_dot_product(dd, f_2.inputs))
#print(mat_mat_dot_product(transpose(dd), f_2.inputs))

#print( mat_mat_dot_product([ [1,1,1] , [2,2,2] ]  ,  [ [2], [4], [5]]))

error_signal = [ a*b for a,b in zip(error, relu_function_derivative(f_2.activation)[0])]

#print(error_signal)
#print(f_1.output)
#print("\n")

#test_transpose = [0.030179407659672877, 0.5467014038049781, 0.03869175148617696]
#test_transpose = [ [a] for a in test_transpose ]

#print([[ a*test_transpose[i] for a in f_1.output[0]] for i in range(len(test_transpose))])

# THINK MORE ABOUT BACKPROPAGATION
# WE UPDATE THE WEIGHTS ACCORDING TO THEIR CONNECTIONS
# WHAT SHAPE DOES OUR DELTA NEED TO BE ??
# INTUITIVE IT SHOULD BE THE SAME AS THE WEIGHTS WE ARE UPDATING, SINCE
# NEW_WEIGHTS = OLD_WEIGHTS - UPDATE

#error_signal =