import math 
import random
random.seed(42)
from dense_helper import *



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
		self.actvation				= None
		self.built					= False


	def __call__(self, inputs, targets):
		self.inputs 		= inputs
		self.targets 		= targets
		
		if not self.built:
			inputs_size 	= len(inputs) if not isinstance(inputs[0], list) else len(inputs[0])
			self.weights 	= [ [random.uniform(-0.5, +0.5) for _ in range(inputs_size)] for _ in range(self.output_size) ]
			self.bias 		= [ random.uniform(-0.5, +0.5) for _ in range(self.output_size) ]
			self.built 		= True

		self.actvation 		= mat_mat_dot_product(self.inputs, self.weights)
		self.actvation 		= mat_plus_vec(self.actvation, self.bias)

		self.output 		= self.activation_function(self.actvation)

		return self.output
	

f = DenseLayer(5, relu_function)
zuzu = [[1,-10,20], [1,2,2]]
zizi = 3

print(f(zuzu, zizi))

