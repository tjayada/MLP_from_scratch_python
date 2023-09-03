import math
import random



########### math functions ###########

def vec_vec_dot_product(vector_1, vector_2):
	return sum([v_1 * v_2 for v_1, v_2 in zip(vector_1, vector_2)])

def mat_vec_dot_product(matrix_1, vector_2):
	return [ vec_vec_dot_product(vector_1, vector_2) for vector_1 in matrix_1 ]

def vec_mat_dot_product(vector_1, matrix_2):
	return [ vec_vec_dot_product(vector_1, vector_2) for vector_2 in [ [vector_2[i] for vector_2 in matrix_2] for i in range(len(matrix_2[0])) ] ]

def mat_mat_dot_product(matrix_1, matrix_2):
	return [ mat_vec_dot_product(matrix_2, vector_1) for vector_1 in matrix_1]

def vec_plus_vec(vector_1, vector_2):
	return [ v_1 + v_2 for v_1, v_2 in zip(vector_1, vector_2) ]

def mat_plus_vec(matrix_1, vector_2):
	return [ vec_plus_vec(vector_1, vector_2) for vector_1 in matrix_1]




########### activation functions ###########

def relu_function(activation):
	if isinstance(activation, list):
		return [ relu_function(activation_i) for activation_i in activation ]
	else:
		return 0 if activation < 0 else 1


#print(relu_function(3))
#print(relu_function(-3))

#print(relu_function([-2,-1,0,1,2]))
#print(relu_function([ [-2,0,2], [-1,0,1]]))


