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
	return [ vec_mat_dot_product(vector_1, matrix_2) for vector_1 in matrix_1]

#print( mat_mat_dot_product([ [1,1,1] , [2,2,2] ]  ,  [ [2], [4], [5]]))



def vec_plus_vec(vector_1, vector_2):
	return [ v_1 + v_2 for v_1, v_2 in zip(vector_1, vector_2) ]

def vec_minus_vec(vector_1, vector_2):
	return [ v_1 - v_2 for v_1, v_2 in zip(vector_1, vector_2)]

def mat_plus_vec(matrix_1, vector_2):
	return [ vec_plus_vec(vector_1, vector_2) for vector_1 in matrix_1]

def vec_scalar_divide(vector_1, scalar_2):
	return [ v_1 / scalar_2 for v_1 in vector_1]

def vec_scalar_multiply(vector_1, scalar_2):
	return [ v_1 * scalar_2 for v_1 in vector_1]

def vec_vec_multiplication(vector_1, vector_2):
	return [ v_1 * v_2 for v_1, v_2 in zip(vector_1, vector_2)]

def mat_vec_multiplication(matrix_1, vector_2):
	return [ vec_vec_multiplication(vector_1, vector_2) for vector_1 in matrix_1]

def sum_up_matrix_by_rows(matrix_1):
	return [ sum(vector_1) for vector_1 in matrix_1]

def sum_up_matrix_by_cols(matrix_1):
	return [ sum([ column_1[i] for column_1 in matrix_1]) for i in range(len(matrix_1[0]))]

def mat_scalar_divide(matrix_1, scalar_2):
	return [ [v_1 / scalar_2 for v_1 in vector_1] for vector_1 in matrix_1 ]

def mat_scalar_multiply(matrix_1, scalar_2):
	return [ [v_1 * scalar_2 for v_1 in vector_1] for vector_1 in matrix_1 ]

def mat_mat_minus(matrix_1, matrix_2):
	assert len(matrix_1[0]) == len(matrix_2[0])
	assert len(matrix_1)    == len(matrix_2)
	return [ [v_1 - v_2 for v_1, v_2 in zip(vector_1, vector_2)] for vector_1, vector_2 in zip(matrix_1, matrix_2) ]



def transpose(matrix_1):
	return [  [column[i] for column in matrix_1] for i in range(len(matrix_1[0])) ]





########### activation functions ###########

def relu_function(activation):
	if isinstance(activation, list):
		return [ relu_function(activation_i) for activation_i in activation ]
	else:
		return 0 if activation < 0 else activation


def relu_function_derivative(activation):
	if isinstance(activation, list):
		return [ relu_function_derivative(activation_i) for activation_i in activation ]
	else:
		return 0 if activation < 0 else 1



def sigmoid(activation):
	if isinstance(activation, list):
		return [ sigmoid(activation_i) for activation_i in activation ]
	else:
		return  (1/ (1 + math.exp(- activation)))


def sigmoid_prime(activation):
	if isinstance(activation, list):
		return [ sigmoid_prime(activation_i) for activation_i in activation ]
	else:
		return  (sigmoid(activation) * (1 - sigmoid(activation)))
	
	

#print(relu_function_derivative([[-1,1,1], [0,1,2]]))

#print(relu_function(3))
#print(relu_function(-3))

#print(relu_function([-2,-1,0,1,2]))
#print(relu_function([ [-2,0,2], [-1,0,1]]))



########### loss functions ###########



def mean_squared_error(prediction, target):
	assert len(prediction) == len(target) 
	assert len(prediction[0]) == len(target[0])

	squared_error = [ [ (pred-targ)**2 for pred,targ in zip(prediction_i, target_i)] for prediction_i,target_i in zip(prediction,target) ]
	summed_up_squared_error = sum_up_matrix_by_cols(squared_error)
	return summed_up_squared_error #[ elem / len(squared_error) for elem in summed_up_squared_error]


def mean_squared_error_prime(prediction, target):
	assert len(prediction) == len(target) 
	assert len(prediction[0]) == len(target[0])

	squared_error = [ [ -(targ - pred) for pred,targ in zip(prediction_i, target_i)] for prediction_i,target_i in zip(prediction,target) ]
	summed_up_squared_error = sum_up_matrix_by_cols(squared_error)
	return summed_up_squared_error 


#print(mean_squared_error( [[1,1], [3,3], [3,3]], [2,2] ))

ert = [ [1,2,3], [2,2,2], [1,2,3] ]
tre = [[2,3,4], [1,1,1], [3,3,3]]

ert = [ [1,1,1], [1,1,1], [1,1,1] ]
ert_2 = [ [2,2,2], [2,2,3], [2,2,2] ]

hm = [[0.9582761279044779], [0]]
yo = [ [1] , [2] ]
a = [1]
b = [[-0.4089634351954671]]

#print(mat_vec_dot_product(b,a))

#print(mean_squared_error( ert, ert_2 ))

#for i,j in zip(ert,tre):
#	print(i,j)

#print(mean_squared_error(ert, tre))

