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

def mat_mat_plus(matrix_1, matrix_2):
	assert len(matrix_1[0]) == len(matrix_2[0])
	assert len(matrix_1)    == len(matrix_2)
	return [ [v_1 + v_2 for v_1, v_2 in zip(vector_1, vector_2)] for vector_1, vector_2 in zip(matrix_1, matrix_2) ]



def mat_mat_multiply(matrix_1, matrix_2):
	assert len(matrix_1[0]) == len(matrix_2[0])
	assert len(matrix_1)    == len(matrix_2)
	return [ [v_1 * v_2 for v_1, v_2 in zip(vector_1, vector_2)] for vector_1, vector_2 in zip(matrix_1, matrix_2) ]



def transpose(matrix_1):
	return [  [column[i] for column in matrix_1] for i in range(len(matrix_1[0])) ]


def one_hot(t, n):
    return [0 if t != i else 1 for i in range(n)]


def get_argmax(vector_1):
    maxi = max(vector_1)
    return max([ 0 if maxi != val else idx for idx,val in enumerate(vector_1)])


def flatten(matrix_1):
    return [v_1 for vector_1 in matrix_1 for v_1 in vector_1]


def clip(value, between_down, between_up):
	if between_down < value < between_up:
		return value
	elif value < between_down:
			return between_down
	else:
		return between_up
	


########### activation functions ###########

def relu(activation):
	if isinstance(activation, list):
		return [ relu(activation_i) for activation_i in activation ]
	else:
		return 0 if activation < 0 else activation


def relu_prime(activation):
	if isinstance(activation, list):
		return [ relu_prime(activation_i) for activation_i in activation ]
	else:
		return 0 if activation < 0 else 1



def sigmoid(activation):
	if isinstance(activation, list):
		return [ sigmoid(activation_i) for activation_i in activation ]
	else:
		activation =  255 if activation > 255 else activation
		activation = -255 if activation < -255 else activation
		return  (1/ (1 + math.exp(- activation)))


def sigmoid_prime(activation):
	if isinstance(activation, list):
		return [ sigmoid_prime(activation_i) for activation_i in activation ]
	else:
		return  (sigmoid(activation) * (1 - sigmoid(activation)))
	
	

#def softmax(activation):
#	print(activation)
	#activation = flatten(activation)
	#return [ math.exp(clip(activation_i, -255, 255)) / sum([math.exp(activation_all) for activation_all in activation]) for activation_i in activation]
	
def softmax(activation):
    softmax_activation = []
    for vector_1 in activation:
        softmax_activation.append([ math.exp(clip(activation_i, -255, 255)) / sum([math.exp(activation_all) for activation_all in vector_1]) for activation_i in vector_1])
    
    return softmax_activation


#def softmax_prime(activation): 
#    derv_col = []
#    for i in range(len(activation)):
#        derv_row = []
#        for j in range(len(activation)):
#            derv_row.append( activation[i] * ((i == j ) -  activation[j] ))
#        
#        derv_col.append(derv_row)
#    return derv_col


def softmax_prime(activation): 
	derv = []
	for vector_1 in activation:
		derv_col = []
		for i in range(len(vector_1)):
			derv_row = []
			for j in range(len(vector_1)):
				derv_row.append( vector_1[i] * ((i == j ) -  vector_1[j] ))
			
			derv_col.append(derv_row)
		derv.append(derv_col)
	return derv

########### loss functions ###########



def mean_squared_error(prediction, target):
	#print(prediction, target)
	assert len(prediction) == len(target) 
	assert len(prediction[0]) == len(target[0])

	squared_error = [ [ (pred-targ)**2 for pred,targ in zip(prediction_i, target_i)] for prediction_i,target_i in zip(prediction,target) ]
	summed_up_squared_error = sum_up_matrix_by_cols(squared_error)
	#print(summed_up_squared_error)
	return summed_up_squared_error #[ elem / len(squared_error) for elem in summed_up_squared_error]


#print(mean_squared_error([[0.8488765622021412]],[[1.659]]))

def mean_squared_error_prime(prediction, target):
	assert len(prediction) == len(target) 
	assert len(prediction[0]) == len(target[0])

	squared_error = [ [ -(targ - pred) for pred,targ in zip(prediction_i, target_i)] for prediction_i,target_i in zip(prediction,target) ]
	#print(squared_error)
	summed_up_squared_error = sum_up_matrix_by_cols(squared_error)
	return summed_up_squared_error 

#print(mean_squared_error_prime([[0.061908245406225415], [0.038427675401711436]] , [[1.659], [3.606]]))

#def cross_entropy(predictions, targets):
#	if isinstance(predictions, list):
#		sum([ -(math.log(p) * t) if p != 0 else -(math.log(1e-100) * t) for p,t in zip(predictions, targets)])
#
#    return sum([ -(math.log(p) * t) if p != 0 else -(math.log(1e-100) * t) for p,t in zip(predictions, targets)])

def cross_entropy(predictions, targets):
    cross = []
    for vector_1, vector_2 in zip(predictions, targets):
        cross.append([ -(math.log(p) * t) if p != 0 else -(math.log(1e-100) * t) for p,t in zip(vector_1, vector_2)])
    
    a = sum_up_matrix_by_cols(cross)
    return a


#def cross_entropy_prime(predictions, targets):
#    return -sum([ t / p if p != 0 else ( t / math.log(1e-100) ) for p,t in zip(predictions, targets)])


def cross_entropy_prime(predictions, targets):
	cross = []
	for vector_1, vector_2 in zip(predictions, targets):
		cross.append([-1 * (t / p) if p != 0 else ( t / math.log(1e-100) ) for p,t in zip(vector_1, vector_2)])
	a = sum_up_matrix_by_cols(cross)
	return a

#print(cross_entropy([[1,0,0] , [0,1,0]], [[1,0,0] , [0,1,0]]))

def hit_or_miss(predictions, targets, round_function = 6, error_margin = 0.01):

	acc = 0
	for p, t in zip(predictions, targets):
		#print(p,t)
		#print(abs(round(p, round_function) - t) < error_margin)
		acc += abs(round(p, round_function) - t) < error_margin
	return acc / len(predictions)


"""
def r_2(predictions, targets):
	acc = 0
	residual_sum_of_squares = []
	total_sum_of_squares = []

	targets_mean = sum(targets) / len(targets)
	print(targets_mean)

	for p, t in zip(predictions, targets):
		# ((y_true - y_pred)** 2).sum() and  is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()
		residual_sum_of_squares.append((t-p)**2)
		total_sum_of_squares.append((t-targets_mean)**2)

	return (1 - (sum(residual_sum_of_squares) / sum(total_sum_of_squares)))
"""

def r_2(predictions, targets):
    acc = 0
    residual_sum_of_squares = []
    total_sum_of_squares = []
    
    if isinstance(predictions[0], list):
        targets_mean = [ t / len(targets) for t in sum_up_matrix_by_cols(targets)]    
        for p, t in zip(predictions, targets):
            a = [ (t_i-p_i)**2 for t_i, p_i in zip(t,p)]
            residual_sum_of_squares.append(a)
            b = [ (t_i-m)**2 for t_i,m in zip(t,targets_mean) ]
            total_sum_of_squares.append(b)

        residual_sum_of_squares = [ sum(v_1) for v_1 in transpose(residual_sum_of_squares)]
        total_sum_of_squares    = [ sum(v_1) for v_1 in transpose(total_sum_of_squares)]

        result = [ (1 - rsd / tsq) if tsq != 0 else 0 for rsd, tsq in zip(residual_sum_of_squares, total_sum_of_squares) ]
        return sum(result) / len(result)
    
    else:
        targets_mean = sum(targets) / len(targets)

        for p, t in zip(predictions, targets):
            residual_sum_of_squares.append((t-p)**2)
            total_sum_of_squares.append((t-targets_mean)**2)

        return (1 - (sum(residual_sum_of_squares) / sum(total_sum_of_squares)))



function_dictionary = {
	"relu" 			: relu,
	"relu_prime" 		: relu_prime,
	"sigmoid"			: sigmoid,
	"sigmoid_prime"	: sigmoid_prime,
	"softmax"		: softmax,
	"softmax_prime"	: softmax_prime,
	"mean_squared_error"		: mean_squared_error,
	"mean_squared_error_prime"	: mean_squared_error_prime,
	"cross_entropy"			: cross_entropy,
	"cross_entropy_prime"	: cross_entropy_prime
}


