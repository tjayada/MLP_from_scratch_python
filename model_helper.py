import math

######################################
########### math functions ###########
######################################

def vec_vec_dot_product(vector_1, vector_2):
	"""dot product between two vectors"""
	return sum([v_1 * v_2 for v_1, v_2 in zip(vector_1, vector_2)])

def mat_vec_dot_product(matrix_1, vector_2):
	"""dot product between matrix and vector"""
	return [ vec_vec_dot_product(vector_1, vector_2) for vector_1 in matrix_1 ]

def vec_mat_dot_product(vector_1, matrix_2):
	"""dot product between vector and matrix"""
	return [ vec_vec_dot_product(vector_1, vector_2) for vector_2 in [ [vector_2[i] for vector_2 in matrix_2] for i in range(len(matrix_2[0])) ] ]

def mat_mat_dot_product(matrix_1, matrix_2):
	"""dot product between matrix and matrix"""
	return [ vec_mat_dot_product(vector_1, matrix_2) for vector_1 in matrix_1]

def vec_plus_vec(vector_1, vector_2):
	"""addition of a vector and vector"""
	return [ v_1 + v_2 for v_1, v_2 in zip(vector_1, vector_2) ]

def vec_minus_vec(vector_1, vector_2):
	"""subtraction of a vector and vector"""
	assert len(vector_1) == len(vector_2)
	return [ v_1 - v_2 for v_1, v_2 in zip(vector_1, vector_2)]

def mat_plus_vec(matrix_1, vector_2):
	"""addition of matrix and vector on each row"""
	return [ vec_plus_vec(vector_1, vector_2) for vector_1 in matrix_1]

def vec_scalar_divide(vector_1, scalar_2):
	"""devision of vector by scalar"""
	return [ v_1 / scalar_2 for v_1 in vector_1]

def vec_scalar_multiply(vector_1, scalar_2):
	"""multiplication of vector by scalar"""
	return [ v_1 * scalar_2 for v_1 in vector_1]

def vec_scalar_exponent(vector_1, scalar_2):
	"""exponentiation of vector by scalar"""
	return [ v_1 ** scalar_2 for v_1 in vector_1]

def vec_vec_multiplication(vector_1, vector_2):
	"""element-wise multiplication of vector and vector"""
	return [ v_1 * v_2 for v_1, v_2 in zip(vector_1, vector_2)]

def vec_vec_divide(vector_1, vector_2):
	"""element-wise division of vector and vector"""
	return [ v_1 / v_2 for v_1, v_2 in zip(vector_1, vector_2)]

def mat_vec_multiplication(matrix_1, vector_2):
	"""row-wise multiplication of matrix and vector"""
	return [ vec_vec_multiplication(vector_1, vector_2) for vector_1 in matrix_1]

def sum_up_matrix_by_rows(matrix_1):
	"""sum up matrix row-wise"""
	return [ sum(vector_1) for vector_1 in matrix_1]

def sum_up_matrix_by_cols(matrix_1):
	"""sum up matrix column-wise"""
	return [ sum([ column_1[i] for column_1 in matrix_1]) for i in range(len(matrix_1[0]))]

def mat_scalar_divide(matrix_1, scalar_2):
	"""multiplication of matrix by scalar"""
	return [ [v_1 / scalar_2 for v_1 in vector_1] for vector_1 in matrix_1 ]

def mat_scalar_multiply(matrix_1, scalar_2):
	"""division of vector by scalar"""
	return [ [v_1 * scalar_2 for v_1 in vector_1] for vector_1 in matrix_1 ]

def mat_mat_minus(matrix_1, matrix_2):
	"""subtraciton of matrix and matrix"""
	assert len(matrix_1[0]) == len(matrix_2[0])
	assert len(matrix_1)    == len(matrix_2)
	return [ [v_1 - v_2 for v_1, v_2 in zip(vector_1, vector_2)] for vector_1, vector_2 in zip(matrix_1, matrix_2) ]

def mat_mat_plus(matrix_1, matrix_2):
	"""addition of matrix and matrix"""
	assert len(matrix_1[0]) == len(matrix_2[0])
	assert len(matrix_1)    == len(matrix_2)
	return [ [v_1 + v_2 for v_1, v_2 in zip(vector_1, vector_2)] for vector_1, vector_2 in zip(matrix_1, matrix_2) ]

def mat_mat_multiply(matrix_1, matrix_2):
	"""element-wise multiplication of matrix and matrix"""
	assert len(matrix_1[0]) == len(matrix_2[0])
	assert len(matrix_1)    == len(matrix_2)
	return [ [v_1 * v_2 for v_1, v_2 in zip(vector_1, vector_2)] for vector_1, vector_2 in zip(matrix_1, matrix_2) ]

#########################################
########### helpful functions ###########
#########################################

def transpose(matrix_1):
	"""transpose a matrix"""
	return [  [column[i] for column in matrix_1] for i in range(len(matrix_1[0])) ]

def one_hot(t, n):
	"""one-hot encode input t with size n"""
	return [0 if t != i else 1 for i in range(n)]

def get_argmax(vector_1):
	"""get index of max value in vector"""
	maxi = max(vector_1)
	return max([ 0 if maxi != val else idx for idx,val in enumerate(vector_1)])

def flatten(matrix_1):
	"""flatten matrix by rows"""
	return [v_1 for vector_1 in matrix_1 for v_1 in vector_1]

def clip(value, between_down, between_up):
	"""clip a value between two boundaries"""
	if between_down < value < between_up:
		return value
	elif value <= between_down:
			return between_down
	else:
		return between_up
	
def scale_data(X):
	"""scale data to mean 0 and unit variance for each feature"""
	# calculate mean
	summed_up = [0] * len(X[0])
	for sample in X:
		summed_up = vec_plus_vec(summed_up, sample) 
	means = vec_scalar_divide(summed_up, len(X))

	# calculate std
	summed_up = [0] * len(X[0])
	for sample in X:
		diff = vec_minus_vec(sample, means)
		diff_squared = vec_vec_multiplication(diff, diff)
		summed_up = vec_plus_vec(summed_up, diff_squared)
	variance = vec_scalar_divide(summed_up, len(X))
	std = vec_scalar_exponent(variance, 1/2)

	# transform data
	scaled_X = X.copy()
	for idx,sample in enumerate(X):
		diff =  vec_minus_vec(sample, means)
		scaled_X[idx] = vec_vec_divide(diff, std)
		
	return scaled_X

############################################
########### activation functions ###########
############################################

def relu(activation):
	"""rectified linear unit activation"""
	if isinstance(activation, list):
		return [ relu(activation_i) for activation_i in activation ]
	else:
		return 0 if activation < 0 else activation

def relu_prime(activation):
	"""derivative of rectified linear unit activation"""
	if isinstance(activation, list):
		return [ relu_prime(activation_i) for activation_i in activation ]
	else:
		return 0 if activation < 0 else 1

def linear(activation):
	"""linear activation"""
	return activation

def linear_prime(activation):
	"""derivative of linear activation"""
	return activation

def sigmoid(activation):
	"""sigmoid activation"""
	if isinstance(activation, list):
		return [ sigmoid(activation_i) for activation_i in activation ]
	else:
		activation =  255 if activation > 255 else activation
		activation = -255 if activation < -255 else activation
		return  (1/ (1 + math.exp(- activation)))

def sigmoid_prime(activation):
	"""derivative of sigmoid activation"""
	if isinstance(activation, list):
		return [ sigmoid_prime(activation_i) for activation_i in activation ]
	else:
		return  (sigmoid(activation) * (1 - sigmoid(activation)))
	
def softmax(activation):
	"""softmax activation"""
	softmax_activation = []
	for vector_1 in activation:
		softmax_activation.append([ math.exp(clip(activation_i, -255, 255)) / sum([math.exp(clip(activation_all, -255, 255)) for activation_all in vector_1]) for activation_i in vector_1])
	return softmax_activation

def softmax_prime(activation):
	"""derivative of softmax activation (jacobian)"""
	derv = []
	for vector_1 in activation:
		derv_col = []
		for i in range(len(vector_1)):
			derv_row = []
			for j in range(len(vector_1)):
				derv_row.append( vector_1[i] * ((i == j ) -  vector_1[j] ) )
			derv_col.append(derv_row)
		derv.append(derv_col)
	return derv

#################################################################
########### activation and loss function combinations ###########
#################################################################

def soft_cross(activation):
	"""special case of softmax and cross entropy combination"""
	return softmax(activation)

def soft_cross_prime(predictions, targets, loss_average=False):
	"""derivative of special case of softmax and cross entropy combination"""
	return mat_scalar_divide(mat_mat_minus(predictions, targets),  len(predictions))

######################################
########### loss functions ###########
######################################

def mean_squared_error(prediction, target):
	"""mean squared error between predictions and targets"""
	assert len(prediction) == len(target) 
	assert len(prediction[0]) == len(target[0])

	squared_error = [ [ (pred-targ)**2 for pred,targ in zip(prediction_i, target_i)] for prediction_i,target_i in zip(prediction,target) ]
	summed_up_squared_error = sum_up_matrix_by_cols(squared_error)
	summed_up_squared_error = [val / len(prediction) for val in summed_up_squared_error]
	return sum(summed_up_squared_error) / len(summed_up_squared_error)

def mean_squared_error_prime(prediction, target, loss_average=False):
	"""derivative of mean squared error between predictions and targets"""
	assert len(prediction) == len(target) 
	assert len(prediction[0]) == len(target[0])

	error = [ [ 2* (pred - targ) for pred,targ in zip(prediction_i, target_i)] for prediction_i,target_i in zip(prediction,target) ]
	summed_up_error = [ [val_1 / len(vec_1) for val_1 in vec_1] for vec_1 in error]

	if loss_average:
		summed_up_error_columns = sum_up_matrix_by_cols(summed_up_error)
		summed_up_error_columns = [ val / len(summed_up_error) for val in summed_up_error_columns]
		summed_up_error_columns = [summed_up_error_columns] * len(summed_up_error)
	else:
		summed_up_error_columns = summed_up_error
	
	return summed_up_error_columns 

def cross_entropy(predictions, targets):
	"""cross entropy between predictions and targets"""
	cross = []
	for vector_1, vector_2 in zip(predictions, targets):
		cross.append([ -(math.log(clip(p, 0.0001, 0.9999)) * t) - (1-t) * math.log(1-clip(p, 0.0001, 0.9999)) for p,t in zip(vector_1, vector_2)])

	a = sum_up_matrix_by_cols(cross)
	a = [val / len(predictions) for val in a]
	return sum(a) / len(a)

def cross_entropy_prime(predictions, targets, loss_average=False):
	"""derivative of cross entropy between predictions and targets"""
	cross = []
	for vector_1, vector_2 in zip(predictions, targets):
		cross.append([(-t + clip(p, 0.0001, 0.9999)) / clip((p * (-p + 1)), 0.0001, 0.9999)  for p,t in zip(vector_1, vector_2)])
	if loss_average:
		summed_up = [sum_up_matrix_by_cols(cross)] * len(predictions)
		cross = []
		for vec_1 in summed_up:
			cross.append([float(b) / len(predictions) for b in vec_1])
	return cross

def hit_or_miss(predictions, targets, round_function = 6, error_margin = 0.01):
	"""calculate whether prediction and target are the same with a certain error margin"""
	acc = 0
	for p, t in zip(predictions, targets):
		acc += abs(round(p, round_function) - t) < error_margin
	return acc / len(predictions)

def r_2(predictions, targets):
	"""calculate coefficient of determination (R squared) between predictions and targets"""
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

############################################################
########### mapping between functions and names ############
############################################################

function_dictionary = {
	"relu" 			: relu,
	"relu_prime" 		: relu_prime,
	"linear" 			: linear,
	"linear_prime" 		: linear_prime,
	"sigmoid"			: sigmoid,
	"sigmoid_prime"	: sigmoid_prime,
	"softmax"		: softmax,
	"softmax_prime"	: softmax_prime,
	"soft_cross"	: soft_cross,
	"soft_cross_prime"	: soft_cross_prime,
	"mean_squared_error"		: mean_squared_error,
	"mean_squared_error_prime"	: mean_squared_error_prime,
	"cross_entropy"			: cross_entropy,
	"cross_entropy_prime"	: cross_entropy_prime
}