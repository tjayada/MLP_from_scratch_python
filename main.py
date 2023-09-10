import math
import random
from model import *
from model_helper import *


from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_california_housing


def main():
	#X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
	X,y = fetch_california_housing(return_X_y=True)
	
	X = X
	y = y

	# just some provisional hyper parameters
	epochs = 50
	batch_size = 100

	model = DenseModel([12, 24, 12, 1], [relu_function, relu_function, relu_function, relu_function])

	#bat_sub = 0
	#bat_top = batch_size
	for epoch in range(epochs):
		accuracy = 0
		for i in range(len(X)):
			#X_batch = X#[bat_sub:bat_top]
			#y_batch = y#[bat_sub:bat_top]

			X_batch = [[ a for a in X[i]]]
			y_batch = [[y[i]]]

			model(X_batch, y_batch)
			#çprint(model.layers[0].weights)
			model.backpropagation()

			#print(model.output)
			#print(y_batch)

			accuracy +=  abs(model.output[0][0] - y_batch[0][0]) < 2
		print("Epoch : ", epoch , "Acc : ", accuracy / len(X))

	return model
 		




if __name__ == '__main__':
	main()
	

	#print(X_batch)
	#print( m(X_batch, y_batch) )
	#print(y_batch)