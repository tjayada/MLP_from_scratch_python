import math
import random
from dense_helper import *


from sklearn.datasets import fetch_openml


def main():
	X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
	X_batch = X[:2]
	y_batch = y[:2]
	
	for i,j in zip(X_batch, y_batch):
		print(i)
		print(j)

 		




if __name__ == '__main__':
	main()