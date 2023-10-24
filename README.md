# MLP (truly) from scratch

## Preliminary
This project was done solely for educational purposes (with a hint of smugness), since all similar projects I have seen claim to be done from scratch, but still rely on numpy. Even just using numpy for the arrays and implementing the mathematical functions myself is a privilege I did not want, since the arrays themselves have wonderful properties such as addition 
		
		np.array([1, 2, 3]) + np.array([1, 2, 3])  
		= array([2, 4, 6]) 

while the same operation using python lists would result in 

	[1, 2, 3, 1, 2, 3]

Thus, in the pursuit of rigorous completeness, any attempt at optimisation was abandoned in favour of trying to control every aspect of the deep learning pipeline without the help of external libraries, which sometimes led to questionable implementation choices altogether. One such questionable choice was the acquisition of the data used to train the models, which was initially done using sklearn, but was eventually pickled and stored so as not to require any libraries afterwards. As you can hopefully see at this point, this project was mostly done for fun, but I also learned a lot along the way. Nevertheless, this project was really done from scratch using only pure Python.

## Usage
Simply running

	python3 main.py
				
will start the California Housing regression task with pre-set hyperparameters (yes, no requirements.txt).

In theory, the DenseModel class can be used for any deep learning task, assuming the desired activation and loss functions are already implemented. In practice, "bigger" tasks will require an enormous amount of training time, as I mentioned before, this project was done for educational purposes, not real life application, so optimisation was not part of the process. For reference, running the mnist example for one epoch took ~30 minutes, but also yielded the following, quite pleasing results 
	
	Training Accuracy:  0.9053 ,    Test Accuracy:  0.9496 ,    Loss:  0.0529
	Validation Accuracy:  0.9513
	
A custom model could be initialized like
			
	DenseModel(layer_config=[size_layer_1, size_layer_2, size_layer_3],
				activation_functions=[af_layer_1, af_layer_2, af_layer_3], 
				classification=Boolean, 
				error_function=error_function, 
				learning_rate=learning_rate, 
				batch_average=Boolean, 
				loss_average=Boolean)
				
and then trained using the training function. 

## Some benchmarks
The pre-implemented examples resulted in the following

### Logic gates (XOR gate)
	Epoch  100 
	Training Accuracy:  1.0000 ,    Test Accuracy:  nan ,    Loss:  0.3722
### California housing dataset
	
	Epoch   20     
	Training Accuracy:  0.7269 ,    Test Accuracy:  0.7251 ,    Loss:  0.3732
	Validation Accuracy:  0.7513
	
### Mnist dataset
	Epoch    1
	Training Accuracy:  0.9053 ,    Test Accuracy:  0.9496 ,    Loss:  0.0529
	Validation Accuracy:  0.9513
	
## Further improvements
1. The biggest problem I faced was incorporating the derivative of the softmax activation function, as it becomes a Jacobian matrix, which inevitably introduces exploding gradients due to the increased size (from vector to matrix), as well as producing values many times larger than the original input. I was not able to control the gradient by clipping or normalisation, which may have been an implementation error on my part, but this feature is still missing. Of course, the shortened version in combination with the cross-entropy loss is implemented and used for example in the mnist task.
2. 