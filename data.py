import math
import random
import pickle
from model_helper import scale_data


def load_logic_gates(logic = "and"):
    """
    returns tuple of inputs to gates and the response of the chosen logic gate
    
    Keyword arguments:
    logic -- the wanted logical operator (default "and") 
    """
    # inputs in default batchsize one, e.g [1,0] for "and" --> 0
    inputs          =   [[[0,0]],
                        [[1,0]],
                        [[0,1]],
                        [[1,1]]]

    # all the logical gates
    and_labels      =   [[[0]],
                        [[0]],
                        [[0]],
                        [[1]]]

    or_labels       =   [[[0]],
                        [[1]],
                        [[1]],
                        [[1]]]

    not_and_labels  =   [[[1]],
                        [[1]],
                        [[1]],
                        [[0]]]

    not_or_labels   =   [[[1]],
                        [[0]],
                        [[0]],
                        [[0]]]

    xor_labels      =   [[[0]],
                        [[1]],
                        [[1]],
                        [[0]]]
    
    # mapping of given argument to data
    logic_dictionary = {
        "and"       : and_labels,
        "or"        : or_labels,
        "not_and"   : not_and_labels,
        "not_or"    : not_or_labels,
        "xor"       : xor_labels
    }

    return inputs, logic_dictionary[logic]


def new_load_housing_data(train_test_split = 0.6, validation = False, batch_size=32):
    """
    Returns tuple of inputs and targets.
    In total its 20,640 samples and 9 features.
    
    Keyword arguments:
    train_test_split -- splitting ratio between training and test data (default 0.6)
    validation -- if value provided its the splitting ratio between test and validation data (default False)
    batch_size -- the size of each batch (default 32)
    """
    # this part is commented out since the data is already loaded and saved
    """
    from sklearn.datasets import fetch_california_housing
    X, y = fetch_california_housing(return_X_y=True)

    with open("pickled_data/housing_data_X.pkl", "wb") as file:
        pickle.dump(X.tolist(), file)
    
    with open("pickled_data/housing_data_y.pkl", "wb") as file:
        pickle.dump(y.tolist(), file)
    """
    # read pickled housing data 
    with open("pickled_data/housing_data_X.pkl", "rb") as file:
        X = pickle.load(file)
    
    with open("pickled_data/housing_data_y.pkl", "rb") as file:
        y = pickle.load(file)
    
    # scale data to mean 0 and unit variance for each feature
    X = scale_data(X)
    
    # shuffle the data
    p = [(i,j) for i,j in zip(X, y)]
    random.shuffle(p)
    X = [i[0] for i in p]
    y = [i[1] for i in p]

    count = 0
    new_x = []
    new_y = []
    batch_x = []
    batch_y = []

    for np_x, np_y in zip(X,y):
        # create batches of data
        if count == batch_size:
            batch_x.append(new_x)
            batch_y.append(new_y)
            
            new_x = []
            new_y = []
            count = 0
        # convert numpy float to "regular float" when using sklearn bc muh pure python
        #new_x.append([ float(x) for x in np_x ] )
        new_x.append(  np_x )
        new_y.append( [np_y] )
        
        count += 1
   
    new_x = batch_x
    new_y = batch_y

    # split data into train, test and validation
    ratio = (len(new_x) / 100 ) * (train_test_split * 100)
    
    train_x = new_x[:int(ratio)]
    train_y = new_y[:int(ratio)]

    if not validation:
        test_x = new_x[int(ratio):]
        test_y = new_y[int(ratio):]
        return [train_x, train_y], [test_x, test_y]
    
    else:
        rest_ratio = (len(new_x[int(ratio):]) / 100 ) * (validation * 100)

        test_x = new_x[int(ratio):-int(rest_ratio)]
        test_y = new_y[int(ratio):-int(rest_ratio)]
        val_x = new_x[-int(rest_ratio):]
        val_y = new_y[-int(rest_ratio):]
        return [train_x, train_y], [test_x, test_y], [val_x, val_y]


def new_load_mnist_data(train_test_split = 0.6, validation = False, batch_size=32):
    """
    Returns tuple of inputs and targets.
    In total its 70,000 images with the dimensions 28x28.
    
    Keyword arguments:
    train_test_split -- splitting ratio between training and test data (default 0.6)
    validation -- if value provided its the splitting ratio between test and validation data (default False)
    batch_size -- the size of each batch (default 32)
    """
    #  this part is commented out since the data is already loaded and saved
    """
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    
    for i in range(0, 5):
        with open(f"pickled_data/mnist_data_X_{i+1}.pkl", "wb") as file:
            chunk = math.ceil(len(X) / 5)
            pickle.dump(X.tolist()[chunk * i : chunk * (i+1)], file)
        
        with open(f"pickled_data/mnist_data_y_{i+1}.pkl", "wb") as file:
            chunk = math.ceil(len(X) / 5)
            pickle.dump(y.tolist()[chunk * i : chunk * (i+1)], file)
    """
    # read pickled mnist data 
    X = []
    y = []
    for i in range(0, 5):
        with open(f"pickled_data/mnist_data_X_{i+1}.pkl", "rb") as file:
            X += pickle.load(file)
        
        with open(f"pickled_data/mnist_data_y_{i+1}.pkl", "rb") as file:
            y += pickle.load(file)

     # shuffle data
    p = [(i,j) for i,j in zip(X, y)]
    random.shuffle(p)
    X = [i[0] for i in p]
    y = [i[1] for i in p]

    new_x = []
    new_y = []

    batch = batch_size
    count = 0

    batch_x = []
    batch_y = []

    for np_x, np_y in zip(X,y):
        
        if count == batch:
            batch_x.append(new_x)
            batch_y.append(new_y)
            
            new_x = []
            new_y = []
            count = 0
  
        new_x.append([ float(x / 255.0) for x in np_x ] )
        new_y.append( [int(elem_y) for elem_y in np_y] )
        
        count += 1
   
    new_x = batch_x
    new_y = batch_y

    # split data into train, test and validation
    ratio = (len(new_x) / 100 ) * (train_test_split * 100)
    
    train_x = new_x[:int(ratio)]
    train_y = new_y[:int(ratio)]

    if not validation:
        test_x = new_x[int(ratio):]
        test_y = new_y[int(ratio):]
        return [train_x, train_y], [test_x, test_y]
    
    else:
        rest_ratio = (len(new_x[int(ratio):]) / 100 ) * (validation * 100)

        test_x = new_x[int(ratio):-int(rest_ratio)]
        test_y = new_y[int(ratio):-int(rest_ratio)]
        val_x = new_x[-int(rest_ratio):]
        val_y = new_y[-int(rest_ratio):]
        return [train_x, train_y], [test_x, test_y], [val_x, val_y]