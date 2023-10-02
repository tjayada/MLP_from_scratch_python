from sklearn.datasets import fetch_california_housing, fetch_openml
import sklearn
import random 

def load_logic_gates(logic = "and"):
    inputs      =   [[0,0],
                    [1,0],
                    [0,1],
                    [1,1]]

    # all the logical gates
    and_labels  =   [[0],
                    [0],
                    [0],
                    [1]]

    or_labels   =   [[0],
                    [1],
                    [1],
                    [1]]

    not_and_labels  =   [[1],
                        [1],
                        [1],
                        [0]]

    not_or_labels   =   [[1],
                        [0],
                        [0],
                        [0]]

    xor_labels      =   [[0],
                        [1],
                        [1],
                        [0]]
    
    logic_dictionary = {
        "and"       : and_labels,
        "or"        : or_labels,
        "not_and"   : not_and_labels,
        "not_or"    : not_or_labels,
        "xor"       : xor_labels
    }

    return inputs, logic_dictionary[logic]



def load_housing_data(train_test_split = 0.6, validation = False):
    X, y = fetch_california_housing(return_X_y=True)
    import sklearn

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # might be overkill, but in the spirit of no extra libraries, the data is
    # converted from numpy arrays to lists
    new_x = []
    new_y = []
    for np_x, np_y in zip(X,y):
        new_x.append([ [float(x) for x in np_x]] )
        new_y.append([ [np_y] ])

    # shuffle data
    p = [(i,j) for i,j in zip(new_x, new_y)]
    random.shuffle(p)
    new_x = [i[0] for i in p]
    new_y = [i[1] for i in p]

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
    



def new_load_housing_data(train_test_split = 0.6, validation = False, batch_size=32):
    X, y = fetch_california_housing(return_X_y=True)
    import sklearn

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    # shuffle data
    p = [(i,j) for i,j in zip(X, y)]
    random.shuffle(p)
    X = [i[0] for i in p]
    y = [i[1] for i in p]


    # might be overkill, but in the spirit of no extra libraries, the data is
    # converted from numpy arrays to lists
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
        #np_x = np_x[:20]    
        new_x.append([ float(x) for x in np_x ] )
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




def load_mnist_data(train_test_split = 0.6, validation = False):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    # might be overkill, but in the spirit of no extra libraries, the data is
    # converted from numpy arrays to lists
    new_x = []
    new_y = []
    for np_x, np_y in zip(X,y):
        new_x.append([ [float(x) for x in np_x]] )
        new_y.append([ np_y ])

    # shuffle data
    p = [(i,j) for i,j in zip(new_x, new_y)]
    random.shuffle(p)
    new_x = [i[0] for i in p]
    new_y = [i[1] for i in p]

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



def new_load_mnist_data(train_test_split = 0.6, validation = False):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    

    #scaler = sklearn.preprocessing.StandardScaler()
    #scaler.fit(X)
    #X = scaler.transform(X)
    #print(len(X))

     # shuffle data
    p = [(i,j) for i,j in zip(X, y)]
    random.shuffle(p)
    X = [i[0] for i in p]
    y = [i[1] for i in p]


    # might be overkill, but in the spirit of no extra libraries, the data is
    # converted from numpy arrays to lists
    new_x = []
    new_y = []

    batch = 32
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
        #np_x = np_x[:20]    
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


