import math
from model_helper import hit_or_miss, one_hot

def train(model, train_data, test_data, val_data, epochs, show_epochs, accuracy_measure = hit_or_miss):
    """training function for model"""
    X_train, y_train = train_data
    if test_data:
        X_test, y_test = test_data
    if val_data:
        X_val, y_val = val_data

    for epoch in range(epochs):
        accuracy = 0
        loss = 0
        preds = []
        trues = []
        for batch_X, batch_y in zip(X_train, y_train):
            # differentiate between regression and classification, since classification needs one-hot encoding
            if model.classification:
                output_size = model.layer_config[-1]
                batch_not_encoded = batch_y
                batch_y = [ one_hot(int(elem_y), output_size) for elem_y in batch_y for elem_y in elem_y]    
                
                model(batch_X, batch_y)
                pred = model.prediction

                [ preds.append(elem_y) for elem_y in pred ]
                [ trues.append(elem_y[0]) for elem_y in batch_not_encoded]

            else:
                batch_y = batch_y     
                
                model(batch_X, batch_y)  
                
                pred = model.output
                preds.append(pred[0])
                trues.append(batch_y[0])

            loss += model.error_function(model.output, batch_y)
            model.backpropagation()

        if epoch % show_epochs == 0:
            train_acc = accuracy_measure(preds, trues)
            if test_data:
                test_acc = test(model, X_test, y_test, accuracy_measure=accuracy_measure)
            else:
                test_acc = math.nan

            if epoch + 1 < 10:
                space = "  "
            elif 9 < epoch + 1 < 100:
                space = " "
            else:
                space = ""
            
            print(f"Epoch {space}", epoch + 1, ",    Training Accuracy: ", "%.4f" % train_acc , ",    Test Accuracy: ", "%.4f" % test_acc , ",    Loss: ", "%.4f" % (loss/len(X_train)) )
    
    if val_data:
        val_acc = test(model, X_val, y_val, accuracy_measure=accuracy_measure)   
    else:
        val_acc = math.nan
    print("\n")
    print("    Validation Accuracy: ", "%.4f" % val_acc )
    return model

def test(model, X, y, accuracy_measure = hit_or_miss):
    """test accuracy of model"""
    preds = []
    trues = []
    for batch_X, batch_y in zip(X, y):
            # differentiate between regression and classification, since classification needs one-hot encoding
            if model.classification:
                batch_not_encoded = batch_y
                batch_y = [ one_hot(int(elem_y), 10) for elem_y in batch_y for elem_y in elem_y]
                
                model(batch_X, batch_y)

                pred = model.prediction

                [ preds.append(elem_y) for elem_y in pred ]
                [ trues.append(elem_y[0]) for elem_y in batch_not_encoded]

            else:
                batch_y = batch_y
                
                model(batch_X, batch_y)
                
                pred = model.output
                preds.append(pred[0])
                trues.append(batch_y[0])
    
    return accuracy_measure(preds, trues)