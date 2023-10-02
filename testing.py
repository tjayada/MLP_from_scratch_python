from model_helper import hit_or_miss, r_2, get_argmax, flatten, one_hot
# import model_helper dictionary that contains acti and error funcs and their derivatives
# then use dic in train loop by accesing model.layers[-1].error_function_derivative to get og
# to then use in train loop

def test(model, X, y, accuracy_measure = hit_or_miss):

    preds = []
    trues = []
    for batch_X, batch_y in zip(X, y):

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