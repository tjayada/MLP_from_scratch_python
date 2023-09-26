from model_helper import hit_or_miss, r_2, get_argmax, flatten, one_hot
# import model_helper dictionary that contains acti and error funcs and their derivatives
# then use dic in train loop by accesing model.layers[-1].error_function_derivative to get og
# to then use in train loop

def train(model, X, y, epochs, show_epochs, accuracy_measure = hit_or_miss):
    
    # error_func = dic[f"{model.error_func_derivative}"]

    for epoch in range(epochs):
        accuracy = 0
        loss = 0
        preds = []
        trues = []
        for batch_X, batch_y in zip(X, y):
            #batch_X = [ [float(x) for x in batch_X]]

            #batch_X = [ batch_X ]
            #print(batch_X)
            #print(batch_y)
            #og_y = [int(elem_y) for elem_y in batch_y]
            #print(og_y)

            #batch_y = [ one_hot(int(elem_y), 10) for elem_y in og_y ]
            #print("training")
            #print(batch_y)

            if model.classification:
                batch_not_encoded = batch_y
                batch_y = [ one_hot(int(elem_y), 10) for elem_y in batch_y for elem_y in elem_y]
                
                model(batch_X, batch_y)

                pred = model.prediction
                #print(pred)
                #print(batch_not_encoded)
                [ preds.append(elem_y) for elem_y in pred ]
                [ trues.append(elem_y[0]) for elem_y in batch_not_encoded]
                #preds.append(pred[0])
                #trues.append(batch_not_encoded[0][0])



            else:
                batch_y = batch_y
                
                model(batch_X, batch_y)
                
                pred = model.output
                preds.append(pred[0])
                trues.append(batch_y[0])


            #model(batch_X, batch_y)
            #print(model.output)

            #pred = [ get_argmax(elem_x) for elem_x in model.output ]
            #pred = model.prediction
            #print(pred)

            #accuracy += accuracy_measure(model.output, batch_y, round_function=1, error_margin=1)
            #print(accuracy)
            #if model.classification:
            #    pred = model.prediction
            #else:
            #    pred = model.output

            #preds.append(pred[0])
            #trues.append(batch_y[0])
            
            #print(preds)
            #print(trues)
            #print(model.output)
            #print(batch_y)
            #print(model.error_function(model.output, batch_y))

            # change MSE to include sum() in return function to not have to use it here ?
            loss += sum(model.error_function(model.output, batch_y)) / len(batch_y)
            #loss += model.error_function(model.output, batch_y) / len(batch_y)
            
            
            #loss += sum(model.error_function(model.output, batch_y))
            #loss += accuracy_measure(pred, og_y)
            #print(loss)

            model.backpropagation()

            #break
        #break
        
        if epoch % show_epochs == 0:
            #r2 = r_2(preds, trues)
            #print(preds)
            #print(trues)
            acc = accuracy_measure(preds, trues)
            print("Epoch ", epoch + 1, ",    Accuracy: ", acc , ",    Loss: ", loss/len(X))
    
    return model
    
