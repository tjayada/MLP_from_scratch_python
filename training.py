from model_helper import accuracy_measure, r_2, get_argmax, flatten, one_hot
# import model_helper dictionary that contains acti and error funcs and their derivatives
# then use dic in train loop by accesing model.layers[-1].error_function_derivative to get og
# to then use in train loop

def train(model, X, y, epochs, show_epochs):
    
    # error_func = dic[f"{model.error_func_derivative}"]

    for epoch in range(epochs):
        accuracy = 0
        loss = 0
        preds = []
        trues = []
        for batch_X, batch_y in zip(X, y):
            #batch_X = [ [float(x) for x in batch_X]]

            #batch_X = [ batch_X ]
            #print(batch_y)
            #og_y = [int(elem_y) for elem_y in batch_y]
            #print(og_y)

            #batch_y = [ one_hot(int(elem_y), 10) for elem_y in og_y ]
            #print("training")
            #print(batch_y)


            model(batch_X, [batch_y])
            #print(model.output)

            #pred = [ get_argmax(elem_x) for elem_x in model.output ]
            #print(pred)

            #accuracy += accuracy_measure(model.output, batch_y, round_function=1, error_margin=1)
            #print(accuracy)

            preds.append(model.output[0][0])
            trues.append(batch_y[0])

            loss += model.error_function(model.output, [batch_y])[0]
            #print(loss)

            model.backpropagation()

        if epoch % show_epochs == 0:
            r2 = r_2(preds, trues)
            print("Epoch ", epoch, ",    Accuracy: ", r2 , ",    Loss: ", loss/len(X))
    
    return model
    
