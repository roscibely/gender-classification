# Gender Classification Using Deep Learning

## Deep learning models 

Models: CNN, MLP, RNN, ![BiLSTM](https://github.com/roscibely/Gender-Classification/blob/main/models/BiLSTM_model.ipynb), GRU. 

Usage is simple. 

    testename = prepare_encod_names({"cibely"})   # name are encod as a vector of numbers
    resu=(LSTMmodel.predict(testename) > 0.5).astype("int32")
    if int(resu)==1:
      print('M')
    else:
      print('F')
      
    out: F
