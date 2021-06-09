# Gender Classification Using Deep Learning

## Deep learning models 

### CNN
### MLP 
### RNN
### BiLSTM
### GRU

Usage is simple. 

    testename = prepare_encod_names({"cibely"})   # Now the names are encod as a vector of numbers with weight
    resu=(LSTMmodel.predict(testename) > 0.5).astype("int32")
    if int(resu)==1:
      print('M')
    else:
      print('F')
      
    out: F
