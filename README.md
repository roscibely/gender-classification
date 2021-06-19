# Gender Classification Using Deep Learning

We classified the gender of Brazilian names using deep learning. See the document ![here](https://github.com/roscibely/Gender-Classification/blob/main/PREDICTING%20GENDER%20%20OF%20BRAZILIAN%20NAMES%20USING%20DEEPLEARNING.pdf).

![Animation](https://github.com/roscibely/Gender-Classification/blob/main/animation.gif)

## Download the dataset

    url = "https://data.brasil.io/dataset/genero-nomes/nomes.csv.gz"
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

## Deep learning models 

Models: ![CNN](https://github.com/roscibely/Gender-Classification/blob/main/models/CNN.h5), ![MLP](https://github.com/roscibely/Gender-Classification/blob/main/models/DNN.h5), ![RNN](https://github.com/roscibely/Gender-Classification/blob/main/models/RNN.h5), ![BiLSTM](https://github.com/roscibely/Gender-Classification/blob/main/models/LSTM.h5), ![GRU](https://github.com/roscibely/Gender-Classification/blob/main/models/GRU.h5). 

Usage is simple. 

    testename = prepare_encod_names({"cibely"})   # name are encod as a vector of numbers
    resu=(LSTMmodel.predict(testename) > 0.5).astype("int32")
    if int(resu)==1:
      print('M')
    else:
      print('F')
      
    out: F
