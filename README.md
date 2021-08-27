# Predicting gender of Brazilian names using deep learning

We classified the gender of Brazilian names using deep learning. See the document ![here](arXiv:2106.10156).

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
    
   ## How cite 

    @article{Rego2021PredictingGO,
        title={Predicting gender of Brazilian names using deep learning},
        author={Rosana C. B. Rego and Veronica M. L. Silva},
        journal={ArXiv},
        year={2021},
        volume={abs/2106.10156}
    }
