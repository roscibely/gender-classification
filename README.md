# Predicting Gender by First Name Using Character-level Machine Learning

We classified the gender of Brazilian names using deep learning and machine learning. See the document [here](https://arxiv.org/abs/2106.10156).

## Download the dataset
```python
    url = "https://data.brasil.io/dataset/genero-nomes/nomes.csv.gz"
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)
```

## Deep learning models 

Models: ![CNN](https://github.com/roscibely/Gender-Classification/blob/main/models/CNN.h5), ![MLP](https://github.com/roscibely/Gender-Classification/blob/main/models/DNN.h5), ![RNN](https://github.com/roscibely/Gender-Classification/blob/main/models/RNN.h5), ![BiLSTM](https://github.com/roscibely/Gender-Classification/blob/main/models/LSTM.h5), ![GRU](https://github.com/roscibely/Gender-Classification/blob/main/models/GRU.h5). 

Usage is simple. 
```python
    testename = prepare_encod_names({"cibely"})   # name are encod as a vector of numbers
    resu=(LSTMmodel.predict(testename) > 0.5).astype("int32")
    if int(resu)==1:
      print('M')
    else:
      print('F')
      
    out: F
```

 ## Papers
 
 R. C. B. Rego, G. d. S. Nascimento, D. E. d. L. Rodrigues, S. M. Nascimento and V. M. L. Silva, ["Brazilian scientific productivity from a gender perspective during the Covid-19 pandemic: classification and analysis via machine learning,"](https://ieeexplore.ieee.org/document/10015223) in IEEE Latin America Transactions, vol. 21, no. 2, pp. 302-309, Feb. 2023, doi: 10.1109/TLA.2023.10015223.
    
  Rego, R. C., Silva, V. M. & Fernandes, V. M. (2021). [Predicting Gender by First Name Using Character-level Machine Learning](https://arxiv.org/abs/2106.10156v2). arXiv preprint arXiv:2106.10156 v2.  
  
 Rego, R. C., & Silva, V. M. (2021). [Predicting gender of Brazilian names using deep learning](https://arxiv.org/abs/2106.10156v1). arXiv preprint arXiv:2106.10156 v1.   
    
