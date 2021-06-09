# -*- coding: utf-8 -*-

# 1. Gender Classification Using Deep Learning


# 1.1 Necessary libraries
import pandas as pd                       # structures and data analysis
import numpy as np
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, GRU, SimpleRNN, LSTM, Bidirectional, Activation, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# %matplotlib inline
import itertools

# 1.2 Download the dataset
import requests
url = "https://data.brasil.io/dataset/genero-nomes/nomes.csv.gz"
filename = url.split("/")[-1]
with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

df = pd.read_csv('nomes.csv.gz') 
df.head()
df.info()
df.describe()

# 1.2.1 Preparing the data
y = df['classification'].astype("category").cat.codes.values    # y labels into numbers 0 is F and 1 is M
names = df['first_name'].apply(lambda x: x.lower())             # input names
plt.figure(figsize=(9,5))
plt.rcParams.update({'font.size': 12})
plt.hist([len(a) for a in names], bins=25, color='black')
plt.xlabel("Length of the names")
plt.ylabel("Number of names")
plt.grid(color='gray', linestyle='-', linewidth=0.3)
plt.gcf().savefig('lennames.eps', format='eps', dpi=600)     # save the figure
print("M : " + str(sum(y==1)))
print("F : " + str(sum(y==0)))
print(len(y))

# 1.3 Encoding Words
maxlen = 20                                               # max lenght of a name
'''Define a vocabulary which corresponds to all the unique letters encountered'''
vocab = set(' '.join([str(i) for i in names]))            # creating a vocab
vocab.add('END')
len_vocab = len(vocab)
char_index = dict((c, i) for i, c in enumerate(vocab))    # creating a dictionary
''' The dictionary maps each letter of vocabulary to a number '''
def set_flag(i):
    aux = np.zeros(len_vocab);
    aux[i] = 1
    return list(aux)
# Truncate names and create the matrix
def prepare_encod_names(X):
    vec_names = []
    trunc_name = [str(i)[0:maxlen] for i in X]  # consider only the first 20 characters
    for i in trunc_name:
        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0,maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        vec_names.append(tmp)
    return vec_names

x = prepare_encod_names(names.values)   # Now the names are encod as a vector of numbers with weight

# 1.4 Split the data into test and train"""
# train, val, test set will be 60%, 20%, 20% of the dataset respectively
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=40)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_val = np.asarray(x_val)
y_val = np.asarray(y_val)

# 1.5 Building the CNN Model
# initilaizing the Sequential nature for CNN model
CNNmodel = keras.Sequential()  
# 32 convolution filters used each of size 2
CNNmodel.add(Conv1D(32, 2, activation='relu', input_shape=(20,28) ))
# 64 convolution filters used each of size 3
CNNmodel.add(Conv1D(64, 3, activation='relu' ))
# choose the best features via pooling
CNNmodel.add(MaxPooling1D(pool_size=(1,)))
# randomly turn neurons on and off to improve convergence
CNNmodel.add(Dropout(0.25))
# flatten we only want a classification output
CNNmodel.add(Flatten())
# fully connected to get all relevant data
CNNmodel.add(Dense(150, activation='relu'))
# one more dropout
CNNmodel.add(Dropout(0.1))
# output 
CNNmodel.add(Dense(1, activation='sigmoid'))
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adagrad(learning_rate=lr_schedule)
CNNmodel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
CNNhistory = CNNmodel.fit(x_train, y_train, validation_data=(x_val, y_val),  epochs=20, batch_size=128) # Getting score metrics
scores = CNNmodel.evaluate(x_test, y_test) 
print("Accuracy: %.2f%%" % (scores[1]*100))

# 1.6 Building the RNN Model
RNNmodel=keras.Sequential()
RNNmodel.add(SimpleRNN(32, input_shape=(x_train.shape[1],x_train.shape[2]), activation='tanh', return_sequences=True))
RNNmodel.add(TimeDistributed(Dense(1))) 
RNNmodel.add(Flatten())
RNNmodel.add(Dense(1, activation='sigmoid'))
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
RNNmodel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
RNNhistory=RNNmodel.fit(x_train, y_train , epochs=20, batch_size=128, validation_data=(x_val,y_val))
RNNscores = RNNmodel.evaluate(x_test, y_test) 
print("Accuracy: %.2f%%" % (RNNscores[1]*100))

# 1.7 Building the GRU Model
GRUmodel=keras.Sequential()
GRUmodel.add(GRU(32, input_shape=(x_train.shape[1],x_train.shape[2]),activation='tanh', return_sequences=True))
GRUmodel.add(TimeDistributed(Dense(1))) 
GRUmodel.add(Flatten())
GRUmodel.add(Dense(1, activation='sigmoid'))
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
GRUmodel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
GRUhistory=GRUmodel.fit(x_train, y_train , epochs=20, batch_size=128, validation_data=(x_val,y_val))
GRUscores = GRUmodel.evaluate(x_test, y_test) 
print("Accuracy: %.2f%%" % (GRUscores[1]*100))

# 1.8 Building the DNN Model
DNNmodel = keras.Sequential()                                   # Sequential model
n_inputs = 28                                                   # nº of states  
inputs = keras.Input(shape=(20,n_inputs,), name = 'input')      # Define inputs
l1=layers.Dense(64, activation='relu', name = 'hidden_layer_1')(inputs) 
ly2=layers.Dense(128, activation='softplus', name = 'hidden_layer_2')(l1) 
l3=layers.Dropout(0.25)(ly2)
l4=layers.Flatten()(l3)                 
output=[] 
output.append(layers.Dense(1, activation='sigmoid', name = 'output')(l4))                                    
DNNmodel = keras.Model(inputs=inputs, outputs=output)       
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
DNNmodel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
DNNhistory = DNNmodel.fit(x_train, y_train, batch_size=32, epochs=20, validation_data =(x_val, y_val))
DNNscores = DNNmodel.evaluate(x_test, y_test) 
print("Accuracy: %.2f%%" % (DNNscores[1]*100))

# 1.9 Building the BiLSTM Model
LSTMmodel = keras.Sequential()
LSTMmodel.add(Bidirectional(LSTM(64, return_sequences=True), backward_layer=LSTM(64, return_sequences=True, go_backwards=True), input_shape=(maxlen,len_vocab)))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(Bidirectional(LSTM(64)))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(Dense(1, activity_regularizer=l2(0.002)))
LSTMmodel.add(Activation('sigmoid'))
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
LSTMmodel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
LSTMhistory = LSTMmodel.fit(x_train, y_train, batch_size=128, epochs=20, validation_data =(x_val, y_val))
LSTMscores = LSTMmodel.evaluate(x_test, y_test) 
print("Accuracy: %.2f%%" % (LSTMscores[1]*100))

# 1.10 Figures
plt.figure(1)
plt.figure(figsize=(9,5))
plt.rcParams.update({'font.size': 12})
# summarize for accuracy
plt.plot(CNNhistory.history['accuracy'], 'g:')
plt.plot(LSTMhistory.history['accuracy'], 'r--')
plt.plot(RNNhistory.history['accuracy'], 'c-.')
plt.plot(DNNhistory.history['accuracy'], 'm-+')
plt.plot(GRUhistory.history['accuracy'], 'y-o')
plt.plot(CNNhistory.history['val_accuracy'], 'g-*')
plt.plot(LSTMhistory.history['val_accuracy'], 'r-v')
plt.plot(RNNhistory.history['val_accuracy'], 'c-h')
plt.plot(DNNhistory.history['val_accuracy'], 'm->')
plt.plot(GRUhistory.history['val_accuracy'], 'y-s')
plt.grid('True')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.8, 0.99)
plt.xlim(0,25)
plt.legend(['Train-CNN', 'Train-BiLSTM', 'Train-RNN', 'Train-MLP', 'Train-GRU','Val-CNN',\
            'Val-BiLSTM', 'Val-RNN', 'Val-MLP', 'Val-GRU'], loc='upper right')

plt.figure(2)
# summarize for loss
plt.plot(CNNhistory.history['loss'], 'g:')
plt.plot(LSTMhistory.history['loss'], 'r--')
plt.plot(RNNhistory.history['loss'], 'c-.')
plt.plot(DNNhistory.history['loss'], 'm-+')
plt.plot(GRUhistory.history['loss'], 'y-o')
plt.plot(CNNhistory.history['val_loss'], 'g-*')
plt.plot(LSTMhistory.history['val_loss'], 'r-v')
plt.plot(RNNhistory.history['val_loss'], 'c-H')
plt.plot(DNNhistory.history['val_loss'], 'm->')
plt.plot(GRUhistory.history['val_loss'], 'y-s')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train-CNN', 'Train-BiLSTM', 'Train-RNN', 'Train-MLP', 'Train-GRU','Val-CNN',\
            'Val-BiLSTM', 'Val-RNN', 'Val-MLP', 'Val-GRU'], loc='upper right')
plt.grid('True')
plt.xlim(0,32)

# 1.11 Statistics
y_predCNN =(CNNmodel.predict(x_test) > 0.5).astype("int32")
y_predLSTM =(LSTMmodel.predict(x_test) > 0.5).astype("int32")
y_predRNN =(RNNmodel.predict(x_test) > 0.5).astype("int32")
y_predGRU =(GRUmodel.predict(x_test) > 0.5).astype("int32")
y_predDNN =(DNNmodel.predict(x_test) > 0.5).astype("int32")


cm1 = confusion_matrix(y_true=y_test, y_pred=y_predLSTM)
cm2 = confusion_matrix(y_true=y_test, y_pred=y_predCNN)
cm3 = confusion_matrix(y_true=y_test, y_pred=y_predDNN)
cm4 = confusion_matrix(y_true=y_test, y_pred=y_predGRU)
cm5 = confusion_matrix(y_true=y_test, y_pred=y_predRNN)

fig, axes = plt.subplots(1, 5, figsize=(12, 4))
fig.tight_layout(pad=2.0)            
plt.rcParams.update({'font.size': 12}) 
cmap=plt.cm.Greys

axes[0].imshow(cm1, interpolation='nearest', cmap=cmap)
axes[0].set_title('BiLSTM')
#axes[0].colorbar()
thresh = cm1.max() / 2.
for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
  axes[0].text(j, i, cm1[i, j],
  horizontalalignment="center",
  color="white" if cm1[i, j] > thresh else "black")
  axes[0].set_ylabel('True label')
  axes[0].set_xlabel('Predicted label')

axes[1].imshow(cm2, interpolation='nearest', cmap=cmap)
axes[1].set_title('CNN')
thresh = cm2.max() / 2.
for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
  axes[1].text(j, i, cm2[i, j],
  horizontalalignment="center",
  color="white" if cm2[i, j] > thresh else "black")
  #axes[1].set_ylabel('True label')
  axes[1].set_xlabel('Predicted label')

axes[2].imshow(cm3, interpolation='nearest', cmap=cmap)
axes[2].set_title('MLP')
thresh = cm3.max() / 2.
for i, j in itertools.product(range(cm3.shape[0]), range(cm3.shape[1])):
  axes[2].text(j, i, cm3[i, j],
  horizontalalignment="center",
  color="white" if cm3[i, j] > thresh else "black")
  #axes[2].set_ylabel('True label')
  axes[2].set_xlabel('Predicted label')

axes[3].imshow(cm4, interpolation='nearest', cmap=cmap)
axes[3].set_title('GRU')
thresh = cm3.max() / 2.
for i, j in itertools.product(range(cm4.shape[0]), range(cm4.shape[1])):
  axes[3].text(j, i, cm4[i, j],
  horizontalalignment="center",
  color="white" if cm4[i, j] > thresh else "black")
  #axes[3].set_ylabel('True label')
  axes[3].set_xlabel('Predicted label')


axes[4].imshow(cm5, interpolation='nearest', cmap=cmap)
axes[4].set_title('RNN')
thresh = cm3.max() / 2.
for i, j in itertools.product(range(cm5.shape[0]), range(cm5.shape[1])):
  axes[4].text(j, i, cm5[i, j],
  horizontalalignment="center",
  color="white" if cm5[i, j] > thresh else "black")
  #axes[4].set_ylabel('True label')
  axes[4].set_xlabel('Predicted label')


plt.show(block=False)


# Finding precision and recall
from sklearn.metrics import precision_score, recall_score
print(precision_score(y_test, y_predLSTM))
print(recall_score(y_test, y_predLSTM))

print(precision_score(y_test, y_predRNN))
print(recall_score(y_test, y_predRNN))

print(precision_score(y_test, y_predCNN))
print(recall_score(y_test, y_predCNN))

print(precision_score(y_test, y_predGRU))
print(recall_score(y_test, y_predGRU))

print(precision_score(y_test, y_predDNN))
print(recall_score(y_test, y_predDNN))

# 1.12 Testing
testename = prepare_encod_names({"cibely"})   # Now the names are encod as a vector of numbers with weight
resu=(LSTMmodel.predict(testename) > 0.5).astype("int32")
if int(resu)==1:
  print('M')
else:
  print('F')