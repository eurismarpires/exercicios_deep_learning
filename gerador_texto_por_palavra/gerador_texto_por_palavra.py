
# coding: utf-8

# In[1]:

# Load LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[2]:

# load ascii text and covert to lowercase
filename = "teste.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()


# In[3]:

palavras_texto = raw_text.split()


# In[4]:

print(palavras_texto[:10])


# In[5]:

len(palavras_texto)


# In[6]:

palavras = sorted(list(set(palavras_texto)))


# In[7]:

print(palavras[:10])


# In[8]:

palavra_to_int = dict((c,i) for i, c in enumerate(palavras))


# In[9]:

int_to_palavra = dict((i, c) for i, c in enumerate(palavras))


# In[10]:

n_palavras = len(palavras_texto)


# In[11]:

n_vocab = len(palavras)


# In[12]:

print("Total Characters: ", n_palavras)
print("Total Vocab: ", n_vocab)


# In[13]:

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []


# In[14]:

for i in range(0, n_palavras - seq_length, 1):
    seq_in = palavras_texto[i:i + seq_length]
    seq_out = palavras_texto[i + seq_length]
    dataX.append([palavra_to_int[palavra] for palavra in seq_in])
    dataY.append(palavra_to_int[seq_out])
n_patterns = len(dataX)    


# In[15]:

print("Total Patterns: ", n_patterns)


# In[16]:

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))


# In[17]:

X.shape


# In[18]:

# normalize
X = X / float(n_vocab)


# In[19]:

# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# In[20]:

# define the LSTM model
model = Sequential()


# In[21]:

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))


# In[22]:

model.add(Dropout(0.2))


# In[23]:

model.add(Dense(y.shape[1], activation='softmax'))


# In[24]:

model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[25]:

# define the checkpoint
filepath="palavras-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[ ]:

# fit the model
model.fit(X, y, nb_epoch=1000, batch_size=512, callbacks=callbacks_list)


# In[ ]:



