
# coding: utf-8

# In[2]:



from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
from keras.callbacks import ModelCheckpoint
#path = get_file('salmos.txt', origin='http://www.manancialvox.com/fontedepaz/19-SALMOS.txt')
#path = get_file('ester.txt',origin="http://www.manancialvox.com/fontedepaz/17-ESTER.txt")
#path = get_file('biblia.txt',origin="https://raw.githubusercontent.com/eurismarpires/exercicios_deep_learning/master/B%C3%ADblia%20Sagrada.txt")
path = get_file('biblia_sagrada.txt',origin="http://drupalbible.org/sites/drupalbible.org/files/Biblia%20ACF%20(Almeida%20Corrigida%20Fiel)%20-%20palavras%20de%20Jesus%20em%20vermelho.txt")
#path = get_file('biblia.txt','biblia.txt')
text = open(path).read().lower()


# In[3]:

# In[13]:

#text = text[2450:150000] 


# In[4]:

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[5]:

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))


# In[6]:

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[7]:

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
filename="weights-biblia-00-1.3100.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[8]:

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[10]:

start_index = random.randint(0, len(text) - maxlen - 1)

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()


# In[ ]:



