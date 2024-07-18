import pandas as pd
import collections
import numpy as np
#from tokenizers import Tokenizer
from keras import Input, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM, Dropout
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import helper

english_sentences = helper.load_data('/content/en.txt')
# Load French data
french_sentences = helper.load_data('/content/fr.txt')


def tokenize(x):
    tokenizer = Tokenizer(split=' ', char_level=False)
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer          

#Padding
def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])

    return pad_sequences(x, maxlen=length, padding='post', truncating='post')

def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)+1

def logits_to_text(logits, tokenizer):

    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])



def token_to_words(sequence, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return [index_to_words[token] for token in sequence if index_to_words[token]!='<PAD>']

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    
    learning_rate = 0.001
    english_input = Input(shape=input_shape[1:], name="input_layer")    # the shape is (input length x 1) as batchsize excluded

    # LSTM takes as input (batchsize,input_length,1) and outputs (batchsize, input_length, 64) because return-seq=True
    x = LSTM(64, return_sequences=True, activation="tanh", name="LSTM_layer")(english_input)
    preds = TimeDistributed(Dense(french_vocab_size, activation="softmax"), name="Dense_layer")(x)
    model = Model(inputs=english_input, outputs=preds, name='simple_LSTM')

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model


# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))  # reshape as 3D (batchsize, timesteps, 1) for LSTM input

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)

simple_rnn_model.summary()

simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)

def translate(prediction, gold_standard):

    translation = logits_to_text(prediction[0], french_tokenizer)
    standard = ' '.join(token_to_words(gold_standard[0][:,0],french_tokenizer))
    print('---- Gold standard ----')
    print(standard)
    print()
    print('---- Prediction ----')
    for w_t, w_s in zip(translation.split(), standard.split()):
        if w_t == w_s:
            print('\033[0;30;0m','{}'.format(w_t), end='')
        else:
            print('\033[0;31;47m', w_t, end='')
    print()


# Print prediction(s)
print('---- Original ----')
print(' '.join(token_to_words(tmp_x[:1][0][:,0],english_tokenizer) ))
print()
translate(simple_rnn_model.predict(tmp_x[:1]), preproc_french_sentences[:1])