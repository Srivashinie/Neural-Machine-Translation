import numpy as np
import pandas as pd
import re
import string
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, TimeDistributed
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#Load the dataset
df = pd.read_csv("/content/eng_-french.csv")
df.columns = ['en', 'fr']

#Drop rows with missing values
df.dropna(inplace=True)

#Define a function to clean text data
custom_punct = string.punctuation.replace("-", "").replace("'", "")
def clean(text):
    text = text.lower()
    text = re.sub("[" + custom_punct + "]", "", text)
    return text

#Apply text cleaning to English and French columns
df["clean_en"] = df["en"].apply(clean)
df["clean_fr"] = df["fr"].apply(clean)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["clean_en"], df["clean_fr"], test_size=0.2)

#Tokenize English and French texts
english_tokenizer = Tokenizer()
french_tokenizer = Tokenizer()
english_tokenizer.fit_on_texts(X_train)
french_tokenizer.fit_on_texts(y_train)

#Determine vocabulary sizes
size_input_vocab = len(english_tokenizer.index_word) + 1
size_output_vocab = len(french_tokenizer.index_word) + 1

#Convert texts to sequences and pad them
maxlen = 55  # max length of all sentences (EN: 48, FR: 55)
X_train_sequences = english_tokenizer.texts_to_sequences(X_train)
X_test_sequences = english_tokenizer.texts_to_sequences(X_test)
y_train_sequences = french_tokenizer.texts_to_sequences(y_train)
y_test_sequences = french_tokenizer.texts_to_sequences(y_test)
X_train_pad = pad_sequences(X_train_sequences, maxlen=maxlen, truncating='post', padding="post")
X_test_pad = pad_sequences(X_test_sequences, maxlen=maxlen, truncating='post', padding="post")
y_train_pad = pad_sequences(y_train_sequences, maxlen=maxlen, truncating='post', padding="post")
y_test_pad = pad_sequences(y_test_sequences, maxlen=maxlen, truncating='post', padding="post")
y_train_pad = y_train_pad.reshape(*y_train_pad.shape, 1)
y_test_pad = y_test_pad.reshape(*y_test_pad.shape, 1)

#Define the Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

#Define the token and position embedding layer
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

#Define model parameters
num_heads = 3  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
embedding_dim = 200
adam = Adam(learning_rate=0.003)

#Define the Transformer model
inputs = tf.keras.layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, size_input_vocab, embedding_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
x = transformer_block(x)
x = TimeDistributed(Dense(256, activation="relu"))(x)
outputs = TimeDistributed(Dense(size_output_vocab, activation="softmax"))(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
model.summary()


#Fit the model
history = model.fit(X_train_pad,
                    y_train_pad,
                    validation_data=(X_test_pad, y_test_pad),
                    verbose=1,
                    batch_size=128,
                    epochs=10,
                   )

#Test the model
samples = [
    "How are you today?",
    "Thank you.",
    "Can you help me?",
    "It’s nice to meet you!"
]
for sample in samples:
    pred = model.predict([pad_sequences(english_tokenizer.texts_to_sequences([sample]), maxlen=maxlen, padding='post', truncating='post')])[0].argmax(axis=1)
    output_text = french_tokenizer.sequences_to_texts([pred])[0]
    print("English: " + sample)
    print("French: " + output_text)
    print()

