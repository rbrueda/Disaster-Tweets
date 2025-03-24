import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import re
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Dropout, MultiHeadAttention, Input, LayerNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import fasttext.util
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import LearningRateScheduler

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load FastText model
fasttext.util.download_model('en', if_exists='ignore')
ft_model = fasttext.load_model('cc.en.300.bin')

lemmatizer = WordNetLemmatizer()

df = pd.read_csv('train.csv')
df = df.dropna(subset=['text'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

df['tokens'] = df['text'].apply(clean_text)
df['target'] = df['target'].astype(int)

MAX_FEATURES = 10000
EMBEDDING_DIM = 300

all_words = [word for tokens in df['tokens'] for word in tokens]
freq_dist = FreqDist(all_words)
vocab = [word for word, freq in freq_dist.most_common(MAX_FEATURES - 1)]
word_index = {word: i + 1 for i, word in enumerate(vocab)}

embedding_matrix = np.zeros((MAX_FEATURES, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_FEATURES:
        embedding_matrix[i] = ft_model.get_word_vector(word)

df['sequences'] = df['tokens'].apply(lambda tokens: [word_index.get(word, 1) for word in tokens])
X = pad_sequences(df['sequences'], padding='post', truncating='post')
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_layer = Input(shape=(None,))
embedding_layer = Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], 
                            trainable=False, mask_zero=True)(input_layer)
conv_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding_layer)

attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(conv_layer, conv_layer)
attention_output = Dropout(0.3)(attention_output)
attention_output = LayerNormalization()(attention_output)

ffn = Dense(128, activation='relu')(attention_output)
ffn_output = Dense(64, activation='relu')(ffn)
residual_connection = tf.keras.layers.Add()([conv_layer, ffn_output])
residual_connection = LayerNormalization()(residual_connection)

pooled_output = GlobalAveragePooling1D()(residual_connection)

dense_layer_1 = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(pooled_output)
dropout_1 = Dropout(0.4)(dense_layer_1)
dense_layer_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(dropout_1)
dropout_2 = Dropout(0.4)(dense_layer_2)

output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0005))(dropout_2)

model = Model(inputs=input_layer, outputs=output_layer)

initial_lr = 0.001
lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=1000, decay_rate=0.9, staircase=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.show()
