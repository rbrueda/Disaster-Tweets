import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import re
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Reshape, Lambda, Input, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import LearningRateScheduler

# Learning rate scheduler
def one_cycle_lr(epoch):
    max_lr = 0.001
    min_lr = max_lr / 10
    return max_lr * (1 - epoch / 10) + min_lr

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmanizer = WordNetLemmatizer()

# Load dataset
df = pd.read_csv('train.csv')
print(df['target'].value_counts())  # Count occurrences of each class

# Drop missing values in 'text' column
df = df.dropna(subset=['text'])

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmanizer.lemmatize(word) for word in tokens]
    return tokens

# Apply preprocessing
df['tokens'] = df['text'].apply(clean_text)

# Convert labels to integer
df['target'] = df['target'].astype(int)

# Hyperparameters
MAX_FEATURES = 10000  # Vocabulary size
EMBEDDING_DIM = 100  # Standard embedding size

# Vocabulary frequency distribution
all_words = [word for tokens in df['tokens'] for word in tokens]
freq_dist = FreqDist(all_words)

# Get top words
vocab = [word for word, freq in freq_dist.most_common(MAX_FEATURES - 1)]
word_index = {word: i + 1 for i, word in enumerate(vocab)}  # Map unknown words to 1

# Convert text tokens into sequences of indices
df['sequences'] = df['tokens'].apply(lambda tokens: [word_index.get(word, 1) for word in tokens])

# Apply Adaptive Padding
X = pad_sequences(df['sequences'], padding='post', truncating='post')

# Convert to NumPy array
y = df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model inputs
input_layer = Input(shape=(None,))

# Embedding layer without FastText
embedding_layer = Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_DIM, trainable=True, mask_zero=True)(input_layer)

# Convolutional layer
conv_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding_layer)

# Hybrid Pooling
global_avg_pooling = GlobalAveragePooling1D()(conv_layer)
global_max_pooling = GlobalMaxPooling1D()(conv_layer)
pooled_output = tf.keras.layers.Concatenate()([global_avg_pooling, global_max_pooling])

# Dense layers
dense_layer_1 = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(pooled_output)
dropout_1 = Dropout(0.4)(dense_layer_1)
dense_layer_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(dropout_1)
dropout_2 = Dropout(0.4)(dense_layer_2)

# Output layer
output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0005))(dropout_2)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Learning rate scheduling
initial_lr = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Callbacks
lr_callback = LearningRateScheduler(one_cycle_lr)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_callback],
    verbose=1
)

# Predicting probabilities
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot training accuracy
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.show()
