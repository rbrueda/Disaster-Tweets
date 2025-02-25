import numpy as np
import pandas as pd
import re
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#todo: optimize the code better -- add better optimizations for the data
#todo: add lemmanization for data preprocessing - generalization
#todo: add work cloud for manual tuning -- if disaster tweets have depcific words, increase max_features or adjust embedding szie
#todo: add Word2Vec or Glove instead of normal embedding
#todo: after manual tuning -- apply Baysian optimization -- batch_size, filters, kernel_size
#todo: apply TF-IDF visualization or bigram analysis to see co-occuring words
#todo: based on the output I get, apply early stopping!!
#todo: try Learning Rate Scheduling to reduce learning rate when model plateaus
#todo: plot the ROC_AUC curves

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('train.csv')

# Drop missing values in 'text' column
df = df.dropna(subset=['text'])

# Preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions & hashtags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

# Apply preprocessing
df['tokens'] = df['text'].apply(clean_text)

# Convert labels to integer
df['target'] = df['target'].astype(int)

# Hyperparameters
MAX_FEATURES = 10000  # Vocabulary size
MAX_LEN = 100  # Max sequence length

# Vocabulary frequency distribution
all_words = [word for tokens in df['tokens'] for word in tokens]
freq_dist = FreqDist(all_words)

# Get top words
vocab = [word for word, freq in freq_dist.most_common(MAX_FEATURES - 1)]
word_index = {word: i + 1 for i, word in enumerate(vocab)}  # Map unknown words to 1

# Convert text tokens into sequences of indices
df['sequences'] = df['tokens'].apply(lambda tokens: [word_index.get(word, 1) for word in tokens])  # Use 1 for unknown words

# Padding function
def pad_sequence(seq, max_len):
    if len(seq) > max_len:
        return seq[:max_len]  # Truncate if too long
    else:
        return seq + [0] * (max_len - len(seq))  # Pad with zeros

df['padded_sequences'] = df['sequences'].apply(lambda seq: pad_sequence(seq, MAX_LEN))

# Convert to NumPy arrays
X = np.array(df['padded_sequences'].tolist())  # Input
y = df['target'].values  # Output

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Embedding(input_dim=MAX_FEATURES, output_dim=128, input_length=MAX_LEN),  # Word embeddings
    Conv1D(filters=64, kernel_size=3, activation='relu'),  # Convolution layer
    GlobalMaxPooling1D(),  # Max pooling
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Regularization
    Dense(1, activation='sigmoid')  # Output layer
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
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
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.show()

