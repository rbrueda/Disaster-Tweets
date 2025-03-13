import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from transformers import TFDistilBertModel, DistilBertTokenizer
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import wordnet

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Load DistilBERT model and tokenizer
distilbert_encoder = TFDistilBertModel.from_pretrained("distilbert-base-uncased", trainable=False)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
lemmatizer = WordNetLemmatizer()

# Custom F1 score function to avoid naming conflict
def custom_f1_score(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float16")
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    precision = tp / (K.sum(y_pred) + K.epsilon())
    recall = tp / (K.sum(y_true) + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

# Transformer layer class
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, transformer, **kwargs):
        super().__init__(**kwargs)
        self.transformer = transformer

    def call(self, inputs):
        input_ids = inputs
        transformer_output = self.transformer(input_ids=input_ids, training=False)
        return transformer_output.last_hidden_state

# Function to build model with hyperparameters
def build_model(transformer, max_len=128):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    
    transformer_layer = TransformerLayer(transformer)
    transformer_output = transformer_layer(input_ids)
    
    conv1D = Conv1D(filters=128, 
                    kernel_size=3, 
                    activation="relu", padding="same")(transformer_output)
    pooling = GlobalAveragePooling1D()(conv1D)
    
    fc = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(pooling)
    dropout = Dropout(0.3)(fc)
    output = Dense(1, activation="sigmoid")(dropout)
    
    model = Model(inputs=[input_ids], outputs=output)
    #0.0046928 result from keras tuner
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0046928),
                  loss="binary_crossentropy",
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    return model

# Text preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load and preprocess dataset
df = pd.read_csv('train.csv')
df['clean_text'] = df['text'].apply(clean_text)
df.dropna(subset=['clean_text'], inplace=True)

# Tokenization
tokens = tokenizer(list(df['clean_text']), padding='max_length', truncation=True, max_length=128, return_tensors="tf")

# Extract inputs
input_ids = tokens['input_ids']
y_train = df['target'].values

# # Set up Hyperband search space
# tuner = kt.Hyperband(
#     hypermodel=lambda hp: build_model(hp, distilbert_encoder),  # Pass transformer here
#     objective='val_loss',
#     max_epochs=10,
#     directory='my_dir',
#     project_name='my_project'
# )

# Training setup with Hyperband
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Start Hyperparameter search
# tuner.search([input_ids], y_train, epochs=4, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Get the best model
# best_model = tuner.get_best_models(num_models=1)[0]
model = build_model(distilbert_encoder, max_len=128)

model.summary()

# Train the model (Increased batch size from 32 → 64)
history = model.fit(
    [input_ids], 
    y_train, 
    epochs=4, 
    batch_size=64,
    validation_split=0.2,  # Use 20% of the training data for validation
    callbacks=[early_stopping, reduce_lr]
)

# Predictions
y_pred_prob = model.predict([input_ids])
y_pred_labels = (y_pred_prob > 0.5).astype(int)

# Evaluate model
accuracy = accuracy_score(y_train, y_pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(y_train, y_pred_labels, average='weighted')

print("Classification Report:\n", classification_report(y_train, y_pred_labels))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot training history
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.show()