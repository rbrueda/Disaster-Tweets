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
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

tf.config.optimizer.set_jit(True)
                    
# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

#keras does not have a built in f1 score -> manually make function
def f1_score(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float16")  # Ensure matching dtype
    y_pred = K.round(y_pred)  # Convert probabilities to 0 or 1
    tp = K.sum(y_true * y_pred)
    precision = tp / (K.sum(y_pred) + K.epsilon())  # Avoid division by zero
    recall = tp / (K.sum(y_true) + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


#other optimiation that can be made -> reducing padding, fewer unnecessary computations are performed, reducing training and inference time
    # lower memory usage -> padding increases the size of input tensors, leading to higher RAM/VRAM usage
    # optimizing padding -> helps fit larger batch sizes in memory
    # too much padding -> deads to poor generalization

#todo: try RMSProp -> has fast convergence, and requires fewer hyperparameter tuning in comparison to AdaGrad and Adam

#todo: try adaptive batch-wise padding -> find optimal amount to reduce removing critical content
#todo: use a pre-trained distilbert model (if this is not that optimal -> try BERT)
#todo: use KerasTuner hyperparameter optimzation framework -> problem is it uses both bayesian and RANDOM search which can slow done the process
#todo: try Hyperband (available in kerastuner)
#    - apparently faster than Bayesian optimization 
#see if 6 layers is good

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Load DistilBERT (Freeze Parameters) -- use pretrained model
distilbert_encoder = TFDistilBertModel.from_pretrained("distilbert-base-uncased", trainable=False)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
lemmatizer = WordNetLemmatizer()

# Custom Transformer Layer for Compatibility
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, transformer, **kwargs):
        super().__init__(**kwargs)
        self.transformer = transformer

    def call(self, inputs):
        input_ids, attention_mask = inputs
        transformer_output = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            training=False  # Prevent DistilBERT from training
        )
        return transformer_output.last_hidden_state  # Use contextual embeddings

# Function to build optimized model
#num_classes = 2 -> 0 - non-disaster tweets, 1 - disaster tweets

def build_model(transformer, max_len=128, num_classes=2):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # Transformer Layer (DistilBERT)
    transformer_layer = TransformerLayer(transformer)
    transformer_output = transformer_layer([input_ids, attention_mask])  # (None, 128, 768)

    # CNN Feature Extraction
    conv1D = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(transformer_output)
    pooling = GlobalAveragePooling1D()(conv1D)

    # Fully Connected Layers with L2 Regularization
    fc = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(pooling)
    
    # Dropout Layer for Regularization -- reguce overfitting
    dropout = Dropout(0.5)(fc)  # 50% dropout to prevent overfitting

    sigmoid = Dense(1, activation="sigmoid")(dropout)  

    # Compile Model with RMSprop and Binary Cross-Entropy Loss
    model = Model(inputs=[input_ids, attention_mask], outputs=sigmoid)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                  loss="binary_crossentropy", metrics=['accuracy', 'Precision', 'Recall', 'AUC', f1_score])

    return model

#generalizing words with same synonym
def synonym_replacement(word):
    synonyms = wordnet.synsets(word)
    if synonyms:
        return synonyms[0].lemmas()[0].name()  # Return first synonym
    return word

# Preprocessing Function
#todo: see if lemmanization is needed
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Replace random words with synonyms (data augmentation)
    tokens = [synonym_replacement(word) if np.random.rand() < 0.1 else word for word in tokens]

    return " ".join(tokens)

# Load and preprocess dataset
df = pd.read_csv('train.csv')
df['clean_text'] = df['text'].apply(clean_text)
df.dropna(subset=['text'], inplace=True)

# Tokenization (Reduced max_length from 512 → 128)
tokens = tokenizer(
    list(df['clean_text']), 
    padding='max_length', 
    truncation=True, 
    max_length=128,  
    return_tensors="tf"
)

# Convert labels to categorical format
num_classes = df['target'].nunique()
y_train = df['target'].values  # Keep as a single-column binary vector


# Extract input_ids and attention_mask
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']


# Build and Train Model
model = build_model(distilbert_encoder, max_len=128, num_classes=num_classes)

# Print Model Summary
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model (Increased batch size from 32 → 64)
history = model.fit(
    [input_ids, attention_mask], 
    y_train, 
    epochs=4, 
    batch_size=64,
    validation_split=0.2,  # Use 20% of the training data for validation
    callbacks=[early_stopping, reduce_lr]
)
# Predicting probabilities
y_pred_prob = model.predict([input_ids, attention_mask])

# Convert predictions from probabilities to labels (0 or 1)
y_pred_labels = (y_pred_prob > 0.5).astype(int)

# Convert true labels to class labels
y_true = np.argmax(y_train, axis=1)  # True labels (convert one-hot to class indices)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_true, y_pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='weighted')

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred_labels))

# Print individual metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot training accuracy
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.show()
