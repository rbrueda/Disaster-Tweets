import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage if necessary (for CPU)

# Load nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions & hashtags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('train.csv')
df.dropna(subset=['text'], inplace=True)
df['target'] = df['target'].astype(int)

# Apply text preprocessing
df['text'] = df['text'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# Load tokenizer (MATCHING MODEL AND TOKENIZER)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_texts(texts, tokenizer, max_length=128):
    encodings = tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return encodings['input_ids'], encodings['attention_mask']

# DataLoader preparation
def prepare_dataloader(X, y, tokenizer, batch_size=32):
    input_ids, attention_mask = tokenize_texts(X, tokenizer)
    labels = torch.tensor(y.values, dtype=torch.float32)  # Ensure correct dtype for BCE loss
    dataset = TensorDataset(input_ids.to(torch.long), attention_mask.to(torch.long), labels)  # Convert to long
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dataloader = prepare_dataloader(X_train, y_train, bert_tokenizer)
val_dataloader = prepare_dataloader(X_test, y_test, bert_tokenizer)

# Define BERT + CNN Model
class BERT_CNN_Model(nn.Module):
    def __init__(self):
        super(BERT_CNN_Model, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.conv1 = nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Freeze BERT to use as feature extractor
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = bert_outputs.last_hidden_state.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_length)
        conv_out = torch.relu(self.conv1(hidden_states))
        conv_out = torch.relu(self.conv2(conv_out))
        pooled_output = self.global_max_pool(conv_out).squeeze(2)  # (batch_size, 64)
        logits = self.fc(pooled_output)
        return logits

# Model training function with early stopping
def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=10, patience=3):
    # Start the timer
    start_time = time.time()
    model.to(device)  # Move model to GPU/CPU
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()  # Set to training mode
        total_loss = 0

        for batch in train_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]  # Move to device
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.squeeze(), labels)  # BCE Loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        y_pred, y_true = [], []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.squeeze(), labels)
                val_loss += loss.item()

                # Compute predictions
                preds = torch.round(torch.sigmoid(outputs.squeeze()))  # Sigmoid for binary classification
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Validation loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total Execution Time: {elapsed_time:.2f} seconds")

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    return metrics

# Initialize and train the model
model = BERT_CNN_Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.BCEWithLogitsLoss()  # Proper loss for binary classification

print('Training BERT_CNN...')
metrics = train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn)
print("Model Metrics:", metrics)
