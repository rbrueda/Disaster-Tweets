# Disaster-Tweets
Project for COMP-4740

Model created: train-model.py

## Run model
```bash
python3 train-model.py
```

- make sure to download all the libraries listed

## Preprocessing
- lemmatization
- lowercase all words
- stopword and special character removal
- tokenized words and converted to sequences of indices

## Model Features
- CNN + MultiAttention Layer + FastText word embeddings
- Added hybrid pooling - Average and Max Pooling
- L2 regularization + Dropouts between layers
- Applied learning rate scheduler and early stopping for preventing excessive overfitting

## Model Metrics
- Accuracy
- Precison
- Recall
- F1 Score
  
## Research Paper
[Analysis of Deep Learning Model for Disaster
Tweet Classification Speed Optimization Paper](https://drive.google.com/file/d/11BR-lSOaI388ehZN8wHtaO0kDUyqG1NU/view?usp=drive_link)
