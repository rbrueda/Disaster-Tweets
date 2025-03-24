import numpy as np
import matplotlib.pyplot as plt

# Data
models = ['FastText + MultiHeadAttention + CNN', 'FastText + BiLSTM', 'BERT + CNN', 'RoBERTa']
times = [16.41, 112.55, 1889.41, 1958.19]

# Bar width and positions
x = np.arange(len(models))

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars = ax.bar(x, times, color=['blue', 'orange', 'green', 'red'])

# Add time values on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 50, f'{height:.2f}', ha='center', va='bottom')

# Labels, title, and formatting
ax.set_xlabel('Models')
ax.set_ylabel('Time Taken (Seconds)')
ax.set_title('Comparison of Model Execution Time')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')

# Display the graph
plt.show()