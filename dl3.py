#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling1D, Flatten, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# In[3]:


max_words = 10000
max_len = 200


# In[4]:


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)


# In[5]:


x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)


# In[6]:


model = Sequential([
    Embedding(input_dim = max_words, output_dim = 128, input_length = max_len ),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[7]:


model.compile(
    optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy']
)


# In[8]:


history = model.fit(x_train, y_train, 
          epochs=5, 
          batch_size = 64, 
          validation_data = (x_test, y_test))


# In[9]:


loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy:{accuracy*100:.2f}")


# In[10]:


plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




