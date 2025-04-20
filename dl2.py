#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# In[15]:


df = pd.read_csv(r"C:\Users\AKANSHA\Downloads\letter-recognition.csv")


# In[ ]:





# In[17]:


# Separate features and labels
X = df.iloc[:, 1:].values  # Features
y = df.iloc[:, 0].values   # Labels


# In[18]:


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # One-hot encoding


# In[19]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)


# In[21]:


model = Sequential([
    Dense(128, activation='relu', input_shape=(16,)),  # Input layer with 16 features
    Dense(64, activation='relu'),  # Hidden layer
    Dense(26, activation='softmax')  # Output layer for 26 letter classes
])



# In[22]:


# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[23]:


# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[24]:


# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# In[25]:


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





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv(r"C:\Users\AKANSHA\Downloads\letter-recognition.csv")


# In[3]:


df.head()


# In[4]:


x = df.iloc[:, 1:].values
y = df.iloc[:,0].values


# In[6]:


label = LabelEncoder()
y_encoded = label.fit_transform(y)
y_categorical = to_categorical(y_encoded)


# In[7]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_categorical, test_size=0.2)


# In[10]:


model = Sequential([
    Dense(126, activation='relu', input_shape=(16,)),
     Dense(64, activation='relu'),
     Dense(26, activation='softmax'),
])


# In[11]:


model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)


# In[12]:


history = model.fit(
        x_train, y_train,
        epochs = 5,
        batch_size = 32,
        validation_data=(x_test, y_test)
)


# In[13]:


loss, accuracy = model.evaluate(x_test, y_test)


# In[ ]:




