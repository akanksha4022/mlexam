#!/usr/bin/env python
# coding: utf-8

# #### 6. Develop a program to forecast future values in time series data, such as weather patterns, using RNN models like LSTM or GRU.
# (Using daily-minimum-temperature data)

# In[12]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[13]:


df = pd.read_csv('daily-min-temperatures.csv')
df.head()


# In[14]:


data = df['Temp'].values.reshape(-1, 1)
data.shape


# In[15]:


scaler = MinMaxScaler(feature_range=(0, 1)) #MinMaxScaler is a normalization technique from the sklearn.preprocessing module.

                                            #It scales your data so that all values fall within a specific range, typically between 0 and 1.

data_norm = scaler.fit_transform(data)

data_norm = data_norm.flatten() #flatten() converts it back into a 1D array:

data_norm.shape


# In[16]:


X, y = [], []
#X → 2D list of input sequences (each of length 10)

#y → list of target values (each is the value immediately after each sequence)

for i in range(len(data_norm) - 10):
    X.append(data_norm[i:i+10])
    y.append(data_norm[i+10])


# In[17]:


X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)


# In[18]:


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[19]:


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(1)
])


# In[20]:


model.summary()


# In[21]:


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[22]:


history = model.fit(
    X, y,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)


# In[23]:


test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Mean Absolute Error: {test_mae}")


# In[24]:


predictions = model.predict(X_test)

predicted_data = scaler.inverse_transform(predictions)
actual_data = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_data, label='Actual Temperature')
plt.plot(predicted_data, label='Predicted Temperature', color='red')
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.legend()
plt.show()


# **Future Predictions**

# In[25]:


preds = []
seq = X[-1]

for _ in range(10):
    pred = model.predict(seq.reshape(1, 10, 1), verbose=0)[0, 0]
    preds.append(pred)
    seq = np.roll(seq, -1) # shift sequence
    seq[-1] = pred # append predicted value
    
predicted_data = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# In[29]:


plt.figure(figsize=(10, 5))
plt.plot(df['Temp'], label='Actual')
plt.plot(range(len(df['Temp']), len(df['Temp']) + 10), predicted_data, 'ro-', label='Future Predictions')
plt.legend()
plt.title('Temperature Forecast for the Next 10 Days')
plt.show()


# In[ ]:





# In[ ]:




