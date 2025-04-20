#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# In[2]:


(x_train, y_train),(x_test, y_test) = mnist.load_data()


# In[3]:


x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0


# In[4]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[9]:


model = Sequential([
    Flatten(input_shape = (28,28)),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(10, activation = 'softmax')
])


# In[10]:


model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


# In[11]:


history = model.fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 32,
    validation_data = (x_test,y_test)    
)


# In[12]:


loss,accuracy = model.evaluate(x_test, y_test)
print(f"test loss:{loss:.4f}")
print(f"test accuracy:{accuracy:.4f}")


# In[1]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')


# In[ ]:





# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


x_train = x_train.astype('float32')/255.0
x_train = x_train.astype('float32')/255.0


# In[4]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[5]:


model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])


# In[9]:


model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']    
              
)


# In[10]:


history = model.fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 32,
    validation_data = (x_test, y_test)
)


# In[11]:


loss, accuracy = model.evaluate(x_test, y_test)


# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label = "accuracy")
plt.plot(history.history['val_accuracy'], label = "accuracy")


# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label = "accuracy")
plt.plot(history.history['val_loss'], label = "accuracy")


# In[ ]:




