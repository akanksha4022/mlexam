#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# In[5]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[6]:


x_train  = x_train.astype('float32')/255.0
x_test  = x_test.astype('float32')/255.0


# In[8]:


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# In[9]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[11]:


model = Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape = (28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3), activation='relu' ),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3), activation='relu' ),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')    
    
])


# In[12]:


model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


# In[14]:


model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))


# In[15]:


loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy:{accuracy*100:.2f}")


# In[ ]:





# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0


# In[4]:


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# In[5]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[6]:


model = Sequential([
    Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])


# In[ ]:


model.compile(
)

