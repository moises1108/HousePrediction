#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Import dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[21]:


# Load the Boston Housing Data set form sklearn.datasets and print it
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston)


# In[23]:


# Transform the data set into a data frame
# data = the date we want/independent variables/X value
# feature_names = the column names of the data
# target = the target variable/price of houses/dependent variable/ Y value

df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)


# In[24]:


# Initialise the linear regresseion model
reg = linear_model.LinearRegression()


# In[26]:


# Split the data into 67% training and 33% testing data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.33, random_state = 42)


# In[27]:


# Train Model with our trainin data
reg.fit(x_train, y_train)


# In[35]:


# Print the predictions on our test data
y_pred = reg.predict(x_test)
print(y_pred)

# Actual values
# print(y_test)


# In[ ]:


# Check the model performance / accuracy is

