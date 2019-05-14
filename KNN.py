#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


print(os.listdir())


# In[3]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#Laoding dataSet
dataSet = pd.read_csv('Social_Network_Ads.csv')


# In[6]:


#Checking the DataSet
dataSet


# In[7]:


# getting information from the dataSet
dataSet.info()


# In[8]:


# split the dataSet
X = dataSet.iloc[:,2:4].values
y = dataSet.iloc[:,4].values


# In[9]:


X


# In[10]:


y


# In[11]:


# prepare data for test and training
from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[13]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier


# In[15]:


classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski',)
classifier.fit(X_train, y_train)


# In[16]:


# get your predictor
y_predictor = classifier.predict(X_test)


# In[17]:


y_predictor


# In[18]:


from sklearn.metrics import confusion_matrix


# In[19]:


cm = confusion_matrix(y_test, y_predictor)


# # Plotting a contour graph

# In[20]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_point, y_point = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_point[:, 0].min() - 1, stop = X_point[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_point[:, 1].min() - 1, stop = X_point[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_point)):
    plt.scatter(X_point[y_point == j, 0], X_point[y_point == j, 1],
                c = ListedColormap(('green', 'blue'))(i), label = j)
plt.title('K-NN Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()


# In[21]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_point, y_point = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_point[:, 0].min() - 1, stop = X_point[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_point[:, 1].min() - 1, stop = X_point[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_point)):
    plt.scatter(X_point[y_point == j, 0], X_point[y_point == j, 1],
                c = ListedColormap(('green', 'blue'))(i), label = j)
plt.title('K-NN Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()

