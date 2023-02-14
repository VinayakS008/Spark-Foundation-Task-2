#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Sparks Foundation
# 
# Data Science and Business Analytics Internship
# 
# Task 2 : Prediction using Unsupervised Machine Learning
# 
# Author : Sapkal Vinayak Dadaso
# 
# In this task for 'Iris' dataset we will be performimg the to predict the optimum number of clusters and represent it visually.
# 
# Clustering : "Clustering" is the process of grouping similar entities together. The goal of this unsupervised machine learning technique is to find similarities in the data point and group similar data points together.

# # Step 1 : Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from matplotlib.pylab import rcParams
rcParams ['figure.figsize']=(12,6)


# In[4]:


# Reading dataset
data=pd.read_csv("C:/Users/Vinayak/Desktop/VINAYAK/Project/data sets/iris dataset of grip.csv")
data.head()


# In[5]:


data.tail() # For first 5 rows


# In[6]:


data.shape #Dimension of dataset


# In[7]:


data.describe() # descriptive Statistics


# In[8]:


data.info() # Information about dataset


# # Step 2 : Exploratory Data Analysis

# In[10]:


new_data1 = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]]
print(new_data1.head())


# In[11]:


sns.pairplot(new_data1, hue="Species");


# In[ ]:





# In[12]:


sns.boxplot(x=data.Species,y=data.SepalLengthCm)
plt.show()


# In[20]:


#profileplot
a=data[["SepalLengthCm","SepalWidthCm"]].plot()


# In[ ]:





# # Step 2 : Model Preparation

# In[13]:


from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[14]:


# Dividing the dataset
x = data.iloc[:,:-1].values
y = data.iloc[:,4].values


# In[15]:


# Performing Elbow Method 
le = LabelEncoder()
y = le.fit_transform(y)


# In[16]:


error =[]
for i in range (1,7):
    model = KMeans(n_clusters=i)
    model.fit(x)
    error.append(model.inertia_)


# In[17]:


plt.plot(range(1,7),error)
plt.xlabel("Number of clusters")
plt.ylabel("Error rate")
plt.title("Elbow Method")


# In[21]:


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,7))
ax=plt.axes(projection = '3d')
ax.scatter3D(data['PetalLengthCm'],data['PetalWidthCm'],data['SepalLengthCm'])
plt.show


# # Step 3 : K- Means Clustering

# In[18]:


model = KMeans(n_clusters=3, max_iter=300)
model.fit(x)
y_pred = model.predict(x)


# In[19]:


plt.scatter(x[y_pred == 0, 0], x[y_pred==0,1], color="blue", label = 'Setosa')
plt.scatter(x[y_pred == 1, 0], x[y_pred==1,1], color="red", label = 'Versicolor')
plt.scatter(x[y_pred == 2, 0], x[y_pred==2,1], color="green", label = 'Virginica')

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color ="brown" , marker="*" , label = 'Centroids', s=100)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




