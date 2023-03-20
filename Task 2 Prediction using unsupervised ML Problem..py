#!/usr/bin/env python
# coding: utf-8

# # The sparks Foundation: Data Science and Business Analytics Internship
# # Task 2: Prediction using unsupervised ML Problem
# # From the given iris dataset predict the optimum number of clusters and represent it visually
# # Author : Raut Nikhil Santosh

# # Step 1: Importing the Dataset

# In[3]:


# Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[4]:


# load the data
data=pd.read_excel("C:\\Users\\Rohit Bhosale\\Desktop\\Data II.xlsx")
# To show the first five columns of data set
data.head()


# In[5]:


# Removing the 'Id' column
data.drop('Id' , axis=1, inplace=True) 
data.head()


# # Step 2 : Exploring the data

# In[6]:


# no of rows & columns
data.shape


# In[7]:


# Summary of Statistics
data.describe()


# In[8]:


# To check the if null values are Present
data.info()


# In[9]:


data.Species.value_counts()


# # Step 3 : Using the Elow Method to find the Optimal numbers of clusters

# In[10]:


# Find the optimal number of clusters for k means clustering
x=data.iloc[:,:-1].values
from sklearn.cluster import KMeans
WCSS=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',
                 max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11),WCSS)
plt.title('The ELOW Method')
plt.xlabel('WCSS')
plt.show()


# In[11]:


# Apply means to the dataset
kmeans=KMeans(n_clusters=3,
             max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans


# # Step 5 : Visualize the test set result

# In[12]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],
           s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],
           s=100,c='blue',label='Iris-versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],
           s=100,c='green',label='Iris-virginica')

# Plotting the centorids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
           s=100,c='yellow',label='Centroids')
plt.legend()


# # Data Visualization

# In[13]:


data.corr()


# In[14]:


plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True,cmap='ocean')


# In[15]:


plt.figure(figsize=(8,6))
sns.boxplot(x="Species",y="SepalLengthCm",data=data)


# In[16]:


sns.pairplot(data.corr())


# # Thank You!

# In[ ]:





# In[ ]:




