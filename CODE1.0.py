#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Data collection and analysis

# In[6]:


customer_data=pd.read_csv("Book1.csv")


# In[7]:


customer_data.head()


# In[8]:


customer_data.isnull()


# In[10]:


x=customer_data.iloc[:,[3,4]].values
print(x)


# CHOOSING NO OF CLUSTERS
# 

# WCSS-> WITHIN CLUSTERS SUM OF SQUARES

# In[13]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[14]:


#plot an elbow graph
sns.set()
plt.plot(range(1,11),wcss)
plt.title("the Elbow Point graph")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show


# THE OPTIMUM NO OF CLUSTERS IS 5

# TRAINING K-MEANS CLUSTERING MODEL

# In[17]:


kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y=kmeans.fit_predict(x)
print(y)


# visualizing clusters

# In[18]:


#plotting all clusters


# In[24]:


plt.figure(figsize=(9,9))
plt.scatter(x[y==0,0],x[y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(x[y==1,0],x[y==1,1],s=50,c='pink',label='Cluster 2')
plt.scatter(x[y==2,0],x[y==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(x[y==3,0],x[y==3,1],s=50,c='red',label='Cluster 4')
plt.scatter(x[y==4,0],x[y==4,1],s=50,c='cyan',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='Centroids')
plt.title("customer Groups")
plt.xlabel("Annual income")
plt.ylabel("Spending score")

As we can see in the following visualization we can provide offers in blue and green circles since they have low and high income respectively via offers we can increase thier spending score thereby inceasing the profits of the company
# In[ ]:




