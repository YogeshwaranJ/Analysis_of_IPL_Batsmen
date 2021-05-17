#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
df = pd.read_csv('/home/yogesh/Downloads/most_runs_average_strikerate.csv')
df.describe()


# In[ ]:





# In[23]:


import matplotlib.pyplot as plt
plt.figure()
plt.scatter(df['average'],df['strikerate'])
plt.xlabel('Average')
plt.ylabel('Strikerate of batsman')


# In[49]:


missing_val_count_by_column = (imputed_x.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[3]:


features =['average','strikerate']
x = df[features]


# In[25]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputed_x= pd.DataFrame(imputer.fit_transform(x))
imputed_x.columns = x.columns


# In[5]:


from sklearn.cluster import KMeans
wss=[]
for i in range (1,11):
    kmeanscluster = KMeans(n_clusters = i, init = 'k-means++')
    kmeanscluster.fit(imputed_x)
    kmeanscluster.inertia_
    wss.append(kmeanscluster.inertia_)


# In[6]:


plt.plot(range(1,11), wss)


# In[50]:


kmeanscluster = KMeans(n_clusters= 4, init = 'k-means++')
y=kmeanscluster.fit_predict(imputed_x)


# In[52]:



centers = kmeanscluster.cluster_centers_
print(centers)


# In[9]:


df['Cluster'] = y


# In[11]:


batsman_cluster_0=df[df['Cluster']==0]['batsman'].tolist()


# In[12]:


batsman_cluster_2=df[df['Cluster']==2]['batsman'].tolist()


# In[13]:


batsman_cluster_1=df[df['Cluster']==1]['batsman'].tolist()


# In[ ]:


batsman_cluster_3=df[df['Cluster']==3]['batsman'].tolist()


# In[21]:


batsman_cluster_0


# In[22]:


batsman_cluster_1


# In[23]:


batsman_cluster_2


# In[25]:


type(imputed_x)


# In[14]:


imputed_x['Cluster'] = y


# In[35]:


imputed_x.(50)


# In[53]:


plt.scatter(df['average'], df['strikerate'],c=y,s =50,cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5);
plt.xlabel('Average of batsmen')
plt.ylabel('Strikerate off batsmen')


# In[11]:


centers


# In[25]:


import scipy.cluster.hierarchy as sch


# In[26]:


dendrogram = sch.dendrogram(sch.linkage(imputed_x, method='ward'))


# In[12]:


from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y1 = model.fit_predict(imputed_x)


# In[50]:


y1


# In[51]:


df['Cluster_ag'] = y1


# In[13]:


plt.scatter(df['average'], df['strikerate'],c=y1,s =50,cmap='viridis')


# In[44]:


centers_ag = model.cluster_centers_


# In[14]:


data1_cluster = df[df['Cluster']==1]


# In[32]:


data1_cluster[data1_cluster['total_runs']>500]


# In[31]:


data1_cluster[data1_cluster['total_runs']<500].head(50)


# In[16]:


data2_cluster = df[df['Cluster']==2]


# In[34]:


data2_cluster[data2_cluster['numberofballs']>50]


# In[70]:


data2_cluster[data2_cluster['total_runs']<200]


# In[18]:


data0_cluster = df[df['Cluster']==0]


# In[35]:


data0_cluster


# In[75]:


data0_cluster[data0_cluster['total_runs']<200]


# In[76]:


data0_cluster[data0_cluster['total_runs']>200]


# In[36]:


data3_cluster = df[df['Cluster']==3]


# In[37]:


data3_cluster.head(50)


# In[44]:


import seaborn as sns
dat = df.iloc[:,1:6]


# In[45]:


sns.heatmap(dat.corr())


# In[ ]:




