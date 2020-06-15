#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd


# In[5]:


# data_path1 = '/kaggle/input/fake-news-cc/'
data_cc = pd.read_csv('data/news.csv')
# data_path2 = '/kaggle/input/fake-and-real-news-dataset/'
data2_fake=pd.read_csv('data/Fake.csv')
data2_real=pd.read_csv('data/True.csv')


# In[3]:


data_cc.head()


# In[4]:


data2_fake.head()


# In[5]:


data2_real.head(80)


# In[6]:


data2_real['label']=1
data2_fake['label']=0


# In[7]:


data2=pd.concat([data2_real,data2_fake])


# In[8]:


print(data2.shape)
data2.head()


# In[9]:


data_cc.drop('Unnamed: 0', inplace = True, axis = 1)


# In[10]:


data2.drop(['subject','date'],inplace=True,axis=1)


# In[11]:


data2.head()


# In[12]:


data_cc=data_cc.replace(to_replace ="FAKE", value =0) 
data_cc=data_cc.replace(to_replace ="REAL", value =1) 


# In[13]:


data_cc.tail()


# In[14]:


data_cc.head()


# In[15]:


data2.head()


# In[16]:


data=pd.concat([data_cc,data2])


# In[17]:


print(data.shape)
data.head()


# In[18]:


data = data.dropna()
data = data.reset_index(drop=True)
print(data.shape)
data.head()


# In[19]:


data['label'].value_counts()


# In[20]:


data.to_csv('fake-news-data.csv', index=False)

