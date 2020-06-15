#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
import sklearn.metrics as metrics
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
import os,re, unicodedata
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import wordnet 
get_ipython().system('pip install tqdm')
from tqdm import tqdm_notebook


# In[2]:


df = pd.read_csv("data/fake-news-data.csv")
print(df.shape)
df.head()


# #### Cleaning Data

# In[3]:


# df.drop('Unnamed: 0', inplace = True, axis = 1)


# In[4]:


df=df.replace(to_replace ="FAKE", value =0) 
df=df.replace(to_replace ="REAL", value =1) 


# In[5]:


df.dropna()
df = df.reset_index(drop=True)
print(df.shape)
df.head()


# In[6]:


df["merged"]=df["title"] + " " + df["text"]


# #### Splitting Data

# In[7]:


x_train,x_test,y_train,y_test=train_test_split(df['merged'], df['label'], test_size=0.25, random_state=1)


# #### TFIDF-Vectorizer

# In[11]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7, use_idf=True)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[12]:


tfidf_vectorizer.get_feature_names()


# #### Passive Aggressive Classifier

# In[13]:


pac=PassiveAggressiveClassifier(early_stopping=True,validation_fraction=.15,verbose=1,shuffle=True,random_state=1)
pac.fit(tfidf_train,y_train)
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# #### Confusion Matrix

# In[14]:


def plot_confusion_matrix(cm, classes,normalize=False,cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.cmap=cmap
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[15]:


pred = pac.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=[0,1])
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


# In[ ]:




