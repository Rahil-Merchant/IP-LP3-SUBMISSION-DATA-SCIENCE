#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from nltk import pos_tag 
import os,re, unicodedata
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import wordnet 
get_ipython().system('pip install tqdm')
from tqdm.notebook import tqdm
from tqdm import tqdm, tqdm_notebook


# In[3]:


df = pd.read_csv("data/fake-news-data.csv")
print(df.shape)
df.head()


# #### Cleaning Data

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
df.head()


# In[7]:


# Stopword list
pattern = re.compile(r'\b('+r'|'.join(stopwords.words('english'))+r')\b\s*')

# @cuda.jit(device=True)
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# @tf.function()
def clean_text(text):
    text = unicode_to_ascii(str(text).lower().strip())
    
    # creating a space between a word and the punctuation following it
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    
    # replacing all the stopwords
    text = pattern.sub('',text)
    
    # removes all the punctuations
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    
    text = text.strip()
    
    return text

clean_text_vect = np.vectorize(clean_text)


# In[8]:


def chunk_clean(array,chunk_size=64):
    cleaned_array = []
    
    for i in tqdm(range(0, len(array), chunk_size)):
        text_chunk = clean_text_vect(array[i:i+chunk_size])
        cleaned_array.extend(text_chunk)

    return np.array(cleaned_array)


# In[9]:


lema=wordnet.WordNetLemmatizer()

def text_normalization(text): 
    tokens=nltk.word_tokenize(text)     
    tags_list=pos_tag(tokens,tagset=None) 

    lema_words=[] 
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val='v'
        elif pos_token.startswith('J'): # Adjective
            pos_val='a'
        elif pos_token.startswith('R'): # Adverb
            pos_val='r'
        else:
            pos_val='n' # Noun
            
        lema_token=lema.lemmatize(token,pos_val) 
        lema_words.append(lema_token) 
    
    return " ".join(lema_words)

text_norm_vect = np.vectorize(text_normalization)


# In[10]:


def chunk_text_normalize(array,chunk_size=64):
    norm_array = []
    
    for i in tqdm(range(0, len(array), chunk_size)):
        text_chunk = text_norm_vect(array[i:i+chunk_size])
        norm_array.extend(text_chunk)
    
    return np.array(norm_array)    


# In[11]:


def truncate(text):
    text = text.split()
    text = text[0:170]
    
    return ' '.join(text)


# In[12]:


df['truncated_text'] = df['merged'].apply(truncate)
df.head()


# In[13]:


# # df['cleaned_data'] = chunk_clean(df.merged.values)

# cleaned_data = chunk_clean(df.merged.values)
# norm_data = chunk_text_normalize(cleaned_data)

df['cleaned_data'] = df['truncated_text'].apply(clean_text)
df['norm_data'] = df['cleaned_data'].apply(text_normalization)
df.head()


# #### Splitting Data

# In[14]:


x_train,x_test,y_train,y_test=train_test_split(df['cleaned_data'], df['label'], test_size=0.025, random_state=1)


# #### TFIDF-Vectorizer

# In[15]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7, use_idf=True)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[ ]:


tfidf_vectorizer.get_feature_names()


# #### Passive Aggressive Classifier

# In[16]:


pac=PassiveAggressiveClassifier(early_stopping=True,validation_fraction=.025,verbose=1,shuffle=True,random_state=1)
pac.fit(tfidf_train,y_train)
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# #### Confusion Matrix

# In[17]:


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


# In[18]:


pred = pac.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=[0,1])
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


# **Normalized Data**

# In[20]:


x_train1,x_test1,y_train1,y_test1=train_test_split(df['norm_data'], df['label'], test_size=0.025, random_state=1)
tfidf_vectorizer1=TfidfVectorizer(stop_words='english', max_df=0.7, use_idf=True)
tfidf_train1=tfidf_vectorizer.fit_transform(x_train1) 
tfidf_test1=tfidf_vectorizer.transform(x_test1)
pac1=PassiveAggressiveClassifier(early_stopping=True,validation_fraction=.025,verbose=1,shuffle=True,random_state=1)
pac1.fit(tfidf_train1,y_train1)
y_pred1=pac1.predict(tfidf_test1)
score1=accuracy_score(y_test1,y_pred1)
print(f'Accuracy: {round(score1*100,2)}%')


# In[22]:


pred1 = pac1.predict(tfidf_test1)
score1 = metrics.accuracy_score(y_test1, pred1)
print("accuracy:   %0.3f" % score1)
cm = metrics.confusion_matrix(y_test1, pred1, labels=[0,1])
plot_confusion_matrix(cm, classes=['Fake', 'Real'])

