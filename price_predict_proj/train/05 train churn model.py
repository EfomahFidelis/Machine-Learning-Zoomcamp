#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


# In[3]:


from collections import Counter
import random


# In[4]:


df = pd.read_csv('../03_churn_prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    
# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[5]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']


# In[6]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender', 
    'seniorcitizen', 
    'partner', 
    'dependents',
    'phoneservice', 
    'multiplelines', 
    'internetservice',
    'onlinesecurity', 
    'onlinebackup', 
    'deviceprotection',
    'techsupport', 
    'streamingtv', 
    'streamingmovies',
    'contract', 
    'paperlessbilling', 
    'paymentmethod'
]


# In[7]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    # model = LogisticRegression(solver='liblinear', random_state=1)
    # model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


# In[8]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[9]:


C = 1.0
n_splits = 5


# In[10]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    
    print('C=%s, %0.3f ± %0.3f' % (C, np.mean(scores), np.std(scores)))


# In[11]:


scores


# In[12]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# #### Save the model

# In[14]:


import pickle


# In[15]:


output_file = f'model_C={C}.bin'
output_file


# In[16]:


# open a file write to it.
f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# In[17]:


# open and write to file using with statement
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# ### Load the model

# In[19]:


import pickle


# In[20]:


model_file = 'model_C=1.0.bin'


# In[21]:


# open file for reading using with statement
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[22]:


dv, model


# In[64]:


customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}


# In[66]:


# turn customer to feature matrice
X = dv.transform([customer])


# In[68]:


# get row of column 1 from result prediction
model.predict_proba(X)[0, 1]


# In[ ]:





# In[ ]:





# In[ ]:




