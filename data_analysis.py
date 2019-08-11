#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


# ### Features
# In[2]:
df = pd.read_csv('data/data.txt', sep=' ') # from TMall
df.columns

# In[3]:
df.shape

# In[4]:
df.head(10)
item_df = df.filter(regex=('item_*'))
user_df = df.filter(regex=('user_*'))
context_df = df.filter(regex=('context_*'))
shop_df = df.filter(regex=('shop_*'))
other_df = df.filter(regex=('^(?!item_)(?!user_)(?!context_)(?!shop_)'))
other_df.columns
# ### Change timestamp to date
# In[5]:
df['context_timestamp'].head(10)

# In[6]:
df['date'] = df['context_timestamp'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))


# In[7]:
df[['context_timestamp', 'date']].head()


# In[8]:
df['date'].unique()


# In[9]:
df['item_id'].unique().size


# In[10]:
df['item_brand_id'].unique().size


# In[11]:
df['item_price_level'].unique()


# In[12]:
# CVP: Conversion Rate on date
df.groupby('date').is_trade.mean().reset_index()


# In[13]:
# CVP on items, good feature
df.groupby('item_id').is_trade.mean().reset_index().head(10)


# In[14]:


oh_encoder = OneHotEncoder(sparse=True, categories='auto')
oh_encoder.fit_transform(df['user_gender_id'].values.reshape((-1, 1))).toarray()


# In[15]:


cv = CountVectorizer()
cv.fit_transform(df['item_category_list']).toarray()[0:10]


# In[16]:


space = oh_encoder.fit_transform(df['user_gender_id'].values.reshape((-1, 1))).toarray()
val = cv.fit_transform(df['item_category_list']).toarray()
np.hstack((space, val))[0:10]


# In[17]:


feature_importance_1 = eval(open('feature_importance_1.txt', 'r').read())
sorted(feature_importance_1.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


feature_importance_2 = eval(open('feature_importance_2.txt', 'r').read())
sorted(feature_importance_2.items(), key=lambda x: x[1], reverse=True)
