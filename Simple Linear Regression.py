#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


weekly_sales = {"Weekly Advert Exp":[41,54,63,54,48,56,62,61,64,71],
                "Weekly Sales":[1250,1380,1425,1425,1450,1300,1400,1510,1575,1650]}


# In[3]:


datafr =pd.DataFrame(weekly_sales)
datafr


# In[4]:


sns.regplot(x="Weekly Advert Exp", y="Weekly Sales", data=datafr);


# In[22]:


y=datafr['Weekly Sales'].values.reshape(10,1)
y


# In[23]:


x=datafr['Weekly Advert Exp'].values.reshape(10,1)
x


# In[9]:


# create a LinearRegression object
lm = LinearRegression()


# In[10]:


lm.fit(x,y)


# In[12]:


print(lm.intercept_)


# In[13]:


print(lm.coef_)


# In[24]:


lm.score(x,y)


# In[ ]:




