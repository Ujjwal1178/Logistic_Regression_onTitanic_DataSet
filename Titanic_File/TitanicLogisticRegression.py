#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[8]:


df = pd.read_csv("D:/titanic.csv")


# In[9]:


df = df.dropna()


# In[10]:





# In[26]:


y=df["Survived"]


# In[27]:


df.head


# In[34]:


y


# In[29]:


y


# In[30]:


import matplotlib.pyplot as plt


# In[31]:


x=df[["Age","Fare"]]


# In[32]:


x


# In[33]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5)


# In[38]:


from sklearn.linear_model import LogisticRegression


# In[41]:


model=LogisticRegression()
#x_train=x_train.value.reshape()
model.fit(x_train,y_train)


# In[45]:


from sklearn.metrics import accuracy_score


# In[52]:


y_predict=model.predict(x_test)
print(accuracy_score(y_predict,y_test))


# In[53]:


df.describe()


# In[56]:


a=np.array([45,600]).reshape(-1,2)


# In[57]:


model.predict(a)


# In[ ]:




