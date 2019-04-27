#!/usr/bin/env python
# coding: utf-8
# Authors: Mrinalini, Sahithya (NUS)

# In[1]:


#get_ipython().system('pip install paho-mqtt')


# In[2]:


#get_ipython().system('pip install -U scikit-learn')


# In[1]:


import pandas as pd
import numpy as np

##Benign = 0, Malicious = 1


# In[3]:


import pandas as pd
import numpy as np

sql = ("Select f1,f2,f3,f4,label from ml_train_data")
cursor.execute(sql)
result = cursor.fetchall()
results = list(result)
data = pd.DataFrame(results, columns=['f1', 'f2', 'f3', 'f4','Label'])
print(data)


# In[4]:


data.head()


# In[13]:


l,r = np.unique(data['Label'], return_counts=True)
# print(l, r)
## MACHINE LEARNING MODULE

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


train, test = train_test_split(data, test_size=0.3)

ytrain = train['Label']
train = train.drop(['Label'], axis=1)
Xtrain = train
ytest = test['Label']
test = test.drop(['Label'], axis=1)
Xtest = test

knn = SVC(kernel='linear')
knn.fit(Xtrain, ytrain)


# In[21]:


import paho.mqtt.client as paho
import pickle
broker="m16.cloudmqtt.com"
username = 'aabltfbh'
password = 'TDQJgjsEgMT8'
client = paho.Client("ML SVM")
client.username_pw_set(username, password)
client.connect(broker,port=15768)
client.publish("SVM1",pickle.dumps(knn),qos=0,retain=True)
print("published")


# In[14]:


y_pred=knn.predict(Xtest)


# In[15]:


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))


# In[16]:


from sklearn.metrics import f1_score
print("F1 score:",f1_score(ytest, y_pred))


# In[17]:


from sklearn.metrics import precision_score
print("Precision score:", precision_score(ytest, y_pred))


# In[18]:


from sklearn.metrics import recall_score
print("Recall score:", recall_score(ytest, y_pred))


# In[31]:


#print(knn.predict([[2, 11, 6, 8.5]]))


# In[ ]:





# In[ ]:



