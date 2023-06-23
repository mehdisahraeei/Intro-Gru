import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[6]:


df = pd.read_csv('BTC-USD.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)
df.head(5)


# - dropping

# In[7]:


df = df.drop(['High','Low','Adj Close','Volume','Close'], axis=1)
df.head()


# In[8]:


dataset = df.values
dataset = dataset.astype('float64')
dataset[:5]


# ### plotting data

# In[9]:


plt.plot(dataset)
plt.show()


# ### normalizing data

# In[10]:


scaler = MinMaxScaler(feature_range = (0,1))
dataset = scaler.fit_transform(dataset)
print(dataset[:5],'\n')
print(dataset.shape)


# ### Method for making data and timestep

# In[11]:


# lookback -> timestep
def create_dataset(dataset,look_back):
    data_x, data_y = [],[] #data_x is data and data_y is label
    for i in range(len(dataset)-look_back-1): #we want if data be beyond len(sequendatasetce), the command will not continue
        data_x.append(dataset[i:(i+look_back),0])
        data_y.append(dataset[i+look_back,0])
    return np.array(data_x) , np.array(data_y)


# ### split dataset

# In[12]:


train_size = int(len(dataset) * 0.90)
train , test = dataset[0:train_size,:] , dataset[train_size:len(dataset),:]
print(train.shape)
print(test.shape)


# In[13]:


train[:3]


# ### Making data train & test x,y

# In[14]:


n_steps = 5 #timestep or look_up

train_x , train_y = create_dataset(train, n_steps)
test_x , test_y = create_dataset(test, n_steps)


# In[15]:


print(train_x.shape , train_y.shape)
print(test_x.shape , test_y.shape)


# In[16]:


train_y[:3]


# In[17]:


for i in range(len(test_x)):   
    print(test_x[i],test_y[i])

# print(train_x[:5],'\n')
# print(train_y[:5])


# ### Converting data to three-dimensional or three-channel
# - The input must be three-dimensional or three-channel, that's why we reshape it
# 

# In[18]:


trainxr = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
testxr = np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))

print(trainxr.shape)
print(testxr.shape)


# In[19]:


train_x = trainxr
test_x = testxr


# ### training model

# In[43]:


n_features = 1 
model = Sequential()
model.add(GRU(30, activation = 'tanh', input_shape=(n_steps, n_features)))  
model.add(Dense(1)) #n-output  
model.compile(optimizer='RMSprop',loss='mse',metrics=['accuracy'])   
# model.summary()


# In[50]:


#verbose is the choice that how you want to see the output of your Nural Network while it's training. If you set verbose = 0, It will show nothing
model.fit(train_x, train_y, epochs=400, shuffle=False, batch_size=2)


# ### saving model

# In[25]:


model.save('/content/drive/MyDrive/savedata')


# In[26]:


# loading model
new_model = tf.keras.models.load_model('/content/drive/MyDrive/savedata')


# ### Testing model

# In[52]:


predict_train = model.predict(train_x)
predict_test = model.predict(test_x)
print('predicted y(train):', np.reshape(predict_train[:5],-1))
print('real y(train):', train_y[:5])


# In[53]:


predict_train = scaler.inverse_transform(predict_train)
trainy = scaler.inverse_transform([train_y])

predict_test = scaler.inverse_transform(predict_test)
testy = scaler.inverse_transform([test_y])


# In[54]:


print(predict_train[:5])


# ### creating df and plotting

# In[57]:


Answer1 = pd.DataFrame({
    "Predicted": predict_train.ravel(),
    "real": trainy.ravel()
}) 
Answer1.head()


# In[58]:


#train
Answer1.plot(title="outcome", figsize=(10,5));


# In[126]:


# xx2 = test_x.reshape(-1,1)
# xx2 = scaler.inverse_transform(xx2)
# xx2[:8]


# In[59]:


Answer2 = pd.DataFrame({
    "Predicted": predict_test.ravel(),
    "real": testy.ravel()
}) 
Answer2.head()


# In[60]:


#test
Answer2.plot(title="outcome", figsize=(10,5));


# ### Evaluate the model

# In[61]:


model.evaluate(test_x, test_y)


# In[62]:


train_score = math.sqrt(mean_squared_error(trainy.reshape(-1),predict_train))
print('rmse ', train_score)
test_score = math.sqrt(mean_squared_error(test_y.reshape(-1),predict_test))
print('rmse ', test_score)


# ### new testing
# - we must change dim

# In[63]:


test_x.shape , test_y.shape


# In[64]:


test_ = array([0.0272989,  0.03199121, 0.03083671, 0.02082067, 0.03542986])
test_ = test_.reshape(1,n_steps,n_features)
test_.shape


# In[65]:


y_hat = model.predict(test_)
y_hat


# In[66]:


y_hat = scaler.inverse_transform(y_hat)
y_hat

