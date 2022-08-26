from posixpath import split
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from keras.models import Sequential 
from keras.layers import Dense


#read data in csv file 
bc = pd.read_csv("Breast_Cancer.csv")
#print(bc.head())



print(bc.info())
bc = bc.drop('Race',axis = 1)

# label encoder 
obj_list= [a for a in bc.columns if bc[a].dtype == object ]
num_list = [a for a in bc.columns if bc[a].dtype != object ]

le=LabelEncoder()
for col in obj_list:
  bc[col] = le.fit_transform(bc[col])
print(bc.info())

x = bc.drop('Status',axis = 1)
y = bc.Status

#normalise 
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(x)
print(X_scaled.shape,y.shape)
# y = y.values.reshape(-1,1)
# print(y.shape)

# min max
# r = MinMaxScaler()
# Y_scaled = r.fit_transform(y)
# print(Y_scaled.shape)

#spilt data as train and test
X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,y,test_size=0.2,shuffle =True)

#model 
model = Sequential()
model.add(Dense(25,input_dim =14,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(optimizer='adam',loss = 'mean_squared_error',metrics=['accuracy'])

fit = model.fit(X_train,Y_train,epochs=50,batch_size=5,verbose =1,validation_split=0.2)
# print(fit)

# print(fit.history.keys())
# plt.plot(fit.history['loss'])
# plt.plot(fit.history['val_loss'])
# plt.title('Model Loss Progression During Training/Validation')
# plt.ylabel('Training and Validation Losses')
# plt.xlabel('Epoch Number')
# plt.legend(['Training Loss', 'Validation Loss'])
# plt.show()

sample = np.array([[19,0,2,1,3,2,3,1,128,1,1,25,12,100]])
predict = model.predict(sample)
print('status:',predict)




