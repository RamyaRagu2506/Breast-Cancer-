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
from keras import Input
from keras.layers import Dense, SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
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
print(y[:10])

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
model.add(Input(shape=(14,1), name='Input-Layer'))
# model.add(Input(16,input_shape =(x.shape[1],len(x.columns)),activation='relu'))
model.add(SimpleRNN(units=1, activation='tanh', name='Hidden-Recurrent-Layer'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()


model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

# early stopping callback
# This callback will stop the training when there is no improvement in  
# the validation loss for 10 consecutive epochs.  
es = EarlyStopping(monitor='val_accuracy', 
                                   mode='max', # don't minimize the accuracy!
                                   patience=10,
                                   restore_best_weights=True)

fit = model.fit(X_train,Y_train,epochs=50,batch_size=5,verbose =1,validation_split=0.2,callbacks=[es],shuffle=True)
# print(fit)

print(fit.history.keys())
plt.plot(fit.history['accuracy'])
plt.plot(fit.history['val_accuracy'])
plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

#classification report

preds = np.round(model.predict(X_test),0)
print(confusion_matrix(Y_test, preds))

print(classification_report(Y_test, preds))

sample = np.array([[19,0,2,1,3,2,3,1,128,1,1,25,12,100]])
predict = model.predict(sample)
print('Predicted Status For Ramya:',predict)



#serialize model to json
model_json = model.to_json()
with open("model.json","w") as json_file:
  json_file.write(model_json)
#serialize weight to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

