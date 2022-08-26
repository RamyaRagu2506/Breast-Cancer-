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

X = bc.drop('Status',axis=1)
y = bc['Status']
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
print(X_scaled.shape)
print(y.shape)

y = y.values.reshape(-1,1)

print(y.shape)


scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.20)

model = Sequential()
model.add(Dense(25, input_dim=14, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=5,  verbose=1, validation_split=0.2)
# print(epochs_hist)

# plt.plot(epochs_hist.history['loss'])
# plt.plot(epochs_hist.history['val_loss'])

# plt.title('Model Loss Progression During Training/Validation')
# plt.ylabel('Training and Validation Losses')
# plt.xlabel('Epoch Number')
# plt.legend(['Training Loss', 'Validation Loss'])
# plt.show()

print(bc.columns)

print(bc['Grade'].describe())
print(bc['A Stage'].describe())
print(bc['Tumor Size'].describe())
print(bc['Estrogen Status'].describe())
print(bc['Progesterone Status'].describe())
print(bc['Regional Node Examined'].describe())
print(bc['Reginol Node Positive'].describe())
print(bc['Survival Months'].describe())
