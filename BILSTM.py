from keras.layers import Bidirectional
import pandas as pd
import numpy as np
from keras.optimizers import SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import  Dense,LSTM

#Then loaded the dataset
data=pd.read_csv('dataset.csv')
#convert the data to dataframe to numerical
x2=data.to_numpy()
Fea = x2[:, 0:16]
lab=x2[:,16]
#then prepare to given Reduction datas into Bidirectional
Feature_org=Fea
Feature = np.empty([len(Feature_org),1,len(Feature_org[0])], dtype=object)
for j in range(len(Feature_org)):
    for i in range(len(Feature_org[0])):
      Feature[j,0,i]=Feature_org[j,i]
    
x=Feature

def dat_nor(X_train):
        scalers = {}
        for i in range(X_train.shape[1]):
            scalers[i] = StandardScaler()
            X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
        return X_train,scalers
def Label_con(Label):
        Lab = np.empty([len(Label),1], dtype=object)
        for i in range(len(Label)):
            Lab[i,0]=Label[i]
        return Lab
x_tr = x.astype('float32')
dat,scalers = dat_nor(x_tr)
Lab=Label_con(lab)
Label = Lab.astype('float32')

# Spliting into train and test
x_train, x_test, y_train, y_test = train_test_split(dat,Label, test_size=0.2, random_state=0)

model = Sequential()

#After that Bidirectional layers
model.add(Bidirectional(LSTM(5, input_shape=(1, len(Fea[0])), activation='tanh')))
model.add(Dense(10, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(1000, activation='tanh'))
# model.add(Dense(100, activation='tanh'))
model.add(Dense(1))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss="mean_squared_error", optimizer=optimizer)

# Trainning process
history=model.fit(dat,Label,epochs=500, batch_size=8, verbose=1)
trainPredict = model.predict(x_test)