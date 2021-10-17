import pandas as pd
import numpy as np
np.random.seed(11)
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from keras.layers import Dense, LeakyReLU
from keras import Sequential
from keras import callbacks
from sklearn.model_selection import train_test_split


data = pd.read_csv('marico_high.csv')
target = 'PredHigh'

price = data['Price'].values
open = data['Open'].values
high = data['High'].values
low = data['Low'].values
obv = data['OBV'].values
vpt = data['VPT'].values
gold = data['GOLD'].values
oil = data['OIL'].values
fex = data['FEX'].values
pred = data['PredHigh'].values

price = np.array(price)
open = np.array(open)
high = np.array(high)
low = np.array(low)
obv = np.array(obv)
vpt = np.array(vpt)
gold = np.array(gold)
oil = np.array(oil)
fex = np.array(fex)
pred = np.array(pred)

obv = obv.reshape(-1,1)
vpt = vpt.reshape(-1,1)
gold = gold.reshape(-1,1)
oil = oil.reshape(-1,1)
pred = pred.reshape(-1,1)

scaler1 = MinMaxScaler(feature_range=(20,150))
#scaler1 = MaxAbsScaler()

vpt_scaled = scaler1.fit_transform(vpt)
obv_scaled = scaler1.fit_transform(obv)
gold_scaled =scaler1.fit_transform(gold)
oil_scaled = scaler1.fit_transform(oil)

#df = pd.DataFrame({'Price':price , 'Open' : open ,'High' : high,'Low' : low,'OBV' : obv,'VPT':vpt,'GOLD' :gold,'OIL' :oil,'FEX':fex,'Pred':pred },columns=['Price','Open','High','Low','OBV','VPT','GOLD','OIL','FEX','Pred'])
#print(df.head())

#print(vpt_scaled[0:10])
#print(obv_scaled[0:16])
#print(gold_scaled[0:10])
#print(oil_scaled[0:10])
price = price.reshape(1,-1)
open = open.reshape(1,-1)
high = high.reshape(1,-1)
low = low.reshape(1,-1)
obv_scaled = obv_scaled.reshape(1,-1)
vpt_scaled = vpt_scaled.reshape(1,-1)
gold_scaled = gold_scaled.reshape(1,-1)
oil_scaled= oil_scaled.reshape(1,-1)
fex = fex.reshape(1,-1)
pred = pred.reshape(1,-1)


data2 = np.stack((price,open,high,low,obv_scaled,vpt_scaled,gold_scaled,oil_scaled,fex),axis=-1)
data2 = data2.reshape(-1,9)
print(data2[0:5])

pred = pred.reshape(-1,1)

xtrain = data2[0:4200]
ytrain = pred[0:4200]
xtest = data2[4200:]
ytest = pred[4200:]

xtrain = xtrain.reshape(-1,9)
ytrain = ytrain.reshape(-1,1)
xtest = xtest.reshape(-1,9)
ytest = ytest.reshape(-1,1)

print(np.shape(xtrain))
print(np.shape(ytrain))
print(np.shape(xtest))
print(np.shape(ytest))

model = Sequential()

model.add(Dense(9,input_dim=9,kernel_initializer='uniform'))
#model.add(LeakyReLU(alpha=0.001))

#model.add(Dense(5,kernel_initializer='uniform'))
#model.add(LeakyReLU(alpha = 0.01))

model.add(Dense(4,kernel_initializer='uniform'))
model.add(LeakyReLU(alpha = 0.01))

model.add(Dense(1,kernel_initializer= 'uniform'))

model.compile(loss='mape', optimizer='adam',metrics=['mape'])

history = model.fit(xtrain,ytrain,verbose=1 ,epochs=20,batch_size=10,validation_data=(xtest,ytest))
score1 = model.evaluate(xtest, ytest, verbose=1)
print('Test score:', score1)

score2 = model.evaluate(xtrain, ytrain, verbose=1)
print('train score:', score2)

model.save("marico_high_model.h5")
