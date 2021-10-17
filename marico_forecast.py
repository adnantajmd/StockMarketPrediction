import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from keras.layers import Dense, LeakyReLU
from keras import Sequential
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.models import model_from_json as m
from keras.models import load_model

np.random.seed(11)


datac = pd.read_csv('marico_forecast.csv')

##################### creating 1-d arrays for each column for transforming required parameter #####################

vol = datac['Volume'].values
price = datac['Price'].values
open = datac['Open'].values
high = datac['High'].values
low = datac['Low'].values
obv = datac['OBV'].values
vpt = datac['VPT'].values
gold = datac['GOLD'].values
oil = datac['OIL'].values
fex = datac['FEX'].values

price = np.array(price)
open = np.array(open)
high = np.array(high)
low = np.array(low)
obv = np.array(obv)
vpt = np.array(vpt)
gold = np.array(gold)
oil = np.array(oil)
fex = np.array(fex)
vol = np.array(vol)

obv = obv.reshape(-1,1)
vpt = vpt.reshape(-1,1)
gold = gold.reshape(-1,1)
oil = oil.reshape(-1,1)

scaler1 = MinMaxScaler(feature_range=(20,150))

vpt_scaled = scaler1.fit_transform(vpt)
obv_scaled = scaler1.fit_transform(obv)
gold_scaled =scaler1.fit_transform(gold)
oil_scaled = scaler1.fit_transform(oil)

############# changing the axis of all the 1-d arrays #######################

price = price.reshape(1,-1)
open = open.reshape(1,-1)
high = high.reshape(1,-1)
low = low.reshape(1,-1)
obv_scaled = obv_scaled.reshape(1,-1)
vpt_scaled = vpt_scaled.reshape(1,-1)
gold_scaled = gold_scaled.reshape(1,-1)
oil_scaled= oil_scaled.reshape(1,-1)
fex = fex.reshape(1,-1)

################ stacking all the 1-d arrays horizontally to create combined 2-d array ###################

data2 = np.stack((price,open,high,low,obv_scaled,vpt_scaled,gold_scaled,oil_scaled,fex),axis=-1)
data2 = data2.reshape(-1,9)

#################################### for high #########################################

datah = pd.read_csv('marico_forecast.csv')

price = datah['Price'].values
open = datah['Open'].values
high = datah['High'].values
low = datah['Low'].values
obv = datah['OBV'].values
vpt = datah['VPT'].values
gold = datah['GOLD'].values
oil = datah['OIL'].values
fex = datah['FEX'].values


price = np.array(price)
open = np.array(open)
high = np.array(high)
low = np.array(low)
obv = np.array(obv)
vpt = np.array(vpt)
gold = np.array(gold)
oil = np.array(oil)
fex = np.array(fex)

obv = obv.reshape(-1,1)
vpt = vpt.reshape(-1,1)
gold = gold.reshape(-1,1)
oil = oil.reshape(-1,1)

scaler1 = MinMaxScaler(feature_range=(20,150))

vpt_scaled = scaler1.fit_transform(vpt)
obv_scaled = scaler1.fit_transform(obv)
gold_scaled =scaler1.fit_transform(gold)
oil_scaled = scaler1.fit_transform(oil)

price = price.reshape(1,-1)
open = open.reshape(1,-1)
high = high.reshape(1,-1)
low = low.reshape(1,-1)
obv_scaled = obv_scaled.reshape(1,-1)
vpt_scaled = vpt_scaled.reshape(1,-1)
gold_scaled = gold_scaled.reshape(1,-1)
oil_scaled= oil_scaled.reshape(1,-1)
fex = fex.reshape(1,-1)

datahigh = np.stack((price,open,high,low,obv_scaled,vpt_scaled,gold_scaled,oil_scaled,fex),axis=-1)
datahigh = datahigh.reshape(-1,9)


#################################### for low #############################################

datal = pd.read_csv('marico_forecast.csv')

price = datal['Price'].values
open = datal['Open'].values
high = datal['High'].values
low = datal['Low'].values
obv = datal['OBV'].values
vpt = datal['VPT'].values
gold = datal['GOLD'].values
oil = datal['OIL'].values
fex = datal['FEX'].values

price = np.array(price)
open = np.array(open)
high = np.array(high)
low = np.array(low)
obv = np.array(obv)
vpt = np.array(vpt)
gold = np.array(gold)
oil = np.array(oil)
fex = np.array(fex)

obv = obv.reshape(-1,1)
vpt = vpt.reshape(-1,1)
gold = gold.reshape(-1,1)
oil = oil.reshape(-1,1)

scaler1 = MinMaxScaler(feature_range=(20,150))

vpt_scaled = scaler1.fit_transform(vpt)
obv_scaled = scaler1.fit_transform(obv)
gold_scaled =scaler1.fit_transform(gold)
oil_scaled = scaler1.fit_transform(oil)

price = price.reshape(1,-1)
open = open.reshape(1,-1)
high = high.reshape(1,-1)
low = low.reshape(1,-1)
obv_scaled = obv_scaled.reshape(1,-1)
vpt_scaled = vpt_scaled.reshape(1,-1)
gold_scaled = gold_scaled.reshape(1,-1)
oil_scaled= oil_scaled.reshape(1,-1)
fex = fex.reshape(1,-1)

datalow = np.stack((price,open,high,low,obv_scaled,vpt_scaled,gold_scaled,oil_scaled,fex),axis=-1)
datalow = datalow.reshape(-1,9)


##################################### predicting high low close from their respective models ############################
model2 =  Sequential()

new_model_close = load_model("marico_close_model.h5")
new_model_close.compile(loss = 'mape',optimizer = 'adam',metrics = ['mape'])

new_model_high = load_model("marico_high_model.h5")
new_model_high.compile(loss = 'mape',optimizer = 'adam',metrics = ['mape'])

new_model_low = load_model("marico_low_model.h5")
new_model_low.compile(loss = 'mape',optimizer = 'adam',metrics = ['mape'])

cp = new_model_close.predict(data2)
hp = new_model_high.predict(datahigh)
lp = new_model_low.predict(datalow)

marico_predicted_close = cp[0][0]
marico_predicted_high = hp[0][0]
marico_predicted_low = lp[0][0]
marico_vol = vol[0]
marico_open = open[0][0]
marico_close = price[0][0]
marico_high = high[0][0]
marico_low = low[0][0]



print("predicted close",marico_predicted_close)
print("predeicted high",marico_predicted_high)
print("predicted_low",marico_predicted_low)
print("Volume",marico_vol)
print("open",marico_open)
print("close",marico_close)
print("high",marico_high)
print("low",marico_low)

