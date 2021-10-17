import numpy as np
import pandas as pd
import datetime

data = pd.read_csv("marico_close_with_dates.csv")
price = list(data["Price"])
index = pd.DatetimeIndex(data["Date"])
price_indexed1 = pd.Series(data=price,index=index)

price_indexed2 = []

#price_indexed2 = price_indexed1.loc['01-01-2018':'31-12-2018']
price_indexed2 = price_indexed1.loc['2018-01-01':'2018-12-31']

m = []
def mov_avg1(span=10):
    m1 = []
    m1 = price_indexed2.rolling(span).mean()
    m = m1[-1]
    return m

m2 =[]
def mov_avg2(span = 10):
    k =[]
    for i in range(span):

        k = mov_avg1(i)
        m2.append(k)
    return m2

marico_m = mov_avg2()
print(marico_m)