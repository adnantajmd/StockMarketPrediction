from datetime import datetime
import _datetime
import pandas as pd
import numpy as np
from dateutil import parser
data1 = pd.read_csv("marico_close_with_dates.csv")
data1 = pd.DataFrame(data1,columns=["Date","Price"])
data2 = list(data1['Date'])
data3 = list(data1['Price'])

index = pd.DatetimeIndex(data2)
data = pd.Series(data3,index=index)

date_list = ["1999","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018"]
d1 =[]
for i in range(np.size(date_list)):
    d = np.mean(data[date_list[i]])
    d1.append(d)

marico_date_data = pd.Series(d1,index=date_list)  # creating series(array) with years as indices


def mean_over_years(data=marico_date_data,num_years=5,index=date_list):
    marico_avg = []
    avg2=[]
    if num_years == 1:
       return list(marico_date_data)

    elif np.size(index)%num_years == 0:
        k=0
        for i in range(int(np.size(index)/num_years)):
            d = 0

            for j in range(num_years):

                 d = d + data[k + j]

            marico_avg.append(d/num_years)

            k = (i+1)*j +1 + i

        return marico_avg

    elif np.size(index)%num_years !=0:

        k = 0
        for i in range(int(np.size(index) / num_years)):
            d = 0

            for j in range(num_years):
                d = d + data[k + j]

            marico_avg.append(d / num_years)

            k = (i + 1) * j + 1 + i
        d =0
        for i in range(np.size(index)%num_years):

            d = d+ data[k+i]

        marico_avg.append(d/(np.size(index)%num_years))

        return marico_avg


