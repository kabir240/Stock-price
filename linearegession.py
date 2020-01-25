import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df=quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/ df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)#in real life ml appication we cant deal with vacant data so we need to fill all the vacant spaces inside our dataset

forecast_out = int(math.ceil(0.01*len(df)))#tAKE 10% of the data and ceil this data upto a whole number{least integer function}
#print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)#shifted the data 10 days in fututre



x=np.array(df.drop(['label'],1))

x=preprocessing.scale(x)

x=x[:-forecast_out]
x_lately=x[-forecast_out:]#for these data we dont have y value

df.dropna(inplace=True)

y=np.array(df['label'])
Y=np.array(df['label'])



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#clf = LinearRegression(n_jobs=-1)#n_jobs means how many jobs u are performing how many data are processing at time; -1 means max by the machine itself
#clf.fit(x_train,y_train)
#with open('linearregression.pickle','wb') as f:
    #pickle.dump(clf, f)


pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(x_test,y_test)

#print(accuracy)

forecast_set=clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



