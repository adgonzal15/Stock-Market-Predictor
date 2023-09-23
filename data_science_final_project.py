import yfinance as yf
import pandas as pd
import numpy as np
#import nsepy as nse
#from datetime import date
import matplotlib.pyplot as plt
#import statistics as stat
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


# Getting ticker data
ticker_symbol = "SPY"
SPY_raw = yf.Ticker(ticker_symbol)
hist = SPY_raw.history(period="max")
SPY_raw_data = hist
#-----------------------------

# Manipulating the Data Set for Weekly and Daily Returns
SPY_frame = pd.DataFrame()
SPY_frame['Close'] = SPY_raw_data['Close']
SPY_frame['Weekly Returns'] = (SPY_frame['Close']-SPY_frame['Close'].shift(6))/SPY_frame['Close'].shift(6)*100
SPY_frame['Daily Returns'] = (SPY_frame['Close']-SPY_frame['Close'].shift(1))/SPY_frame['Close'].shift(1)*100
SPY_frame['Forward Daily Returns'] = (SPY_frame['Close'].shift(-6)-SPY_frame['Close'])/SPY_frame['Close']*100
SPY_frame.dropna(inplace = True)
#Features: Weekly Returns, Daily Returns 
#Target: Forward Daily Returns
corr_matrix = SPY_frame.corr()
#--------------------------------------------------------------

# Data Visualization

plt.figure(figsize=(12, 6))
plt.plot(SPY_frame['Weekly Returns'], SPY_frame['Forward Daily Returns'], 'o')           
plt.xlabel('Weekly Returns')
plt.ylabel('Forward Daily Returns')
plt.title('Weekly Returns vs Forward Daily Returns')

plt.show()

plt.figure(figsize=(12, 6))
plt.plot(SPY_frame['Daily Returns'], SPY_frame['Forward Daily Returns'], 'o')           
plt.xlabel('Daily Returns')
plt.ylabel('Forward Daily Returns')
plt.title('Daily Returns vs Forward Daily Returns')

plt.show()

plt.figure(figsize=(12, 6))
plt.plot(SPY_frame['Daily Returns'], SPY_frame['Weekly Returns'], 'o')           
plt.xlabel('Daily Returns')
plt.ylabel('Weekly Returns')
plt.title('Daily Returns vs Weekly Returns')

plt.show()


fig = plt.figure()
ax = plt.axes(projection="3d")
x = SPY_frame['Daily Returns']
y = SPY_frame['Weekly Returns']
z = SPY_frame['Forward Daily Returns']

ax.plot3D(x, y, z)

ax.scatter3D(x, y, z, c=z, cmap='cividis');

plt.show()

#------------------------------------------------------------

# Building the predictive model

predictors = ['Daily Returns','Weekly Returns']
X = SPY_frame[predictors]
y = SPY_frame['Forward Daily Returns']

# Initialise and fit model
lm = LinearRegression()
model = lm.fit(X, y)

print(f'intercept = {model.intercept_}')
print(f'coefficient = {model.coef_}')

#------------------------------------------------------------

# Graphing Solution with points

def F(x,y):
    return -0.12366351*x+(-0.06644829)*y+0.27612313783061254

fig = plt.figure()
ax = plt.axes(projection="3d")
x1 = SPY_frame['Daily Returns']
x2 = SPY_frame['Weekly Returns']
y1 = SPY_frame['Forward Daily Returns']
z2 = F(x,y)

#ax.plot3D(x, y, z1)
ax.plot3D(x1,x2,z2)


ax.scatter3D(x1,x2 ,y1, c=y1, cmap='cividis');
ax.scatter3D(x1,x2,z2, c=z2, cmap='cividis');

plt.show()

# ----------------------------------------------------------

# Analysing the Error

pred_y = model.predict(X)

MSE = np.square(np.subtract(pred_y,y1)).mean()
print('MSE: ' + str(MSE))
RMSE = np.sqrt(MSE)
print('RMSE: '+ str(RMSE))

# ----------------------------------------------------------