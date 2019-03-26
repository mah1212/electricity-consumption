# Electricity Power Consumption Practice Project
# Time Series, Basic LSTM, CNN

import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making

from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout


df = pd.read_csv('household_power_consumption.txt', sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='dt')



# =============================================================================
# 
# 1) Note that data include 'nan' and '?' as a string. I converted both to numpy 
# nan in importing stage (above) and treated both of them the same.
# 2) I merged two columns 'Date' and 'Time' to 'dt'.
# 3) I also converted in the above, the data to time-series type, by taking 
# index to be the time.
# =============================================================================

df.head()

df.info()
# =============================================================================
# 
# DatetimeIndex: 2075259 entries, 2006-12-16 17:24:00 to 2010-11-26 21:02:00
# Data columns (total 7 columns):
# Global_active_power      float64
# Global_reactive_power    float64
# Voltage                  float64
# Global_intensity         float64
# Sub_metering_1           float64
# Sub_metering_2           float64
# Sub_metering_3           float64
# dtypes: float64(7)
# memory usage: 126.7 MB
# =============================================================================


df.dtypes
# =============================================================================
# 
# Global_active_power      float64
# Global_reactive_power    float64
# Voltage                  float64
# Global_intensity         float64
# Sub_metering_1           float64
# Sub_metering_2           float64
# Sub_metering_3           float64
# dtype: object
# =============================================================================

df.shape # (2075259, 7)

df.describe()
# =============================================================================
# 
#        Global_active_power       ...        Sub_metering_3
# count         2.049280e+06       ...          2.049280e+06
# mean          1.091615e+00       ...          6.458447e+00
# std           1.057294e+00       ...          8.437154e+00
# min           7.600000e-02       ...          0.000000e+00
# 25%           3.080000e-01       ...          0.000000e+00
# 50%           6.020000e-01       ...          1.000000e+00
# 75%           1.528000e+00       ...          1.700000e+01
# max           1.112200e+01       ...          3.100000e+01
# 
# 
# =============================================================================

df.columns
# =============================================================================
# 
# Index(['Global_active_power', 'Global_reactive_power', 'Voltage',
#        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
#        'Sub_metering_3'],
#       dtype='object')
# =============================================================================


## finding all columns that have nan:

droping_list_all=[]
for j in range(0,7):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
        #print(df.iloc[:,j].unique())
droping_list_all


# filling nan with mean in any columns
for j in range(0,7):        
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

# another sanity check to make sure that there are not more any nan
df.isnull().sum()


# resample over day, and show the sum and mean of Global_active_power. It is 
# seen that mean and sum of resampled data set, have similar structure.

df.Global_active_power.resample('D').sum().plot(title='Global_active_power resampled over day for sum') 
plt.tight_layout()
plt.show()   

df.Global_active_power.resample('D').mean().plot(title='Global_active_power resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# mean and std of 'Global_intensity' resampled over day 
r = df.Global_intensity.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='Global_intensity resampled over day')
plt.show()


# mean and std of 'Global_reactive_power' resampled over day
r2 = df.Global_reactive_power.resample('D').agg(['mean', 'std'])
r2.plot(subplots = True, title='Global_reactive_power resampled over day', color='red')
plt.show()


# Average of 'Global_active_power' resampled over month
df['Global_active_power'].resample('M').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('Global_active_power')
plt.title('Global_active_power per month (averaged over month)')
plt.show()


## Mean of 'Global_active_power' resampled over quarter
df['Global_active_power'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('Global_active_power')
plt.title('Global_active_power per quarter (averaged over quarter)')
plt.show()


# --------It is very important to note from above two plots that resampling 
#---------over larger time inteval, will diminish the periodicity of system 
#---------as we expect. This is important for machine learning feature engineering.

## mean of 'Voltage' resampled over month
df['Voltage'].resample('M').mean().plot(kind='bar', color='red')
plt.xticks(rotation=60)
plt.ylabel('Voltage')
plt.title('Voltage per quarter (summed over quarter)')
plt.show()


df['Sub_metering_1'].resample('M').mean().plot(kind='bar', color='brown')
plt.xticks(rotation=60)
plt.ylabel('Sub_metering_1')
plt.title('Sub_metering_1 per quarter (summed over quarter)')
plt.show()


# compare the mean of different features resampled over day. 
# specify columns to plot
cols = [0, 1, 2, 3, 5, 6]
i = 1
groups=cols
values = df.resample('D').mean().values
# plot each column
plt.figure(figsize=(15, 10))
for group in groups:
	plt.subplot(len(cols), 1, i)
	plt.plot(values[:, group])
	plt.title(df.columns[group], y=0.75, loc='right')
	i += 1
plt.show()


## resampling over week and computing mean
df.Global_reactive_power.resample('W').mean().plot(color='y', legend=True)
df.Global_active_power.resample('W').mean().plot(color='r', legend=True)
df.Sub_metering_1.resample('W').mean().plot(color='b', legend=True)
df.Global_intensity.resample('W').mean().plot(color='g', legend=True)
plt.show()


# hist plot of the mean of different feature resampled over month 
df.Global_active_power.resample('M').mean().plot(kind='hist', color='r', legend=True )
df.Global_reactive_power.resample('M').mean().plot(kind='hist',color='b', legend=True)
#df.Voltage.resample('M').sum().plot(kind='hist',color='g', legend=True)
df.Global_intensity.resample('M').mean().plot(kind='hist', color='g', legend=True)
df.Sub_metering_1.resample('M').mean().plot(kind='hist', color='y', legend=True)
plt.show()


## The correlations between 'Global_intensity', 'Global_active_power'
data_returns = df.pct_change()
sns.jointplot(x='Global_intensity', y='Global_active_power', data=data_returns)  

plt.show()


# From above two plots it is seen that 'Global_intensity' and 
# 'Global_active_power' correlated. But 'Voltage', 'Global_active_power' 
# are less correlated. This is important observation for machine learning 
# purpose.

## The correlations between 'Voltage' and  'Global_active_power'
sns.jointplot(x='Voltage', y='Global_active_power', data=data_returns)  
plt.show()


# correlation among features
# Correlations among columns
plt.matshow(df.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('without resampling', size=15)
plt.colorbar()
plt.show()


# Correlations of mean of features resampled over months
plt.matshow(df.resample('M').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('resampled over month', size=15)
plt.colorbar()
plt.margins(0.02)
plt.matshow(df.resample('A').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('resampled over year', size=15)
plt.colorbar()
plt.show()

# It is seen from above that with resampling techniques one can change the 
# correlations among features. This is important for feature engineering.

# frame the supervised learning problem as predicting the Global_active_power 
# at the current time (t) given the Global_active_power measurement and other 
# features at the prior time step.

type(data_returns)
data_returns.shape[1]
n_vars = 1 if type(data_returns) is list else data_returns.shape[1]

dff = pd.DataFrame(data_returns)
dff.head()

cols, names = list(), list()
cols.append(dff.shift(1))
cols

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# In order to reduce the computation time, and also get a quick result to 
# test the model. One can resmaple the data over hour (the original data are 
# given in minutes). This will reduce the size of data from 2075259 to 34589 
# but keep the overall strucure of data as shown in the above.

## resampling of data over hour
df_resample = df.resample('h').mean() 
df_resample.shape
df_resample.head()

# scale all features in range of [0,1].

## If you would like to train based on the resampled data (over hour), 
# then used below
values = df_resample.values 
#np_val = df_resample.to_numpy()
#np_val
# =============================================================================
# 
#  DataFrame.values
# 
#     Return a Numpy representation of the DataFrame.
# 
#     Warning
# 
#     We recommend using DataFrame.to_numpy() instead.
# 
#     Only the values in the DataFrame will be returned, the axes labels will be removed.
# 
# =============================================================================
## full data without resampling
#values = df.values

# integer encode direction
# ensure all data is float
#values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.head()
# drop columns we don't want to predict
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())

# Above I showed 7 input variables (input series) and the 1 output variable 
# for 'Global_active_power' at the current time in hour (depending on resampling)




#------- Train Test Split --------------------------------------------------


# First, split the prepared dataset into train and test sets. 
# To speed up the training of the model (for the sake of the demonstration), 
# we will only train the model on the first year of data, then evaluate it on 
# the next 3 years of data.

# split into train and test sets
values = reframed.values

n_train_time = 365*24
train = values[:n_train_time, :]
test = values[n_train_time:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 

# We reshaped the input into the 3D format as expected by LSTMs, 
# namely [samples, timesteps, features].

# =============================================================================
# 
# Model architecture
# 1) LSTM with 100 neurons in the first visible layer
# 3) dropout 20%
# 4) 1 neuron in the output layer for predicting Global_active_power.
# 5) The input shape will be 1 time step with 7 features.
# 6) I use the Mean Absolute Error (MAE) loss function and the efficient Adam version of stochastic gradient descent.
# 7) The model will be fit for 20 training epochs with a batch size of 70
# 
# 
# =============================================================================

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
#    model.add(LSTM(70))
#    model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')



# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 7))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# Note that in order to improve the model, one has to adjust epochs and batch_size.


## time steps, every step is one hour (you can easily convert the time step to the actual time index)
## for a demonstration purpose, I only compare the predictions in 200 hours. 
aa=[x for x in range(200)]
plt.plot(aa, inv_y[:200], marker='.', label="actual")
plt.plot(aa, inv_yhat[:200], 'r', label="prediction")
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()


# =============================================================================
# 
# Final remarks
# * Here I have used the LSTM neural network which is now the state-of-the-art for sequencial problems.
# * In order to reduce the computation time, and get some results quickly, I took the first year of data (resampled over hour) to train the model and the rest of data to test the model.
# * I put together a very simple LSTM neural-network to show that one can obtain reasonable predictions. However numbers of rows is too high and as a result the computation is very time-consuming (even for the simple model in the above it took few mins to be run on 2.8 GHz Intel Core i7). The Best is to write the last part of code using Spark (MLlib) running on GPU.
# * Moreover, the neural-network architecture that I have designed is a toy model. It can be easily improved by adding CNN and dropout layers. The CNN is useful here since there are correlations in data (CNN layer is a good way to probe the local structure of data).
# 
# =============================================================================
