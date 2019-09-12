import os, sys, pickle, datetime, time, csv, math
import pandas as pd
from models import *
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import numpy as npy
from matplotlib import pyplot
import xgboost as xgb
from xgboost import plot_importance
from stockstats import StockDataFrame as Sdf
from itertools import combinations, chain
from csvgenerator import generate_csvs
from vars import *

def build_Model(file, fwdtime, predict_list, stock, threshold, message):
	
	group = pd.read_csv(file)
	#Sort by date
	group = group.sort_values(by=['Date'])
	#Set up a range of times
	group['time'] = group['Date']
	group['Date'] = pd.to_datetime(group['Date'])
	group['Date'] = group['Date'].dt.strftime('%Y-%m-%d')

	group.set_index(keys='Date', inplace=True)
	
	#Drop holidays here:
	group.drop('2018-07-03', inplace=True)

	group['time'] = pd.DatetimeIndex(group['time'])
	group.set_index(keys='time', inplace=True)
	
	group = group.between_time('9:29','16:00')
	#Retype into a stockstats-dataframe
	group = Sdf.retype(group)

	#Generate parameters
	for predictor in predict_list:
		if predictor not in ['close60to3900', 'close390to3900', 'close180to390', 'close180to3900']:
			group[predictor]

	group['close60to3900'] = group['close_60_sma'] / group['close_3900_sma']
	group['close390to3900'] = group['close_390_sma'] / group['close_3900_sma']
	group['close180to390'] = group['close_180_sma'] / group['close_390_sma']
	group['close180to3900'] = group['close_180_sma'] / group['close_3900_sma']

	basic_info = ['volume', 'close_12_ema', 'close_26_ema', 'boll', 'close_10_sma', 'close_20_sma', 'close_20_mstd', "Target", "symbol", "time", "open", "close", "high", "low", f'close_{fwdtime}', 'close_eod', 'percent_change']

	'''
	#Create a shifted closing value column as a testing target
	group[f'close_{fwdtime}'] = group['close']
	#shift it by a the forward time
	group[f'close_{fwdtime}'] = group[f'close_{fwdtime}'].shift(-fwdtime)
	group['percent_change'] = (100 * ((group[f'close_{fwdtime}'] - group['close']) / group['close']))

	'''

	#This is for EoD style prediction
	#Create new empty column
	group[f'close_eod'] = ''
	#Get how many days we need to set the EOD price for
	number_of_days = int(len(group.index) / 390)
	#Get the last tradable value for each day
	EOD_targets = group.iloc[389::390, 3]
	for c in range(number_of_days):
		group.close_eod.iloc[(c * 390):(390 + (c * 390))] = EOD_targets[c]
	group['percent_change'] = (100 * ((group['close_eod'] - group['close']) / group['close']))

	
	#Cut off the top x rows used for back calculation
	group = group.tail(len(group.index)-3900)

	#Drop any NaN rows
	group = group.dropna()

	#Create targets. These cannot be continuous. 
	group['Target'] = group.apply(target_classifier, args=(threshold,), axis=1)
	target = group['Target']

	#Grab the columns specified for the model
	X_Data = group[predict_list]
	print(predict_list)
	#Set up the training and testing fractions
	x_train, x_test, y_train, y_test = train_test_split(X_Data, target, test_size=0.01, random_state=0, shuffle=False) #shuffle=False for non-random
	model = xgb.XGBClassifier()
	model.fit(x_train, y_train)
	pickle.dump(model, open(f"models/{stock}_{message}_{threshold}_model.pickle.dat", "wb"))
	print(f'Model saved for {stock}')

def use_Model(dataframe, predict_list, xgb_model):

	group = Sdf.retype(dataframe)

	#Generate parameters
	for predictor in predict_list:
		if predictor not in ['close60to3900', 'close390to3900', 'close180to390', 'close180to3900']:
			group[predictor]

	group['close60to3900'] = group['close_60_sma'] / group['close_3900_sma']
	group['close390to3900'] = group['close_390_sma'] / group['close_3900_sma']
	group['close180to390'] = group['close_180_sma'] / group['close_390_sma']
	group['close180to3900'] = group['close_180_sma'] / group['close_3900_sma']

	#Grab the columns specified for the model
	predict_data = group[predict_list]
	predict_data = predict_data.tail(1)
	#Set up the training and testing fractions
	prediction = xgb_model.predict(predict_data)
	
	return prediction[0]




