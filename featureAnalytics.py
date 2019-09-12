import os, sys, pickle, datetime, time, csv, math, glob, timeit
import pandas as pd
import multiprocessing

import warnings
import talib

from models import *
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
import numpy as npy
from matplotlib import pyplot
import xgboost as xgb
from stockstats import StockDataFrame as Sdf
from itertools import combinations, chain
from collections import OrderedDict

import pymysql
import boto3
import dbconfig as dbc

from sqlalchemy import create_engine


rds = boto3.client('rds')
files = glob.glob('C:\\Projects\\Stocks\\Feature Importance Testing\\*.csv')
#thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
thresholds = [0.2]
#fwdtime = 60
pd.options.mode.chained_assignment = None


def windowSumPercentChange(df, time):

	df['firstClose'] = df.groupby(df.index.date)['close'].transform('first')
	df['intradayPercentChange'] = (-100 * ((df['firstClose'] - df['close']) / df['firstClose']))
	
	value = pd.Series(df['intradayPercentChange'].rolling(time).sum())

	return value

def dailyCumulativePercentChange(df):
	df['firstClose'] = df.groupby(df.index.date)['close'].transform('first')
	df['intradayPercentChange'] = (-100 * ((df['firstClose'] - df['close']) / df['firstClose']))
	
	value = df['intradayPercentChange'].groupby(df.index.date).cumsum()   
	
	return value

def closeWindowLooseComparison(df, time):
	df['farClose'] = df['close'].rolling(time).mean()
	df['distantClose'] = df['close'].rolling(time*4).mean()

	value = pd.Series(df.apply(closeWindowLooseSignal, axis=1))

	return value

def closeWindowTightComparison(df, time):
	df['farClose'] = df['close'].rolling(time).mean()
	df['tightClose'] = df['close'].rolling(int(time/4)).mean()

	value = pd.Series(df.apply(closeWindowTightSignal, axis=1))

	return value

def volumeWindowComparison(df, time):

	df['farVol'] = df['volume'].rolling(time).mean()
	df['closeVol'] = df['volume'].rolling(int(time/4)).mean()

	value = pd.Series(df.apply(volumeWindowSignal, axis=1))

	#df.drop(['farVol', 'closeVol', 'farClose', 'closeClose', 'distantClose'], axis=1, inplace=True)
	return value

def voluminousDrop(df, time):
	df['end_close'] = df.groupby(df.index.date)['close'].transform('last')
	df['priorClose'] = df['end_close'].shift(1, freq ='B') 
	df['firstClose'] = df.groupby(df.index.date)['close'].transform('first')
	
	df['overnightChange'] = -100 * ((df['priorClose'] - df['firstClose']) / df['priorClose'])

	df['minOvernightChange'] = df['overnightChange'].rolling(time, min_periods=1).min()

	df['timeVolume'] = df['volume'].rolling(time).mean()
	df['historicVolume'] = df['volume'].rolling(time*4).mean()

	value = pd.Series(df.apply(voluminousDropSignal, axis=1))

	#df.drop(['farVol', 'closeVol', 'farClose', 'closeClose', 'distantClose'], axis=1, inplace=True)''
	return value

def voluminousRise(df, time):
	df['end_close'] = df.groupby(df.index.date)['close'].transform('last')
	df['priorClose'] = df['end_close'].shift(1, freq ='B') 
	df['firstClose'] = df.groupby(df.index.date)['close'].transform('first')
	
	df['overnightChange'] = -100 * ((df['priorClose'] - df['firstClose']) / df['priorClose'])

	df['maxOvernightChange'] = df['overnightChange'].rolling(time, min_periods=1).max()

	df['timeVolume'] = df['volume'].rolling(time).mean()
	df['historicVolume'] = df['volume'].rolling(time*4).mean()


	

	value = pd.Series(df.apply(voluminousRiseSignal, axis=1))

	#df.drop(['farVol', 'closeVol', 'farClose', 'closeClose', 'distantClose'], axis=1, inplace=True)
	return value

def combination_test(stock, df, comboList, chunk_size, first_monday, last_monday, threshold, mId):
	#ADD/INITIALIZE THE AVERAGE RATINGS HERE!!!
	#num_chunks = int(math.floor(len(df.index) // chunk_mover) - (chunk_size // chunk_mover))

	print(f'Running backtest on {stock} at threshold: {threshold} for features: {comboList}')
	
	first_day = datetime.datetime.strptime(first_monday, "%Y-%m-%d")
	last_day = datetime.datetime.strptime(last_monday, "%Y-%m-%d")
	
	#So this is the last monday + 6 to get to the last sunday (last actual day in dataframe) + 1 more day due to off by one rounding for floor division.
	last_day = last_day + datetime.timedelta(days = 7)

	#Let's count the amount of weeks present between the first and last mondays
	num_chunks = ((last_day - first_day).days // 7) - chunk_size
	

	tailreport_df = 0

	chunkreportDf = 0
	tailAccuracies = {}
	trainAccuracies = {}
	tailFeatImportances = {}


	tailreport_dfReg = 0
	tailFeatImportancesReg = {}


	for c in range(num_chunks):
		
		#if not (c % 9 == 0 or c == 0):
	
		
		
		tailTime = datetime.datetime.now()


		#We need chunksize number of weeks (default 20) to calculate the model, so we need to bump down chunksize weeks from our first monday
		move_weeks = chunk_size + c
		week_delta = datetime.timedelta(weeks = move_weeks)
		first_day = datetime.datetime.strptime(first_monday, "%Y-%m-%d")
		tail_firstday = first_day + week_delta
		tail_lastday = tail_firstday + datetime.timedelta(days = 6)

		tail_df = df.loc[str(tail_firstday):str(tail_lastday)]
		tail_target = tail_df[f'Target_{threshold}']
		tailTargetReg = tail_df['percent_change']

		#The chunk will be constructed similarly to the tail minus chunksize weeks. The last day of the chunk will be the sunday right before the starting monday of the tail.
		chunk_firstday = first_day + datetime.timedelta(weeks = c)
		chunk_lastday = tail_firstday - datetime.timedelta(days = 1)
		chunk_df = df.loc[str(chunk_firstday):str(chunk_lastday)]
		target = chunk_df[f'Target_{threshold}']
		targetReg = chunk_df[f'percent_change']
		

		X_Data = chunk_df[comboList]
		#Set up the training and testing fractions
		x_train, x_test, y_train, y_test = train_test_split(X_Data, target, test_size=0, random_state=0, shuffle=False) #shuffle=False for non-random
		#For reg
		x_trainReg, x_testReg, y_trainReg, y_testReg = train_test_split(X_Data, targetReg, test_size=0, random_state=0, shuffle=False) #shuffle=False for non-random

		model = xgb.XGBClassifier(
				max_depth=3
				)
		'''
		regModel = xgb.XGBRegressor(
				max_depth=3
				)
		'''

		tail_x = tail_df[comboList]
		model.fit(x_train, y_train)
		'''
		regModel.fit(x_trainReg, y_trainReg)
		tailFeatImportancesReg[c] = regModel.get_booster().get_score(importance_type= 'gain')
		tail_y_predReg = regModel.predict(tail_x)
		'''

		tailFeatImportances[c] = model.get_booster().get_score(importance_type= 'gain')
		

		#Test the tail
		
		tail_y_pred = model.predict(tail_x)
		tail_predictions = [round(value) for value in tail_y_pred]
		tail_accuracy = accuracy_score(tail_target, tail_predictions)


		tailAccuracies[c] = tail_accuracy


		tail_df['Prediction'] = tail_predictions
		tail_df['Prediction_RAW'] = tail_y_pred
		#tail_df['Prediction_Regression'] = tail_y_predReg
		
		df_taillist = tail_df.groupby(['Prediction'])

		tail_df['tailId'] = c
		tail_df['tailStart'] = tailTime
		tail_df['symbol'] = stock
		tail_df['modelId'] = mId
		tail_df['threshold'] = threshold


		#Test the model on the training data

		trainPredictions = model.predict(x_train)
		trainPredictions = [round(value) for value in trainPredictions]
		trainAccuracy = accuracy_score(target, trainPredictions)
		trainAccuracies[c] = trainAccuracy

		chunk_df['Prediction'] = trainPredictions
		

		#Store the Tail Dataframes for analysis
		if c == 0:
			tailreport_df = tail_df
			chunkreportDf = chunk_df
		else:
			tailreport_df = pd.concat([tailreport_df, tail_df])
			if c % chunk_size == 0:
				chunkreportDf = pd.concat([chunkreportDf, chunk_df])
			
		
		T_Start_Day = f"{tail_df.iloc[0]['dateslice']} {tail_df.iloc[0]['day_of_week']}"
		T_End_Day = f"{tail_df.iloc[-1]['dateslice']} {tail_df.iloc[-1]['day_of_week']}"

		#print(f'Chunk: {c} ending at {T_End_Day}')
		print(f'Chunk: {c} for {stock} done')
			
	
	
	SQL_df = tailreport_df[['modelId', 'tailId', 'symbol', 'threshold', 'tailStart', 'dateslice', 'close', 'percent_change', 'Prediction']].copy()
	SQL_df.rename(index=str, columns={"dateslice": "datetime", "percent_change": "percentChange", "Prediction": "prediction"}, inplace=True)
	#print(list(SQL_df.columns.values))
	SQL_df['datetime'] = pd.to_datetime(SQL_df['datetime'])


	#Reg_df = tailreport_df[['modelId', 'tailId', 'symbol', 'threshold', 'tailStart', 'dateslice', 'close', 'percent_change', 'Prediction', 'Prediction_Regression']].copy()


	'''
	engine = create_engine(f"mysql://{dbc.user}:{dbc.pw}@{dbc.hostname}:{dbc.port}/{dbc.dbname}")

	
	#SQL_df.to_sql(name='tailBackTestDataframes', con=engine, if_exists = 'append', index=False)
	
	
	db = pymysql.connect(host=dbc.hostname,user=dbc.user,passwd=dbc.pw,db=dbc.dbname, port=dbc.port)
	cursor = db.cursor()
	sqlQuery = f"INSERT INTO modelRunTracker (modelId, datetime, symbol, threshold) VALUES ('{mId}', '{tailTime}', '{stock}', '{threshold}')"
	cursor.execute(sqlQuery)
	db.commit()
	'''
	
	print(f'Backtest for {stock} at threshold: {threshold} for features: {comboList} complete')
	
	with open('tailAccuracies.csv', 'w') as f:
		for key in tailAccuracies.keys():
			f.write("%s,%s\n"%(key,tailAccuracies[key]))

	with open('trainAccuracies.csv', 'w') as f:
		for key in trainAccuracies.keys():
			f.write("%s,%s\n"%(key,trainAccuracies[key]))

	
	featureImportanceDf = pd.DataFrame.from_dict(tailFeatImportances, orient='index')
	featureImportanceDf.to_csv('tailFeatGain.csv')
	'''
	featureImportanceDfReg = pd.DataFrame.from_dict(tailFeatImportancesReg, orient='index')
	featureImportanceDfReg.to_csv('tailFeatGainReg.csv')
	'''
	
	SQL_df.to_csv('backtest.csv')
	chunkreportDf.to_csv('chunksDf.csv')






def Initialize(file, thresholds, combos, candidates, validated, candidatesTime, timeRange, analysisFlags, analysisCandidates):


	fileName = file.split('\\')[-1]
	stock = fileName.split('_')[0]
	#print(len(combos), stock)

	warnings.filterwarnings(action='ignore', category=DeprecationWarning)
	tic=timeit.default_timer()
	group = pd.read_csv(file)
	toc=timeit.default_timer()
	print(f'Load complete for {stock} in {toc - tic} seconds.')

	
	#Sort by date
	group = group.sort_values(by=['Date'])
	#Set up a range of times
	group['DateSlice'] = group['Date']

	group['Date'] = pd.to_datetime(group['Date'])

	group['day_of_week'] = group['Date'].dt.weekday_name
	group = pd.concat([group, pd.get_dummies(group['day_of_week'])], axis=1)
	
	#Lets take time out for later
	group['time'] = group['Date'].dt.strftime('%H:%M')
	#Split hours and minutes, then convert it to minutes before making it numeric
	group['time'] = group['time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
	group['time'] = group['time'] - 570

	group['Date'] = group['Date'].dt.strftime('%Y-%m-%d')

	group.set_index(keys='Date', inplace=True)

	group.drop('2015-11-27', inplace=True)
	group.drop('2015-12-24', inplace=True)
	group.drop('2016-11-25', inplace=True)
	group.drop('2017-07-03', inplace=True)
	group.drop('2017-12-24', inplace=True)
	group.drop('2017-11-24', inplace=True)
	group.drop('2018-07-03', inplace=True)
	group.drop('2018-11-23', inplace=True)
	group.drop('2018-12-24', inplace=True)



	group['Date'] = group['DateSlice'] 
	group['Date'] = pd.DatetimeIndex(group['Date'])

	group.set_index(keys='Date', inplace=True)
	group = group['2015':'2019']

	group = group.between_time('9:29','16:00')

	#Retype into a stockstats-dataframe
	group = Sdf.retype(group)

	
	#///////////// This is for EoD style prediction
	group['close_eod'] = group.groupby(group.index.date)['close'].transform('last')
	group['percent_change'] = (100 * ((group['close_eod'] - group['close']) / group['close']))
	#///////////// This is for EoD style prediction^^^^^
	

	'''
	#Create a shifted closing value column as a testing target
	group[f'close_10'] = group['close']
	#shift it by a the forward time
	group[f'close_10'] = group[f'close_10'].shift(-10)
	
	group['percent_change'] = (100 * ((group[f'close_10'] - group['close']) / group['close']))
	'''
	

	



	#FEATURES
	tic=timeit.default_timer()
	###Overlap Indicators 
	if "close_SMA" in candidatesTime:
		for time in timeRange:
			group[f'close_SMA_{time}'] = talib.SMA(group['close'], time)

	#TEMA
	if "TEMA" in candidatesTime:
		for time in timeRange:
			group[f'TEMA_{time}'] = talib.TEMA(group['close'], time)
	
	#MA
	if "MA" in candidatesTime:
		for time in timeRange:
			group[f'MA_{time}'] = talib.MA(group['close'], time)

	#MA
	if "WMA" in candidatesTime:
		for time in timeRange:
			group[f'WMA_{time}'] = talib.WMA(group['close'], time)

	#MA
	if "TRIMA" in candidatesTime:
		for time in timeRange:
			group[f'TRIMA_{time}'] = talib.TRIMA(group['close'], time)

	
	#DEMA
	if "DEMA" in candidatesTime:
		for time in timeRange:
			group[f'DEMA_{time}'] = talib.DEMA(group['close'], time)
	
	#KAMA
	if "KAMA" in candidatesTime:
		for time in timeRange:  
			group[f'KAMA_{time}'] = talib.KAMA(group['close'], time)

	#CMO
	if "CMO" in candidatesTime:
		for time in timeRange:
			group[f'CMO_{time}'] = talib.CMO(group['close'], time)

	#AROONOSC
	if "AROONOSC" in candidatesTime:
		for time in timeRange:
			group[f'AROONOSC_{time}'] = talib.AROONOSC(group['high'], group['low'], time)

	#CCI
	if "CCI" in candidatesTime:
		for time in timeRange:
			group[f'CCI_{time}'] = talib.CCI(group['high'], group['low'], group['close'], time)

	#DX
	if "DX" in candidatesTime:
		for time in timeRange:
			group[f'DX_{time}'] = talib.DX(group['high'], group['low'], group['close'], time)

	#MFI
	if "MFI" in candidatesTime:
		for time in timeRange:
			group[f'MFI_{time}'] = talib.MFI(group['high'], group['low'], group['close'], group['volume'], time)

	#MINUS_DI
	if "MINUS_DI" in candidatesTime:
		for time in timeRange:
			group[f'MINUS_DI_{time}'] = talib.MINUS_DI(group['high'], group['low'], group['close'], time)

	#Momentum
	if "MOM" in candidatesTime:
		for time in timeRange:
			group[f'MOM_{time}'] = talib.MOM(group['close'], time)
	
	#ROCR100
	if "ROCR100" in candidatesTime:
		for time in timeRange:
			group[f'ROCR100_{time}'] = talib.ROCR100(group['close'], time)


	#PPO
	if "PPO" in candidatesTime:
		for time in timeRange:
			group[f'PPO_{time}'] = talib.PPO(group['close'], fastperiod=int(round(time/2)), slowperiod=time, matype=0)

	#WILLR
	if "WILLR" in candidatesTime:
		for time in timeRange:
			group[f'WILLR_{time}'] = talib.WILLR(group['high'], group['low'], group['close'], time)


	###Volume Indicators
	#AD
	if "AD" in candidates:
		group['AD'] = talib.AD(group['high'], group['low'], group['close'], group['volume'])


	#ADOSC
	if "ADOSC" in candidatesTime:
		for time in timeRange:
			group[f'ADOSC_{time}'] = talib.ADOSC(group['high'], group['low'], group['close'], group['volume'], fastperiod=int(round(time/2)), slowperiod=time)

	#ADOSC
	if "ULTOSC" in candidatesTime:
		for time in timeRange:
			group[f'ULTOSC_{time}'] = talib.ULTOSC(group['high'], group['low'], group['close'], timeperiod1=int(round(time/4)), timeperiod2=int(round(time/2)), timeperiod3=time)

	#OBV
	if "OBV" in candidates:
		group['OBV'] = talib.OBV(group['close'], group['volume'])


	###Volatility
	#ATR
	if "ATR" in candidatesTime:
		for time in timeRange:
			group[f'ATR_{time}'] = talib.ATR(group['high'], group['low'], group['close'], time)
		
	#NATR
	if "NATR" in candidatesTime:
		for time in timeRange:
			group[f'NATR_{time}'] = talib.NATR(group['high'], group['low'], group['close'], time)

	###Statistics
	#Correl Pearson's Correlation Coefficient (r)
	if "CORREL" in candidatesTime:
		for time in timeRange:
			group[f'CORREL_{time}'] = talib.CORREL(group['high'], group['low'], time)

	#Linear Reg 
	if "LINEARREG" in candidatesTime:
		for time in timeRange:
			group[f'LINEARREG_{time}'] = talib.LINEARREG(group['close'], time)

	#Linear Reg Intercept
	if "LINEARREG_INTERCEPT" in candidatesTime:
		for time in timeRange:
			group[f'LINEARREG_INTERCEPT_{time}'] = talib.LINEARREG_INTERCEPT(group['close'], time)

	#TSF
	if "TSF" in candidatesTime:
		for time in timeRange:
			group[f'TSF_{time}'] = talib.TSF(group['close'], time)

	#ADXR
	if "ADXR" in candidatesTime:
		for time in timeRange:
			group[f'ADXR_{time}'] = talib.ADXR(group['high'], group['low'], group['close'], time)

	#Bollinger Bands
	if "upperb" in candidatesTime or "midb" in candidatesTime or "lowerb" in candidatesTime:
		for time in timeRange:
			group[f'upperb_{time}'], group[f'midb_{time}'], group[f'lowerb_{time}'] = talib.BBANDS(group['close'], timeperiod=time, nbdevup=2, nbdevdn=2, matype=0)
	
	#MACD
	if "macd" in candidatesTime or "macdsignal" in candidatesTime or "macdhist" in candidatesTime:
		for time in timeRange:
			group[f'macd_{time}'], group[f'macdsignal_{time}'], group[f'macdhist_{time}'] = talib.MACD(group['close'], fastperiod=int(round(time/2)), slowperiod=time, signalperiod=int(round(time/3)))
			

	#ADX
	if "ADX" in candidatesTime:
		for time in timeRange:
			group[f'ADX_{time}'] = talib.ADX(group['high'], group['low'], group['close'], time)

	#MIDPRICE
	if "MIDPRICE" in candidatesTime:
		for time in timeRange:
			group[f'MIDPRICE_{time}'] = talib.MIDPRICE(group['high'], group['low'], time)


	#N Row Relative Strength Index
	if "RSI" in candidatesTime:
		for time in timeRange:
			group[f'RSI_{time}'] = talib.RSI(group['close'], time)
	
	if "RSI_Class" in candidatesTime:
		for time in timeRange:
			group[f'RSI_{time}'] = talib.RSI(group['close'], time)

		for time in timeRange:
			group[f'RSI_Class_{time}'] = group.apply(RSI_70_30_classifier, args=(f'RSI_{time}',), axis=1)

	#TRIX
	if "TRIX" in candidatesTime:
		for time in timeRange:
			group[f'TRIX_{time}'] = talib.TRIX(group['close'], time)
	#VAR
	if "VAR" in candidatesTime:
		for time in timeRange:
			group[f'VAR_{time}'] = talib.VAR(group['close'], timeperiod=time, nbdev=1)

	#BOP
	if "BOP" in candidates:
		group['BOP'] = talib.BOP(group['open'], group['high'], group['low'], group['close'])

	#HT_TRENDLINE
	if "HT_TRENDLINE" in candidates:
		group[f'HT_TRENDLINE'] = talib.HT_TRENDLINE(group['close'])

	
	#-------------------------------------

	#Build custom indicators here

	if "closeWindowTightComparison" in candidatesTime:
		for time in timeRange:
			group[f'closeWindowTightComparison_{time}'] = closeWindowTightComparison(group, time)

	if "closeWindowLooseComparison" in candidatesTime:
		for time in timeRange:
			group[f'closeWindowLooseComparison_{time}'] = closeWindowLooseComparison(group, time)

	if "volumeWindowComparison" in candidatesTime:
		for time in timeRange:
			group[f'volumeWindowComparison_{time}'] = volumeWindowComparison(group, time)

	if "voluminousDrop" in candidatesTime:
		for time in timeRange:
			group[f'voluminousDrop_{time}'] = voluminousDrop(group, time)

	if "voluminousRise" in candidatesTime:
		for time in timeRange:
			group[f'voluminousRise_{time}'] = voluminousRise(group, time)

	if "dailyCumulativePercentChange" in candidates:
		group['dailyCumulativePercentChange'] = dailyCumulativePercentChange(group)

	if "windowSumPercentChange" in candidatesTime:
		for time in timeRange:
			group[f'windowSumPercentChange_{time}'] = windowSumPercentChange(group, time)


	#-------------------------------------

	if analysisFlags:
		for operation in analysisFlags:
			for candidate in candidatesTime:
				timeCombos = combinations(timeRange, 2)
				for t in timeCombos:
					#eg. (5, 60)
					if operation == '/':
						group[f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'] = group[f'{candidate}_{t[0]}'] / group[f'{candidate}_{t[1]}']
					if operation == '*':
						group[f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'] = group[f'{candidate}_{t[0]}'] * group[f'{candidate}_{t[1]}']
					if operation == '+':
						group[f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'] = group[f'{candidate}_{t[0]}'] + group[f'{candidate}_{t[1]}']
					if operation == '-':
						group[f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'] = group[f'{candidate}_{t[0]}'] - group[f'{candidate}_{t[1]}']
					
					mask1 = group.loc[group[f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'] != npy.inf, f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'].max()
					group[f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'].replace(npy.inf,mask1,inplace=True)

					mask2 = group.loc[group[f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'] != npy.NINF, f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'].min()
					group[f'{candidate}_{t[0]}{operation}{candidate}_{t[1]}'].replace(npy.NINF,mask2,inplace=True)

					group.fillna(0, inplace=True)

			
	#group.to_csv('test1.csv')              
	
	

	#-------------------------------------
	#Internal Math Operations END
	#-------------------------------------

	#current predict_list = ['upperb_60', 'rsi_60', 'close60to3900', 'close_60_sma', 'close_180_sma', 'close_3900_sma', 'close_390_sma']
		



	#Cut off the top x rows used for back calculation (default 2 weeks = 3900)
	group = group.tail(len(group.index)-3900)

	#Create targets. These cannot be continuous. 
	for threshold in thresholds:
		group[f'Target_{threshold}'] = group.apply(target_classifier, args=(threshold,), axis=1)


	toc=timeit.default_timer()
	print(f'Features built for {stock} in {toc - tic} seconds.')
	chunk_size = 40

	#lock onto the first monday to start our weekly bracketing
	first_monday = str(group[group['day_of_week'] == 'Monday'].index[0])
	first_monday = first_monday[0:10]

	last_monday = str(group[group['day_of_week'] == 'Monday'].index[-1])
	last_monday = last_monday[0:10]
	
	print(f"{first_monday} to {last_monday}")
	idTracker = []
	
	for threshold in thresholds:
		for combination in combos:

			combinationList = list(combination)

			if combinationList != validated:
				combinationList.extend(validated)
			if combinationList == validated:
				print('Base model flag')

			flat_list = []
			for feature in combinationList: 
				if type(feature) is list:
					for item in feature:
						flat_list.append(item)
				else:
					flat_list.append(feature)
			combinationList = flat_list

			#Remove duplicate features
			combinationListPure = list(OrderedDict.fromkeys(combinationList))

			modelString = ""
			for feature in combinationListPure:
				modelString += str(feature) + "|"
			modelString = modelString[:-1]
			

			#Register the model if it does not exist yet
			'''
			db = pymysql.connect(host=dbc.hostname,user=dbc.user,passwd=dbc.pw,db=dbc.dbname, port=dbc.port)
			cursor = db.cursor()
			sqlQuery = f"INSERT IGNORE INTO modelId (features) VALUES ('{modelString}')"
			cursor.execute(sqlQuery)
			db.commit()


			cursor.execute(f"SELECT id FROM modelId WHERE features = '{modelString}'")
			mId = cursor.fetchone()[0]
			'''
			print(group.tail(390))
			mId = 999999999999999
			if mId not in idTracker:
				idTracker.append(mId)
				combination_test(stock, group, combinationListPure, chunk_size, first_monday, last_monday, threshold, mId)

	print(f"Total models tested for {stock}: {len(idTracker)}, there were {len(combos) - len(idTracker)} duplicate models")
			

def error_handler(e):
	print('error')
	print(dir(e), "\n")
	print("-->{}<--".format(e.__cause__))

def main():
	'''
	
	
	candidates = ['volume', 'BOP', 'AD', 'OBV', 'HT_TRENDLINE']
	candidatesTime = ['DEMA', 'KAMA', 'CMO', 'AROONOSC', 'CCI', 'DX', 'MFI', 'PPO', 'WILLR', 'ATR', 'NATR', 'ULTOSC', 'TRIX', 'VAR', 
	'LINEARREG', 'LINEARREG_INTERCEPT', 'CORREL', 'TSF', 'ADXR', 'MINUS_DI', 'MOM', 'RSI', 'upperb', 'lowerb', 'midb', 'close_SMA', 'ADX', 'TEMA', 'MA', 'MIDPRICE', 'WMA', 'TRIMA', 'macd', 'macdhist', 'macdsignal', 'ROCR100']
	'''

	#NOT WORKING
	#weSuck = ['ADOSC']

	#For candidates not dependent on timeRange (features that do not use time)
	candidates = ['dailyCumulativePercentChange']
	#For features that require timeRange
	'''
	candidatesTime = ['DEMA', 'KAMA', 'CMO', 'AROONOSC', 'CCI', 'DX', 'MFI', 'PPO', 'WILLR', 'ATR', 'NATR', 'ULTOSC', 'TRIX', 'VAR', 'MARTY', 'voluminousDrop',
	'LINEARREG', 'LINEARREG_INTERCEPT', 'CORREL', 'TSF', 'ADXR', 'MINUS_DI', 'MOM', 'RSI', 'upperb', 'lowerb', 'midb', 'close_SMA', 'ADX', 'TEMA', 'MA', 'MIDPRICE', 'WMA', 'TRIMA', 'macd', 'macdhist', 'macdsignal', 'ROCR100']
	'''
	candidatesTime = ['voluminousDrop', 'voluminousRise', 'volumeWindowComparison', 'windowSumPercentChange']
	#Set time ranges
	timeRange = [60, 1950]
	#For internal operations
	analysisFlags = []
	#means internal testing only (ex: RSI_60/RSI_120, but no RSI_60 and RSI_120 seperate)
	internalOnly = []
	#Specific features only?
	specificFeatures = ['dailyCumulativePercentChange', 'voluminousDrop_1950', 'voluminousRise_1950', 'volumeWindowComparison_1950', 'windowSumPercentChange_1950', 'windowSumPercentChange_60']

	#Validated/mandatory features (ONLY USE WITH SPECIFIC FEATURES ENABLED, AND MAKE SURE THEY ARE ADDED AS CANDIDATES/CANDIDATESTIME)
	validated = []
	#We need to attach time ranges to indicators in the candidatesTime bucket, then stick them with the other candidates
	for candidate in candidatesTime:
		if candidate not in internalOnly:
			for time in timeRange:

				candidates.append(candidate + "_" + str(time))


	analysisCandidates = []

	if analysisFlags:
		for operation in analysisFlags:
			if len(timeRange) >= 2:
				for candidate in candidatesTime:
					timeCombos = combinations(timeRange, 2)
					for t in timeCombos:
						#eg. (5, 60)
						analysisCandidates.append(f"{candidate}_{t[0]}{operation}{candidate}_{t[1]}")
						candidates.append(f"{candidate}_{t[0]}{operation}{candidate}_{t[1]}")

			else:
				print("ratio analysis impossible for 1 or fewer timeRange")

	
	if specificFeatures:
		candidates = specificFeatures       


	print(candidates)


	feature_combinations = []
	
	#This is for testing all unique combinations of validated + combinations of candidates
	#for i in range(len(candidates)):
	#   feature_combinations += combinations(candidates, i+1)
	
	feature_combinations += combinations(candidates, len(candidates))

	
	print(len(feature_combinations))
	print(feature_combinations)

	pool = multiprocessing.Pool() #use all available cores, otherwise specify the number you want as an argument
	for file in files:
		
		pool.apply_async(Initialize, args=(file, thresholds, feature_combinations, candidates, validated, candidatesTime, timeRange, analysisFlags, analysisCandidates), error_callback=error_handler)
		
		#Initialize(file, thresholds, feature_combinations, candidates, validated, candidatesTime, timeRange, analysisFlags, analysisCandidates)
				
				
	pool.close()
	pool.join()
	print(f'Finished at {datetime.datetime.now()}')
	


if __name__ == '__main__':
	main()