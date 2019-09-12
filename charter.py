import os, sys, pickle, datetime, time, csv, math
import plotly, pymysql, visuals
import plotly.graph_objs as go
import pandas as pd
import multiprocessing


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
from xgboost import plot_importance
from stockstats import StockDataFrame as Sdf
from itertools import combinations, chain
from csvgenerator import generate_csvs

def Train(stock, dpast, fwdtime, data_path, thresholds):
	
	combo_tail_summary = []

	for threshold in thresholds:


		group = pd.read_csv(f'{data_path}{stock}_historic_data.csv')
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


		#Weird duplicate days
		group.drop('2015-12-18', inplace=True)
		group.drop('2016-03-18', inplace=True)
		group.drop('2017-04-07', inplace=True)
		group.drop('2018-03-23', inplace=True)

		group['Date'] = group['DateSlice'] 
		group['Date'] = pd.DatetimeIndex(group['Date'])

		group.set_index(keys='Date', inplace=True)

		group = group['2015':'2019']
		
		group = group.between_time('9:29','16:00')
		

		#Retype into a stockstats-dataframe
		group = Sdf.retype(group)
		#Verified parameters (Tested and are measureably better than their peers)

		#N Day Simple Moving Average
		'''
		group[f'close_5_sma']
		group[f'close_15_sma']
		group[f'close_30_sma']
		'''
		group[f'close_60_sma']
		group[f'close_390_sma']
		group[f'close_3900_sma']
		group[f'close_180_sma']
		'''
		group[f'close_1950_sma']
		group[f'close_7800_sma']
		group[f'close_78000_sma']
		group[f'close_39000_sma']
		group[f'close_780_sma']
		group[f'close_19500_sma']
		'''


		#N Row Relative Strength Index
		group['RSI_60'] = talib.RSI(group['close'], 60)
		group['RSI_120'] = talib.RSI(group['close'], 120)
		group['RSI_30'] = talib.RSI(group['close'], 30)
		group['RSI_15'] = talib.RSI(group['close'], 15)
		group['RSI_5'] = talib.RSI(group['close'], 5)
		group['RSI_390'] = talib.RSI(group['close'], 390)
		group['RSI_1950'] = talib.RSI(group['close'], 1950)

		#Bollinger Bands
		group['upperb_60'], group['midb_60'], group['lowerb_60'] = talib.BBANDS(group['close'], timeperiod=60, nbdevup=2, nbdevdn=2, matype=0)
		group['upperb_120'], group['midb_120'], group['lowerb_120'] = talib.BBANDS(group['close'], timeperiod=120, nbdevup=2, nbdevdn=2, matype=0)
		group['upperb_390'], group['midb_390'], group['lowerb_390'] = talib.BBANDS(group['close'], timeperiod=390, nbdevup=2, nbdevdn=2, matype=0)
		group['upperb_30'], group['midb_30'], group['lowerb_30'] = talib.BBANDS(group['close'], timeperiod=30, nbdevup=2, nbdevdn=2, matype=0)
		group['upperb_1950'], group['midb_1950'], group['lowerb_1950'] = talib.BBANDS(group['close'], timeperiod=1950, nbdevup=2, nbdevdn=2, matype=0)
		group['upperb_3900'], group['midb_3900'], group['lowerb_3900'] = talib.BBANDS(group['close'], timeperiod=3900, nbdevup=2, nbdevdn=2, matype=0)
		group['upperb_15'], group['midb_15'], group['lowerb_15'] = talib.BBANDS(group['close'], timeperiod=15, nbdevup=2, nbdevdn=2, matype=0)


		#ADX
		'''
		group['ADX_60'] = talib.ADX(group['high'], group['low'], group['close'], 60)
		group['ADX_120'] = talib.ADX(group['high'], group['low'], group['close'], 120)
		group['ADX_390'] = talib.ADX(group['high'], group['low'], group['close'], 390)
		group['ADX_1950'] = talib.ADX(group['high'], group['low'], group['close'], 1950)
		'''

		#TEMA
		'''
		group['TEMA_60'] = talib.TEMA(group['close'], 60)
		group['TEMA_120'] = talib.TEMA(group['close'], 120)

		group['TEMA_390'] = talib.TEMA(group['close'], 390)
		group['TEMA_3900'] = talib.TEMA(group['close'], 3900)
		'''

		#-------------------------------------
		#CUSTOM INDICATORS
		#Build custom indicators here
		#-------------------------------------


		group['tick'] = group.close.diff()
		

		#-------------------------------------
		#CUSTOM INDICATORS END
		#-------------------------------------

		group['close60to3900'] = group['close_60_sma'] / group['close_3900_sma']
		group['close390to3900'] = group['close_390_sma'] / group['close_3900_sma']
		group['close180to390'] = group['close_180_sma'] / group['close_390_sma']
		group['close180to3900'] = group['close_180_sma'] / group['close_3900_sma']

		#validated_predictors = ['boll_ub', 'boll_lb', 'close60to3900', 'close180to3900', 'close390to3900', 'close180to390']
		
		validated_predictors = ['RSI_60', 'close60to3900', 'close_60_sma', 'close_180_sma', 'close_3900_sma', 'close_390_sma', 'time']

		candidates = ['upperb_60']


		basic_info = ['volume', 'close_12_ema', 'close_26_ema', 'boll', 'close_10_sma', 'close_20_sma', 'close_20_mstd', "Target", "symbol", "time", "open", "close", "high", "low", f'close_{fwdtime}', 'close_eod', 'percent_change']
		#basic_info = ['close_180_sma', 'close_3900_sma', 'close_390_sma', 'close_60_sma', 'close_12_ema', 'close_26_ema', 'boll', 'close_10_sma', 'close_20_sma', 'close_20_mstd', "Target", "symbol", "time", "open", "close", "high", "low", f'close_{fwdtime}', 'close_eod', 'percent_change']

		#DMA, difference of 10 and 50 moving average
		#group['dma']

		#Testing/Unused Parameters

		#N Day Williams %R
		#group[f'wr_{dpast}']
		
		#N Day Stochastic Oscillator
		#group[f'kdjk_{dpast}']
		#group[f'kdjd_{dpast}']
		#group[f'kdjj_{dpast}']
		
		
		#MACD and friends
		#group['macd']
		#group['macds']
		

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

		#group.to_csv(f'group.csv', index=True)



		'''
		#This is for time horizon prediction
		#Create a shifted closing value column as a testing target
		group[f'close_{fwdtime}'] = group['close']

		#shift it by a the forward time
		group[f'close_{fwdtime}'] = group[f'close_{fwdtime}'].shift(-fwdtime)
		
		group['percent_change'] = (100 * ((group[f'close_{fwdtime}'] - group['close']) / group['close']))
		'''

		#Cut off the top x rows used for back calculation
		group = group.tail(len(group.index)-3900)

		#Drop any NaN rows
		group = group.dropna()

		#Create targets. These cannot be continuous. 
		


		group['Target'] = group.apply(target_classifier, args=(threshold,), axis=1)

		#///Fit a rudimentary decision tree///
		#Choose all predictors 
		#predictors = [x for x in group.columns if x not in chain(basic_info, validated_predictors)]

		# Build all possible combinations of predictors
		feature_combinations = []
		#Use predictors if doing weird bulk test stuff
		for i in range(len(candidates)):
			feature_combinations += combinations(candidates, i+1)
		'''
		#For time shifted eod only
		if fwdtime == 60:
			group = group.between_time('9:30','14:59')
		elif fwdtime == 120:
			group = group.between_time('9:30','13:59')
		elif fwdtime == 180:
			group = group.between_time('9:30','12:59')
		'''
		#feature_combinations += [validated_predictors]
		print(f"Processing {len(feature_combinations)} different models.")
		
		#chunk_size is the size of the dataframe used to make the model
		#chunk_mover is the time interval that we use the prior built model on (the tail)
		'''
		#For time shifted eod only
		day_hours = 16 - (9.5 + (fwdtime/60))
		'''
		chunk_size = 20

		#lock onto the first monday to start our weekly bracketing
		first_monday = str(group[group['day_of_week'] == 'Monday'].index[0])
		first_monday = first_monday[0:10]

		last_monday = str(group[group['day_of_week'] == 'Monday'].index[-1])
		last_monday = last_monday[0:10]
		
		print(f"{first_monday} to {last_monday}")
		
		r_df = []
		
		for comb in feature_combinations:
			#ADD/INITIALIZE THE AVERAGE RATINGS HERE!!!
			summary_average_tail_value = 0
			summary_average_tail_short_percent = 0
			summary_average_tail_long_percent = 0
			summary_average_tail_short_num = 0
			summary_average_tail_long_num = 0
			summary_average_tail_short_value = 0
			summary_average_tail_long_value = 0

			summary_average_full_long_average = 0
			summary_average_full_short_average = 0
			summary_average_full_long_count = 0
			summary_average_full_short_count = 0

			summary_byhour_long_total = {}
			summary_byhour_long_count = {}
			summary_byhour_short_total = {}
			summary_byhour_short_count = {}


			first_day = datetime.datetime.strptime(first_monday, "%Y-%m-%d")
			last_day = datetime.datetime.strptime(last_monday, "%Y-%m-%d")
			
			#So this is the last monday + 6 to get to the last sunday (last actual day in dataframe) + 1 more day due to off by one rounding for floor division.
			last_day = last_day + datetime.timedelta(days = 7)

			#Let's count the amount of weeks present between the first and last mondays
			num_chunks = ((last_day - first_day).days // 7) - chunk_size
			print(num_chunks)

			average_nonzero_counter_longs = num_chunks
			average_nonzero_counter_shorts = num_chunks


			for c in range(num_chunks):
				#/// REVISED TAIL/CHUNK SELECTION
				#We need chunksize number of weeks (default 20) to calculate the model, so we need to bump down chunksize weeks from our first monday
				move_weeks = chunk_size + c
				week_delta = datetime.timedelta(weeks = move_weeks)
				first_day = datetime.datetime.strptime(first_monday, "%Y-%m-%d")
				tail_firstday = first_day + week_delta
				tail_lastday = tail_firstday + datetime.timedelta(days = 6)

				tail_df = group.loc[str(tail_firstday):str(tail_lastday)]
				tail_target = tail_df['Target']

				#The chunk will be constructed similarly to the tail minus chunksize weeks. The last day of the chunk will be the sunday right before the starting monday of the tail.
				chunk_firstday = first_day + datetime.timedelta(weeks = c)
				chunk_lastday = tail_firstday - datetime.timedelta(days = 1)
				chunk_df = group.loc[str(chunk_firstday):str(chunk_lastday)]
				target = chunk_df['Target']

				#Grab the columns selected by the combination
				list_comb = list(comb)
				list_comb.extend(validated_predictors)
				X_Data = chunk_df[list_comb]
				#Set up the training and testing fractions
				x_train, x_test, y_train, y_test = train_test_split(X_Data, target, test_size=0, random_state=0, shuffle=False) #shuffle=False for non-random
				model = xgb.XGBClassifier()
				model.fit(x_train, y_train)

				#make predictions for training data and evaluate (check for overfitting)
				train_y_pred = model.predict(x_train)
				train_predictions = [round(value) for value in train_y_pred]
				train_accuracy = accuracy_score(y_train, train_predictions)

				#Test the tail
				tail_x = tail_df[list_comb]
				tail_y_pred = model.predict(tail_x)
				tail_predictions = [round(value) for value in tail_y_pred]
				tail_accuracy = accuracy_score(tail_target, tail_predictions)
				tail_df['Prediction'] = tail_predictions
				
				df_taillist = tail_df.groupby(['Prediction'])
				
				#if c == (num_chunks - 1) or c == (num_chunks // 2):

				'''
				tail_df.to_csv(f'{stock}_{threshold}_tail_df_{c}.csv')
				chunk_df.to_csv(f'{stock}_{threshold}_chunk_df_{c}.csv')
				'''

				#initialize
				long_tail_accuracy = 0
				number_of_tail_longs = 0
				percent_tail_longs = 0
				short_tail_accuracy = 0
				number_of_tail_shorts = 0
				percent_tail_shorts = 0
				tail_bad_longs = 0
				tail_bad_shorts = 0
				tail_average_long_percent = 0
				tail_average_short_percent = 0

				#assign
				for name, sub in df_taillist:
					if name == 1:
						long_tail_accuracy = accuracy_score(sub['Target'], sub['Prediction'])
						number_of_tail_longs = len(sub.index)
						percent_tail_longs = (100 * len(sub.index) / len(tail_df.index))
						tail_average_long_percent = sub['percent_change'].sum() / number_of_tail_longs
						summary_average_tail_long_percent = summary_average_tail_long_percent + tail_average_long_percent
						
						if number_of_tail_longs == 0:
							average_nonzero_counter_longs -= 1

						summary_average_full_long_average += sub['percent_change'].sum()
						summary_average_full_long_count += len(sub.index)

						sub_group = sub.groupby(sub.index.strftime('%H:%M'))
						for subname, subsub in sub_group:
							if not subname in summary_byhour_long_total:
								summary_byhour_long_total[subname] = subsub['percent_change'].sum()
							else:
								summary_byhour_long_total[subname] += subsub['percent_change'].sum()

							if not subname in summary_byhour_long_count:
								summary_byhour_long_count[subname] = len(subsub.index)
							else:
								summary_byhour_long_count[subname] += len(subsub.index)


					if name == -1:
						short_tail_accuracy = accuracy_score(sub['Target'], sub['Prediction'])
						number_of_tail_shorts = len(sub.index)
						percent_tail_shorts = (100 * len(sub.index) / len(tail_df.index))
						tail_average_short_percent = sub['percent_change'].sum() / number_of_tail_shorts
						summary_average_tail_short_percent = summary_average_tail_short_percent + tail_average_short_percent
						

						if number_of_tail_shorts == 0:
							average_nonzero_counter_shorts -= 1

						summary_average_full_short_average += sub['percent_change'].sum()
						summary_average_full_short_count += len(sub.index)

						sub_group = sub.groupby(sub.index.strftime('%H:%M'))
						for subname, subsub in sub_group:
							if not subname in summary_byhour_short_total:
								summary_byhour_short_total[subname] = subsub['percent_change'].sum()
							else:
								summary_byhour_short_total[subname] += subsub['percent_change'].sum()

							if not subname in summary_byhour_short_count:
								summary_byhour_short_count[subname] = len(subsub.index)
							else:
								summary_byhour_short_count[subname] += len(subsub.index)

				
				tail_adj_long = (tail_average_long_percent * number_of_tail_longs) / 100
				tail_adj_short = (tail_average_short_percent * number_of_tail_shorts) / 100
				tail_value_rating = (tail_adj_long - tail_adj_short)
				
				summary_average_tail_short_num = summary_average_tail_short_num + number_of_tail_shorts
				summary_average_tail_long_num = summary_average_tail_long_num + number_of_tail_longs
				summary_average_tail_value = summary_average_tail_value + tail_value_rating
				summary_average_tail_short_value = summary_average_tail_short_value + tail_adj_short
				summary_average_tail_long_value = summary_average_tail_long_value + tail_adj_long

				feat_imp = model.feature_importances_
				T_Start_Day = f"{tail_df.iloc[0]['dateslice']} {tail_df.iloc[0]['day_of_week']}"
				T_End_Day = f"{tail_df.iloc[-1]['dateslice']} {tail_df.iloc[-1]['day_of_week']}"
				print(T_Start_Day)
				print(T_End_Day)

				d = {
				'Name' : str(list_comb) + "_Tail" + str(tail_firstday),
				'T Short Acc' : (short_tail_accuracy * 100.0),
				'T Long Acc' : (long_tail_accuracy * 100.0),
				'T Num Shorts' : number_of_tail_shorts,	
				'T Num Longs' : number_of_tail_longs,	
				'T Average Short Loss' : tail_average_short_percent,
				'T Average Long Gain' : tail_average_long_percent,
				'T Short Rating' : tail_adj_short,
				'T Long Rating' : tail_adj_long,
				'T Start Day' : T_Start_Day,
				'T End Day' : T_End_Day,
				'T Value Rating' : tail_value_rating
				}

				for counter, feature in enumerate(list_comb):
					d[feature] = feat_imp[counter]
				r_df.append(d)

			summary_average_tail_value = summary_average_tail_value / num_chunks

			summary_average_tail_short_percent = summary_average_tail_short_percent / num_chunks
			summary_average_tail_long_percent = summary_average_tail_long_percent / num_chunks

			summary_average_tail_short_num = summary_average_tail_short_num / num_chunks
			summary_average_tail_long_num = summary_average_tail_long_num / num_chunks
			summary_average_tail_short_value = summary_average_tail_short_value / num_chunks
			summary_average_tail_long_value = summary_average_tail_long_value / num_chunks

			if summary_average_full_long_count != 0:
				summary_average_full_long_average = summary_average_full_long_average / summary_average_full_long_count
			else:
				summary_average_full_long_average = 0


			if summary_average_full_short_count != 0:
				summary_average_full_short_average = summary_average_full_short_average / summary_average_full_short_count
			else:
				summary_average_full_short_average = 0 
			
			summary = {
			'Threshold' : threshold,
			'Model' : str(list_comb),
			'Average Tail Value Rating' : summary_average_tail_value,
			'Average Tail Short Loss' : summary_average_tail_short_percent,
			'Average Tail Long Gain' : summary_average_tail_long_percent,
			'Average Num Tail Shorts' : summary_average_tail_short_num,
			'Average Num Tail Longs' : summary_average_tail_long_num,
			'Average Long Gain Overall' : summary_average_full_long_average,
			'Average Short Loss Overall' : summary_average_full_short_average,
			'Average Tail Short Value' : summary_average_tail_short_value,
			'Average Tail Long Value' : summary_average_tail_long_value
			}
			combo_tail_summary.append(summary)


		r_df = pd.DataFrame(r_df)
		r_df.to_csv(f'results_{fwdtime}_{stock}_{threshold}.csv', index=False)


		long_profile = pd.DataFrame.from_dict(summary_byhour_long_total, orient = 'index')
		long_profile_count = pd.DataFrame.from_dict(summary_byhour_long_count, orient = 'index')

		short_profile = pd.DataFrame.from_dict(summary_byhour_short_total, orient = 'index')
		short_profile_count = pd.DataFrame.from_dict(summary_byhour_short_count, orient = 'index')

		profiling_df = pd.concat([long_profile, long_profile_count, short_profile, short_profile_count], axis=1)

		profiling_df.to_csv(f'profile_{fwdtime}_{stock}_{threshold}.csv')

		

	combo_tail_summary = pd.DataFrame(combo_tail_summary)
	combo_tail_summary = combo_tail_summary.sort_values(by=['Threshold'], ascending=True)
	combo_tail_summary.to_csv(f'tail_summary_{fwdtime}_{stock}.csv', index=False)
		

