import pymysql
import boto3
import dbconfig as dbc
import numpy as np
import pandas as pd
import multiprocessing 
from multiprocessing.pool import ThreadPool
import os, sys, datetime, time, csv, math, glob
from queue import PriorityQueue
import random

#Define the number of stocks being used


files = glob.glob('C:\\Projects\\Stocks\\Trade Test Data\\*.csv')


#Controls
#Amount allowed to trade before balancing needed
shortLimit = 3
longLimit = 3

#Total limit of trades allowed per day
totalLimit = 25

#Time required between trades of the same stock
penaltyTimeout = 45

#Trades cannot start before min time, and cannot be placed after max time
minTime = 10
maxTime = 14

transactionList = []


shortQueue = PriorityQueue()
longQueue = PriorityQueue()

timeouts = {}


class transactionDef:
	def __init__(self, datetime, symbol, modelId, threshold, percentChange, prediction):
		self.datetime = datetime
		self.symbol = symbol
		self.modelId = modelId
		self.threshold = threshold
		self.percentChange = percentChange
		self.prediction = prediction

def error_handler(e):
	print('error')
	print(dir(e), "\n")
	print("-->{}<--".format(e.__cause__))




def transmitOrders():
	global shortLimit
	global longLimit
	global totalLimit

	global longQueue
	global shortQueue

	global timeouts
	global penaltyTimeout

	global transactionList

	if (not longQueue.empty()) or (not shortQueue.empty()):

		if not longQueue.empty():
			
			#get the predicted stock out of queue
			longOrderRaw = longQueue.get()
			#Grab the second part of this tuple; the first part is the priority, the second part is the order object with all the characteristics
			longOrder = longOrderRaw[1]

			#if we have available longs and trades, execute the trade and edit counters
			if longLimit > 0 and totalLimit > 0 and (longOrder.symbol not in timeouts or timeouts[longOrder.symbol] == 0):
				#Penalty Box
				timeouts[longOrder.symbol] = penaltyTimeout
				#print(f'Placing long, current number left : {longLimit}')
				transactionList.append([longOrder.datetime, longOrder.modelId, longOrder.symbol, longOrder.threshold, longOrder.percentChange, longOrder.prediction])
				longLimit -= 1
				shortLimit += 1
				totalLimit -= 1
				#print(f'Long placed, current number left : {longLimit}')


		if not shortQueue.empty():
			
			#get the predicted stock out of queue
			shortOrderRaw = shortQueue.get()
			#Grab the second part of this tuple; the first part is the priority, the second part is the order object with all the characteristics
			shortOrder = shortOrderRaw[1]

			#if we have available shorts and trades, execute the trade and edit counters
			if shortLimit > 0 and totalLimit > 0 and (shortOrder.symbol not in timeouts or timeouts[shortOrder.symbol] == 0):
				#Penalty Box
				timeouts[shortOrder.symbol] = penaltyTimeout
				#print(f'Placing short, current number left : {shortLimit}')
				transactionList.append([shortOrder.datetime, shortOrder.modelId, shortOrder.symbol, shortOrder.threshold, shortOrder.percentChange, shortOrder.prediction])
				shortLimit -= 1
				longLimit += 1
				totalLimit -= 1	
				#print(f'short placed, current number left : {shortLimit}')	





		#Call the function again until we're done clearing both queues
		transmitOrders()

def parseTails(file):
	global shortLimit
	global longLimit
	global totalLimit

	global longQueue
	global shortQueue

	global timeouts
	global penaltyTimeout

	df = pd.read_csv(file)

	new_headers = []

	for header in df.columns: # data.columns is your list of headers
		header = header.strip('"') # Remove the quotes off each header
		new_headers.append(header) # Save the new strings without the quotes

	df.columns = new_headers # Replace the old headers with the new list
	df['Date'] = pd.to_datetime(df['datetime'], errors ='coerce')
	df['Date'] = pd.DatetimeIndex(df['Date'])
	df.set_index(keys='Date', inplace=True)
	df['prediction'] = pd.to_numeric(df['prediction'])


	dfByDate = df.groupby(df.index.date)
	for date, dfDay in dfByDate:
		dfByHour = dfDay.groupby(dfDay.index.hour)
		longLimit = 3
		totalLimit = 25
		shortLimit = 3




		for hour, dfHour in dfByHour:
			dfByMinute = dfHour.groupby(dfHour.index.minute)
			
			for minute, dfMinute in dfByMinute:
				numStocks = len(dfMinute.index)
				priorities = random.sample(range(numStocks), numStocks)
				longCounter = 0
				shortCounter = 0

				for k, v in timeouts.items():
					if v > 0:
						timeouts[k] = v - 1	

				for row in dfMinute.itertuples(index=True):
					if row.prediction == 1 and row.Index.hour >= minTime and row.Index.hour < maxTime:
						transaction = transactionDef(row.Index, row.symbol, row.modelId, row.threshold, row.percentChange, row.prediction)
						longQueue.put((priorities[longCounter], transaction))
						longCounter += 1


					if row.prediction == -1 and row.Index.hour >= minTime and row.Index.hour < maxTime:
						transaction = transactionDef(row.Index, row.symbol, row.modelId, row.threshold, row.percentChange, row.prediction)
						shortQueue.put((priorities[shortCounter], transaction))
						shortCounter += 1

				transmitOrders()




	'''
	transactionDf = pd.DataFrame(transactionList, columns=['Date','modelId','tailId','symbol','percentChange'])		
	transactionDf.to_csv(f'transactionTest_{stock}.csv')
	'''




def main():
	global transactionList

	for f in files:
		parseTails(f)

	
	
	print(len(transactionList))	
	transactionDf = pd.DataFrame(transactionList, columns=['Date','modelId','tailId','symbol','percentChange','prediction'])	
	transactionDf.to_csv(f'transactionTest.csv')	
	print(f'Finished at {datetime.datetime.now()}')



if __name__ == '__main__':
	main()