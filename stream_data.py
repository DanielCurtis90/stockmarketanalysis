from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract as IBcontract

import time, csv, datetime, queue, pickle, threading, warnings
from threading import Thread
import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf
import xgboost as xgb

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler

from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from model_build import *
from buy_order import *
from historic_data import *

from vars import *

## marker for when queue is finished
FINISHED = object()
STARTED = object()
TIME_OUT = object()


class stock_Transaction:
	def __init__(self, close, action, orderID):
		self.close = close
		self.action = action
		self.orderID = orderID


class finishableQueue(object):

	def __init__(self, queue_to_finish):

		self._queue = queue_to_finish
		self.status = STARTED

	def get(self, timeout):
		"""
		Returns a list of queue elements once timeout is finished, or a FINISHED flag is received in the queue
		:param timeout: how long to wait before giving up
		:return: list of queue elements
		"""
		contents_of_queue=[]
		finished=False

		while not finished:
			try:
				current_element = self._queue.get(timeout=timeout)
				if current_element is FINISHED:
					finished = True
					self.status = FINISHED
				else:
					contents_of_queue.append(current_element)
					## keep going and try and get more data

			except queue.Empty:
				## If we hit a time out it's most probable we're not getting a finished element any time soon
				## give up and return what we have
				finished = True
				self.status = TIME_OUT


		return contents_of_queue

	def timed_out(self):
		return self.status is TIME_OUT


def _nan_or_int(x):
	if not np.isnan(x):
		return int(x)
	else:
		return x

class TestWrapper(EWrapper):
	"""
	The wrapper deals with the action coming back from the IB gateway or TWS instance
	We override methods in EWrapper that will get called when this action happens, like currentTime
	Extra methods are added as we need to store the results in this object
	"""
	def __init__(self):
		self._my_contract_details = {}
		self._my_market_data_dict = {}
		self._my_errors = queue.Queue()

	## error handling code
	def init_error(self):
		error_queue=queue.Queue()
		self._my_errors = error_queue

	def get_error(self, timeout=5):
		if self.is_error():
			try:
				return self._my_errors.get(timeout=timeout)
			except queue.Empty:
				return None

		return None

	def is_error(self):
		an_error_if=not self._my_errors.empty()
		return an_error_if

	def error(self, id, errorCode, errorString):
		## Overriden method
		errormsg = "IB error id %d errorcode %d string %s" % (id, errorCode, errorString)
		self._my_errors.put(errormsg)


	## get contract details code
	def init_contractdetails(self, reqId):
		contract_details_queue = self._my_contract_details[reqId] = queue.Queue()

		return contract_details_queue

	def contractDetails(self, reqId, contractDetails):
		## overridden method

		if reqId not in self._my_contract_details.keys():
			self.init_contractdetails(reqId)

		self._my_contract_details[reqId].put(contractDetails)

	def contractDetailsEnd(self, reqId):
		## overriden method
		if reqId not in self._my_contract_details.keys():
			self.init_contractdetails(reqId)

		self._my_contract_details[reqId].put(FINISHED)

	# market data
	def init_market_data(self, tickerid):
		market_data_queue = self._my_market_data_dict[tickerid] = queue.Queue()

		return market_data_queue


	def get_time_stamp(self):
		## Time stamp to apply to market data
		## We could also use IB server time
		return datetime.datetime.now()


	def realtimeBar(self, reqId, timep:int, openp:float, high:float, low:float, close:float, volume:int, wap:float, count:int):

		super().realtimeBar(reqId, timep, openp, high, low, close, volume, wap, count)

		stock = stock_rev_reqId_dict[reqId]

		time_seconds = time.strftime('%S', time.localtime(timep))
		time_hour = time.strftime('%H', time.localtime(timep))
		#time_minutely = time.strftime('%Y-%m-%d %H:%M', time.localtime(timep))
		time_minutely = time.strftime('%Y%m%d  %H:%M:00', time.localtime(timep))
		timep = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timep))
		
		stock_dict[stock].volume += volume

		if time_seconds == '00':
			stock_dict[stock].volume = volume
		

		bar_list = [timep, high, low, openp, close, volume]
		bar_dict = {'date' : time_minutely, 'high' : high, 'low' : low, 'open' : openp, 'close' : close, 'volume' : stock_dict[stock].volume}


		with open(f'logs/5_second_bars_{time.strftime("%Y-%m-%d", time.gmtime())}_{stock}.csv', 'a', newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			spamwriter.writerow(bar_list)

		if time_seconds == '55':
			#Load new minutely data into the dataframe
			bar_df = pd.DataFrame([bar_dict], columns=bar_dict.keys())
			bar_df['Date'] = pd.DatetimeIndex(bar_df['date'])
			bar_df.set_index(keys='Date', inplace=True)
			bar_df = Sdf.retype(bar_df)

			stock_dict[stock].dataframe = pd.concat([stock_dict[stock].dataframe, bar_df])
			stock_dict[stock].dataframe = Sdf.retype(stock_dict[stock].dataframe)

			long_output = use_Model(stock_dict[stock].dataframe.tail(3900), predict_list, stock_dict[stock].long_model)
			
			short_output = use_Model(stock_dict[stock].dataframe.tail(3900), predict_list, stock_dict[stock].short_model)

			if long_output == 1 and stock_dict[stock].lspriority != 'short_only':
				output = 1
			elif short_output == -1 and stock_dict[stock].lspriority != 'long_only':
				output = -1
			elif long_output == 1 and short_output == -1:
				if stock_dict[stock].lspriority == 'long' or stock_dict[stock].lspriority == 'long_only':
					output = 1
				if stock_dict[stock].lspriority == 'short' or stock_dict[stock].lspriority == 'short_only':
					output = -1
			else:
				output = 0

			#The amount of the stock we long/short
			stock_quantity = int(trade_amount / close)

			base_minute_time = datetime.datetime.strptime(time_minutely, '%Y%m%d  %H:%M:%S') 
			
			
			#How far ahead we exit the placed order
			shift_delta = datetime.timedelta(minutes=5)
			shifted_time = base_minute_time + shift_delta

			shifted_time = shifted_time.strftime("%Y%m%d 15:55:00")
			if stock_dict[stock].penalty < 1 and int(time_hour) < 15 and stock_dict[stock].count < 10:
				stock_order = Stock_Order(stock, stock_quantity, close, shifted_time, time_minutely)
				#If the output says do something, put the stock and its priority into queue.
				if output == 1:
					stock_long_q.put((stock_priority_dict[stock][0], stock_order))
				if output == -1:
					stock_short_q.put((stock_priority_dict[stock][1], stock_order))

			
			print(f'Longs in queue: {stock_long_q.queue}')
			print(f'Shorts in queue: {stock_short_q.queue}')

			bar_list2 = [time_minutely, high, low, openp, close, stock_dict[stock].volume, output]
			bar_list3 = [time_minutely, high, low, openp, close, stock_dict[stock].volume]

			#Dump into archive for constructed minutely data
			with open(f'logs/constructed_minutely_{time.strftime("%Y-%m-%d", time.gmtime())}_{stock}.csv', 'a', newline='') as csvfile2:
				spamwriter2 = csv.writer(csvfile2, delimiter=',')
				spamwriter2.writerow(bar_list2)
			#Append constructed minutely to the historical data
			with open(f'StockData/{stock}_historic_data.csv', 'a', newline='') as csvfile3:
				spamwriter3 = csv.writer(csvfile3, delimiter=',')
				spamwriter3.writerow(bar_list3)

			global transmit_counter
			global transmit_counter_end

			transmit_counter += 1
			print(transmit_counter)
			if transmit_counter == transmit_counter_end:
				transmit_counter = 0
				transmit_orders()
				
		

class TestClient(EClient):
	"""
	The client method
	We don't override native methods, but instead call them from our own wrappers
	"""
	def __init__(self, wrapper):
		## Set up with a wrapper inside
		EClient.__init__(self, wrapper)

		self._market_data_q_dict = {}

	def resolve_ib_contract(self, ibcontract, reqId=DEFAULT_GET_CONTRACT_ID):

		"""
		From a partially formed contract, returns a fully fledged version
		:returns fully resolved IB contract
		"""

		## Make a place to store the data we're going to return
		contract_details_queue = finishableQueue(self.init_contractdetails(reqId))

		print("Getting full contract details from the server... ")

		self.reqContractDetails(reqId, ibcontract)

		## Run until we get a valid contract(s) or get bored waiting
		MAX_WAIT_SECONDS = 10
		new_contract_details = contract_details_queue.get(timeout = MAX_WAIT_SECONDS)

		while self.wrapper.is_error():
			print(self.get_error())

		if contract_details_queue.timed_out():
			print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

		if len(new_contract_details)==0:
			print("Failed to get additional contract details: returning unresolved contract")
			return ibcontract

		if len(new_contract_details)>1:
			print("got multiple contracts using first one")

		new_contract_details=new_contract_details[0]

		resolved_ibcontract=new_contract_details.summary

		return resolved_ibcontract


	def start_getting_IB_market_data(self, resolved_ibcontract, tickerid=DEFAULT_MARKET_DATA_ID):
		"""
		Kick off market data streaming
		:param resolved_ibcontract: a Contract object
		:param tickerid: the identifier for the request
		:return: tickerid
		"""

		self._market_data_q_dict[tickerid] = self.wrapper.init_market_data(tickerid)
		self.reqRealTimeBars(tickerid, resolved_ibcontract, "", 'TRADES', 1, [])

		return tickerid

	def stop_getting_IB_market_data(self, tickerid):
		"""
		Stops the stream of market data and returns all the data we've had since we last asked for it
		:param tickerid: identifier for the request
		:return: market data
		"""

		## native EClient method
		self.cancelRealTimeBars(tickerid)

		## Sometimes a lag whilst this happens, this prevents 'orphan' ticks appearing
		time.sleep(5)

		market_data = self.get_IB_market_data(tickerid)

		## output ay errors
		while self.wrapper.is_error():
			print(self.get_error())

		return market_data

	def get_IB_market_data(self, tickerid):
		"""
		Takes all the market data we have received so far out of the stack, and clear the stack
		:param tickerid: identifier for the request
		:return: market data
		"""

		## how long to wait for next item
		MAX_WAIT_MARKETDATEITEM = 5
		market_data_q = self._market_data_q_dict[tickerid]

		market_data=[]
		finished=False

		while not finished:
			try:
				market_data.append(market_data_q.get(timeout=MAX_WAIT_MARKETDATEITEM))
			except queue.Empty:
				## no more data
				finished=True

		return market_data


class TestApp(TestWrapper, TestClient):
	def __init__(self, ipaddress, portid, clientid):
		TestWrapper.__init__(self)
		TestClient.__init__(self, wrapper=self)

		self.connect(ipaddress, portid, clientid)

		thread = Thread(target = self.run)
		thread.start()

		setattr(self, "_thread", thread)

		self.init_error()



def stream_data_thread(stock):
	print(f'Activating data stream for {stock}')
	app = TestApp("127.0.0.1", 4001, stock_dict[stock].clientID)

	stock_dict[stock].volume = 0
	stock_dict[stock].long_model = pickle.load(open(f"models/{stock}_long_{stock_dict[stock].thresholds[0]}_model.pickle.dat", "rb"))
	stock_dict[stock].short_model = pickle.load(open(f"models/{stock}_short_{stock_dict[stock].thresholds[1]}_model.pickle.dat", "rb"))
	FMT = '%Y%m%d  %H:%M:%S'

	#Make sure the historical data has been gleaned before running the datastream!!!

	stock_dict[stock].dataframe = pd.read_csv(f'StockData/{stock}_historic_data.csv')
	
	#Set up a range of times
	stock_dict[stock].dataframe['DateRemoval'] = stock_dict[stock].dataframe['Date']
	stock_dict[stock].dataframe['DateRemoval'] = pd.to_datetime(stock_dict[stock].dataframe['DateRemoval'])
	stock_dict[stock].dataframe['DateRemoval'] = stock_dict[stock].dataframe['DateRemoval'].dt.strftime('%Y-%m-%d')

	stock_dict[stock].dataframe.set_index(keys='DateRemoval', inplace=True)
	
	#Drop holidays here:
	stock_dict[stock].dataframe.drop('2018-07-03', inplace=True)



	stock_dict[stock].dataframe['Date'] = pd.DatetimeIndex(stock_dict[stock].dataframe['Date'])
	stock_dict[stock].dataframe.set_index(keys='Date', inplace=True)
	
	#Retype into a stockstats-dataframe
	stock_dict[stock].dataframe = Sdf.retype(stock_dict[stock].dataframe)

	print(f"Historical dataframe loaded for {stock}")

	#stock_dict[stock].dataframe.to_csv(f'check_{stock}.csv')

	#Need to keep track of orders placed
	transaction_header = ['Stock', 'Price', 'Amount', 'Exit Time', 'Order ID', 'Prediction', 'Time Placed']
	with open(f'logs/transactions_{time.strftime("%Y-%m-%d", time.gmtime())}.csv', 'w', newline='') as csvfile:
				headwriter = csv.writer(csvfile, delimiter=',')
				headwriter.writerow(transaction_header)

	## lets get prices for this
	ibcontract = IBcontract()
	ibcontract.secType = "STK"
	ibcontract.symbol = stock
	ibcontract.currency= "USD"
	ibcontract.exchange= "ISLAND"


	## create new recording file
	header = ['date', 'high', 'low', 'open', 'close', 'volume']
	header2 = ['date', 'high', 'low', 'open', 'close', 'volume', 'prediction']

	with open(f'logs/5_second_bars_{time.strftime("%Y-%m-%d", time.gmtime())}_{stock}.csv', 'w', newline='') as csvfile:
				spamwriter = csv.writer(csvfile, delimiter=',')
				spamwriter.writerow(header)

	with open(f'logs/constructed_minutely_{time.strftime("%Y-%m-%d", time.gmtime())}_{stock}.csv', 'w', newline='') as csvfile2:
				spamwriter2 = csv.writer(csvfile2, delimiter=',')
				spamwriter2.writerow(header2)

	## resolve the contract
	resolved_ibcontract = app.resolve_ib_contract(ibcontract, reqId=stock_dict[stock].reqID)

	tickerid = app.start_getting_IB_market_data(resolved_ibcontract, tickerid=stock_dict[stock].tickerID)

	time.sleep(25000)
	print(f'Ending data stream for {stock}')
	app.disconnect()

'''
def my_listener(event):
	if event.exception:
		print('Data stream failed, exception in log.')
	else:
		print('Data stream executed.')
'''


def transmit_orders():
	print("Transmit module running")
	
	if (not stock_long_q.empty()) or (not stock_short_q.empty()):
		if not stock_long_q.empty():
			long_order_raw = stock_long_q.get()
			long_order = long_order_raw[1]

			long_process_Thread = threading.Thread(target=long_short_order, args=(1, long_order.exit_time, long_order.name, long_order.quantity, long_order.price, long_order.placed_time))
			long_process_Thread.start()
			#Penalty Box
			stock_dict[long_order.name].penalty += 1
			stock_dict[long_order.name].count += 1
			if stock_dict[long_order.name].penalty == 1 and stock_dict[long_order.name].count < 10:
				threading.Thread(target=penalty_box, args=(long_order.name,)).start()
				print(f'{long_order.name} disabled for 20 minutes')

		if not stock_short_q.empty():
			short_order_raw = stock_short_q.get()
			short_order = short_order_raw[1]

			short_process_Thread = threading.Thread(target=long_short_order, args=(-1, short_order.exit_time, short_order.name, short_order.quantity, short_order.price, short_order.placed_time))
			short_process_Thread.start()
			#Penalty Box
			stock_dict[short_order.name].penalty += 1
			stock_dict[short_order.name].count += 1
			if stock_dict[short_order.name].penalty == 1 and stock_dict[short_order.name].count < 10:
				threading.Thread(target=penalty_box, args=(short_order.name,)).start()
				print(f'{short_order.name} disabled for 20 minutes')

		#Call the function again until we're done clearing both queues
		transmit_orders()


	#For the long-short pairing system
	#Every minute we will go down the long queue until it is empty. Empty both the long and short queue once we reach the end of either.
	'''
	if (not stock_long_q.empty()) and (not stock_short_q.empty()):
		long_order_raw = stock_long_q.get()
		short_order_raw = stock_short_q.get()

		long_order = long_order_raw[1]
		short_order = short_order_raw[1]
		
		print(f'Placing opposing orders; long: {long_order.name}, short: {short_order.name}.')

		long_process_Thread = threading.Thread(target=long_short_order, args=(1, long_order.exit_time, long_order.name, long_order.quantity, long_order.price, long_order.placed_time))
		long_process_Thread.start()
		#Penalty Box
		stock_dict[long_order.name].penalty += 1
		if stock_dict[long_order.name].penalty == 1:
			threading.Thread(target=penalty_box, args=(long_order.name,)).start()
			print(f'{long_order.name} disabled for one hour')

		short_process_Thread = threading.Thread(target=long_short_order, args=(-1, short_order.exit_time, short_order.name, short_order.quantity, short_order.price, short_order.placed_time))
		short_process_Thread.start()
		#Penalty Box
		stock_dict[short_order.name].penalty += 1
		if stock_dict[short_order.name].penalty == 1:
			threading.Thread(target=penalty_box, args=(short_order.name,)).start()
			print(f'{short_order.name} disabled for ten minutes')


		#Call the function again until we're done clearing both queues
		transmit_orders()
	
	#If either queue is empty, clear both.
	if (stock_long_q.empty()) or (stock_short_q.empty()):
		stock_short_q.queue.clear()
		stock_long_q.queue.clear()
		print(f'Both queues cleared {str(datetime.datetime.now())}')
	'''

def prepare_data_models(stock):
	#Call function to get the historic data and build the model
	historic_data_harvest(stock)

def penalty_box(stock):
	time.sleep(1200)
	stock_dict[stock].penalty = 0
	print(f'{stock} re-enabled')


def main():
	executors = {
    'default': ThreadPoolExecutor(50),   # max threads: 90
    'processpool': ProcessPoolExecutor(20)  # max processes 20
	}

	sched = BlockingScheduler(executors=executors)
	#backsched = BackgroundScheduler()
	
	for stock in stock_list:
		sched.add_job(prepare_data_models, 'cron', args=[stock], misfire_grace_time=10, hour=21, minute=40)

		#sched.add_job(prepare_data_models, 'cron', args=[stock], day_of_week='sat', misfire_grace_time=10, hour=12, minute=1)

		
		#sched.add_job(stream_data_thread, 'cron', args=[stock], day_of_week='mon-fri', misfire_grace_time=10, max_instances=10, hour=9, minute=18)
		
		
		#day_of_week='mon-fri'

	sched.start()        # start the scheduler

   
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
print('Waiting to start')
main()