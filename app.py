from flask import Flask, request, render_template, jsonify, redirect, flash, Response
from werkzeug.utils import secure_filename
import shutil
import warnings

import pickle
import threading
import multiprocessing
from charter import *
from vars import *

app = Flask(__name__)

open_path = "C:/Users/Administrator/Desktop/NASDAQ/Day"
save_path = "C:/Projects/Stocks/Testing Data/"
fwdtime = [60]
thresholds = [1.0]
#thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#thresholds = [0.9, 1.0]

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


#For multiple stock/threshold multiprocessing
'''
if __name__ == '__main__':
	pool = multiprocessing.Pool() #use all available cores, otherwise specify the number you want as an argument
	for stock in stock_list:
		for time in fwdtime:
			pool.apply_async(Train, args=(stock, 60, time, save_path, thresholds))
	pool.close()
	pool.join()
'''

#Use if testing on low power EC2 (each test runs in series)
if __name__ == '__main__':
	for stock in stock_list:
		for time in fwdtime:
			Train(stock, 60, time, save_path, thresholds)


