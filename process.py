import os

import time, csv, datetime, queue, pickle, threading, warnings
import pandas as pd
import numpy as np

open_dir = "C:/Projects/Stocks/Split Data"
save_dir = "C:/Projects/Stocks/Processed Data"

first = True
for filename in os.listdir(open_dir):
	df = pd.read_csv(f"{open_dir}/{filename}")

	print(df.head(5))
	print(df.tail(5))

	df = df.sort_values(by=['Date'])

	df['Date'] = pd.DatetimeIndex(df['Date'])
	group.set_index(keys='Date', inplace=True)

	df_processed = df.between_time('9:29','16:00')

	print(df_processed.head(5))
	print(df_processed.tail(5))

	 