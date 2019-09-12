import os

import time, csv, datetime, queue, pickle, threading, warnings
import pandas as pd
import numpy as np

open_dir = "C:/Projects/Stocks/EOD Minutely"
save_dir = "C:/Projects/Stocks/Split Data"

first = True
for filename in os.listdir(open_dir):
	df = pd.read_csv(f"{open_dir}/{filename}")
	df_grouplist = df.groupby(['Symbol'])
	
	for symbol, sub in df_grouplist:
		if first == True:
			new = sub[['Date', 'High', 'Low', 'Open', 'Close', 'Volume']].copy()
			new.columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume']
			print(new.head(5))
			with open(f'{save_dir}/{symbol}_historic_data.csv', 'w') as f:
				new.to_csv(f, index=False, header=True)

		else:
			new = sub[['Date', 'High', 'Low', 'Open', 'Close', 'Volume']].copy()
			new.columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume']
			with open(f'{save_dir}/{symbol}_historic_data.csv', 'a') as f:
				new.to_csv(f, header=False, index=False)
	first = False

print("Completed EOD data split.")


	 