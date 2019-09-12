import os
import glob
import pandas as pd
import numpy as np
import time

files = glob.glob('C:\\Projects\\Stocks\\StockData\\*.csv')

for f in files:
	


	df = pd.read_csv(f)

	#Get the raw name of the file, no directory
	fileName = f.split('\\')[-1]
	stockName = fileName.split('_')[0]
	print(stockName)
	

	len1 = (len(df.index))
	df2 = df.drop_duplicates()
	len2 = (len(df2.index))
	print(f"{f} contains {len1 - len2} duplicate rows")


	df2['Date'] = pd.DatetimeIndex(df2['Date'])
	mask = (df2['Date'] > '2017-01-16 ') & (df2['Date'] <= '2019-04-22')
	df3 = df2.loc[mask]
	print(len(df3.index))
	if len(df3.index) == 220620:
		df3.to_csv(f'C:\\Projects\\Stocks\\Testing Data\\{stockName}_historic_data_filtered.csv', index=False)

	

	'''
	dfdrop = (df[df.duplicated()])
	dfdrop.to_csv('dropped.csv')

	'''