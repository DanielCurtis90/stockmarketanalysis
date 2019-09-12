import os, sys, csv, pickle, shutil, time, datetime
import pymysql
from models import *


#Database connection
db = pymysql.connect("localhost","root","python1root!","stockdata")

#Prepare a cursor
cursor = db.cursor()

path = "C:\\Users\\Administrator\\Desktop\\NASDAQ\\Input"
save_route = "C:\\Users\\Administrator\\Desktop\\NASDAQ\\Output"

start_time = time.time()

for file in os.listdir(path):
	filename = os.fsdecode(file)
	if file.endswith(".csv"):
		file_time = time.time()
		filepath = os.path.join(path, filename)
		savepath = os.path.join(save_route, filename)
		with open(filepath) as csvfile:
			reader = csv.reader(csvfile, delimiter =',')
			#We do not want the first line
			firstline = True
			for row in reader:
				#Skip first line
				if firstline:
					firstline = False
					continue
				#We first store each symbol for that day and the rest of the data associated with that row
				sql = f"INSERT INTO stock_minute_data (symbol, date, open, high, low, close, volume) VALUES ('{row[0]}', '{datetime.datetime.strptime(row[1],'%d-%b-%Y %H:%M').strftime('%Y-%m-%d %H:%M')}', '{row[2]}', '{row[3]}', '{row[4]}', '{row[5]}', '{row[6]}')"
				# Execute the SQL command
				cursor.execute(sql)
				# Commit your changes in the database
				db.commit()
		#Move the csv after it has been processed
		shutil.move(filepath, savepath)
		print(f"{filename} ({os.path.getsize(savepath)} bytes) loaded in: {round((time.time() - file_time), 2)} seconds.")

print(f"Total loading time: {round((time.time() - start_time), 2)} seconds.")

		   
