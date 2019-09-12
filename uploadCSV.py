import csv
import MySQLdb
import boto3
import dbconfig as dbc
import glob


rds = boto3.client('rds')
files = glob.glob('C:\\Projects\\Stocks\\Testing Data\\*.csv')

mydb = MySQLdb.connect(host=dbc.hostname,
	port=dbc.port,
	user=dbc.user,
	passwd=dbc.pw,
	db=dbc.dbname)


for file in files:
	fileName = file.split('\\')[-1]
	stock = fileName.split('_')[0]
	with open(file, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		next(reader, None)
		for row in reader:

			#print(row[0], stock, row[1], row[2], row[3], row[4], row[5])


			
			cursor = mydb.cursor()
			cursor.execute(f"INSERT INTO historicData (datetime, symbol, high, low, open, close, volume) VALUES ('{row[0]}', '{stock}', '{row[1]}', '{row[2]}', '{row[3]}', '{row[4]}', '{row[5]}');")
			
	
mydb.commit()
cursor.close()