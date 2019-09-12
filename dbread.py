import boto3, time, os, sys, csv, datetime
import pymysql
import dbconfig as dbc
import pandas as pd

from sqlalchemy import create_engine

path = "C:\\Projects\\Stocks\\TestInput"


rds = boto3.client('rds')

db = pymysql.connect(host=dbc.hostname,user=dbc.user,passwd=dbc.pw,db=dbc.dbname, port=dbc.port)
cursor = db.cursor()
sqlQuery = f"SELECT * FROM DCDataplays.tailBackTestDataframes WHERE modelId = 32560 AND threshold = 1.0 ;"
cursor.execute(sqlQuery)

# Get data in batches
first = True

while True:
	print('writing to file')
	fetched = cursor.fetchmany(1000)
	if len(fetched) == 0:
		break

	with open("output.csv","a", newline='') as outfile:
		writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
		if first == True:
			writer.writerow(col[0] for col in cursor.description)
			first = False

		for row in fetched:
			writer.writerow(row)


# Clean up
cursor.close()




'''
sqlQuery = f"UPDATE transactions SET filledPrice = '{100.11}', filledShares = {69}  WHERE orderId = '{1}' AND date = '{datetime.datetime.now().date()}'"
cursor.execute(sqlQuery)
db.commit()
'''

'''
sqlQuery = f"INSERT INTO Transactions (orderid, key_date, symbol, shareprice, sharequantity, islong, limitplacedtime) VALUES ('{2}', '{datetime.datetime.now().date()}', 'TEST', '{200}', '{50}', '{1}', '{datetime.datetime.now()}')"
cursor.execute(sqlQuery)
db.commit()
sqlQuery = f"UPDATE Transactions SET lmtfillquantity = '{69}' WHERE orderid = '{2}' AND key_date = '{datetime.datetime.now().date()}'"
cursor.execute(sqlQuery)
db.commit()
'''

'''
#sql = f"INSERT INTO Test (Symbol, TimePlaced, OrderID, Type, Price, Quantity, Filled, ExitOrderID, ExitTime) VALUES ('{row[0]}', '{datetime.datetime.strptime(row[3],'%Y%m%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')}', '{row[4]}', '{row[8]}', '{row[1]}', '{row[2]}', '{row[5]}', '{row[6]}', '{datetime.datetime.strptime(row[7],'%Y%m%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')}')"
#Date,High,Low,Open,Close,Volume
#We first store each symbol for that day and the rest of the data associated with that row
sql = f"INSERT INTO Test (Symbol, TimePlaced, OrderID, Type, Price, Quantity, Filled, ExitOrderID, ExitTime) VALUES ('{row[0]}', '{datetime.datetime.strptime(row[3],'%Y%m%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')}', '{row[4]}', '{row[8]}', '{row[1]}', '{row[2]}', '{row[5]}', '{row[6]}', '{datetime.datetime.strptime(row[7],'%Y%m%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')}')"
# Execute the SQL command
cursor.execute(sql)
# Commit your changes in the database
db.commit()
'''