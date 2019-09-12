import os, sys, pickle, datetime, time, csv
import pandas as pd
import numpy as npy

def generate_csvs(path, save_path):

	end_of_Folder = False
	is_indexed = False

	for counter, file in enumerate(os.listdir(path)):
		if counter % 30 == 0 and counter > 0:
			print(f"Loading files {counter-30} to {counter} out of {len(os.listdir(path))}.")
			templist = []
			df = pd.DataFrame()
			for file in os.listdir(path)[(counter - 30):counter]:
				filename = os.fsdecode(file)
				if file.endswith(".csv"):
					file_time = time.time()
					filepath = os.path.join(path, filename)
					tempdf = pd.read_csv(filepath,index_col=None, header=0)
					templist.append(tempdf)
					print(f"{filename} ({os.path.getsize(filepath)} bytes) loaded in: {round((time.time() - file_time), 2)} seconds.")

			df = pd.concat(templist)
			df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y %H:%M")
			print('Datetime conversion complete.')
			df = df.sort_values(by=['Symbol', 'Date'])
			print('Dataframe sorted.')
			df_grouplist = df.groupby(['Symbol'])
			print('Dataframe grouped.')

			for name, group in df_grouplist:
				file_save_path = f'{save_path}{name}_data.csv'
				#append dataframe for that stock to existing file if it exists
				if os.path.isfile(file_save_path):
					with open(file_save_path, 'a') as f:
						print(f"Appending {name} to csv.")
						group.to_csv(f, header=False, index=False)
				else:
					print(f"Writing {name} to csv.")
					group.to_csv(file_save_path, index=False)

			if (len(os.listdir(path)) - counter) < 30:
				print(f"Loading files {counter} to {len(os.listdir(path))} out of {len(os.listdir(path))}.")
				end_of_Folder = False
				print("Loading end of data.")
				templist = []
				df = pd.DataFrame()
				start_time = time.time()
				for file in os.listdir(path)[counter:len(os.listdir(path))]:
					filename = os.fsdecode(file)
					if file.endswith(".csv"):
						file_time = time.time()
						filepath = os.path.join(path, filename)
						tempdf = pd.read_csv(filepath,index_col=None, header=0)
						templist.append(tempdf)
						print(f"{filename} ({os.path.getsize(filepath)} bytes) loaded in: {round((time.time() - file_time), 2)} seconds.")

				df = pd.concat(templist)
				df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y %H:%M")
				print('Datetime conversion complete.')
				df = df.sort_values(by=['Symbol', 'Date'])
				print('Dataframe sorted.')
				df_grouplist = df.groupby(['Symbol'])
				print('Final dataframe grouped.')

				for name, group in df_grouplist:
					file_save_path = f'{save_path}{name}_data.csv'
					#append dataframe for that stock to existing file if it exists
					if os.path.isfile(file_save_path):
						with open(file_save_path, 'a') as f:
							print(f"Appending {name} to csv.")
							group.to_csv(f, header=False, index=False)
					else:
						print(f"Writing {name} to csv.")
						group.to_csv(file_save_path, index=False)






	



	