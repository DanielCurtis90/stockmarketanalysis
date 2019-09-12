import glob
import pandas as pd

files = glob.glob('C:\\Projects\\Stocks\\Testing Data ALL\\*.csv')
#files = glob.glob('C:/Projects/Stocks/*.csv')


d = {f: sum(1 for line in open(f)) for f in files}

series = pd.Series(d)

series.sort_values(ascending=False, inplace=True)
series.to_csv('fileLengths.csv')

