path = "C:\\Projects\\Stocks\\Amal" # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
frame.to_csv('amalgamated_dataframe.csv', index=False)