from queue import PriorityQueue
import pickle

#stock_list = ['INTC', 'ABMD', 'AMGN', 'ANSS', 'ASML', 'ATVI', 'AZPN', 'CBRL', 'CHCO', 'CME', 'CTAS', 'EA', 'EQIX']
#stock_list = ['FFIV', 'FTNT', 'GILD', 'GOLD', 'IAC', 'IBKC', 'ICLR', 'INTU', 'JAZZ', 'JKHY', 'LGND', 'LOGM', 'LRCX', 'LULU', 'MEOH'] 
#stock_list = ['MKSI', 'MKTX', 'NTAP', 'NTES', 'NXST', 'POOL', 'QLYS', 'QQQ', 'QRVO', 'RGLD', 'SAFM', 'SGEN', 'SHPG', 'SIVB', 'SNPS']
#stock_list = ['SODA', 'SOXX', 'STMP', 'TCBI', 'TREE', 'TSLA', 'ULTA', 'ULTI', 'VCSH', 'VRSN', 'VRTX', 'WBA', 'XLNX', 'YY', 'ZBRA']


stock_list = ['UTHR']
#Live stocks
#stock_list = ['FISV', 'CINF', 'HAS', 'INTC', 'QCOM', 'ALGN', 'CONE', 'FIVE', 'GRMN', 'HSIC', 'MASI', 'NVDA', 'CHCO', 'AZPN', 'EA', 'CME', 'FTNT', 'IAC', 'GILD', 'MEOH', 'LOGM', 'QRVO', 'LULU', 'RGLD', 'QLYS', 'SOXX', 'POOL', 'TSLA', 'ZBRA', 'SIVB', 'STMP', 'VRSN', 'WBA']


#stock_list = ['CINF']
#stock_list = ['CINF', 'QCOM', 'CONE', 'HSIC', 'NVDA', 'CME', 'SOXX', 'HAS', 'MASI', 'IAC', 'RGLD', 'STMP', 'ZBRA']

#stock_list = ['FCNCA', 'CACC', 'EQIX', 'CSGP', 'ABMD', 'SAFM', 'EA']
#stock_list = ['ULTI', 'SIVB', 'NWLI', 'CHDN', 'ULTA', 'ICUI', 'TSLA', 'SPSC']
#stock_list = ['DJCO', 'AVGO', 'NTES', 'STMP', 'AZPN', 'CVGW', 'BND', 'CGVIC']
#stock_list = ['HIFS', 'INTU', 'CVCO', 'LGND', 'IAC', 'INGN', 'TREE', 'SPSC', 'SNPS']
#stock_list = ['AMGN', 'ESGR', 'MDGL', 'MKTX', 'CTAS', 'TECH', 'VRTX', 'IJT']
#stock_list = ['LFUS', 'CME', 'QQQ', 'ASML', 'MLAB', 'ITIC', 'SHPG', 'SOXX', 'MOGLC']
#stock_list = ['FFIV', 'DHIL', 'LOXO', 'COKE', 'ANSS', 'JAZZ', 'ZBRA', 'VC']
#stock_list = ['JKI', 'CBRL', 'WDFC', 'JKHY', 'VONG', 'WINA', 'POOL', 'SBAC', 'LOGM']
#stock_list = ['LANC', 'JJSF', 'ICLR', 'VRSN', 'LULU', 'LRCX', 'GWPH', 'SODA', 'FTNT']



#stock_list = ['ADP', 'VTWG', 'ADSK', 'ROLL', 'IPGP', 'WLTW', 'ODFL', 'PLCE']
#stock_list = ['COHR', 'PRFZ', 'SRPT', 'TTWO', 'WDAY', 'MSTR', 'DXCM', 'ATHN']
#stock_list = ['VONE', 'VTHR', 'FANG', 'BLUE', 'PSCH', 'NDSN', 'BBH', 'SAGE']
#stock_list = ['CASY', 'VTWO', 'UTHR', 'STRA', 'PNQI', 'ISRL', 'ERIE', 'ALXN']
#stock_list = ['CASY', 'VTWO', 'UTHR', 'STRA', 'PNQI', 'ISRL', 'ERIE', 'ALXN']
#stock_list = ['ANAT', 'ESLT', 'ALGT', 'HELE', 'IEI', 'EXPE', 'MAR', 'MASI']
#stock_list = ['WYNN', 'MIDD', 'LOPE', 'VRSK', 'FIVE', 'TLT', 'CMPR', 'MPWR']
#stock_list = ['LIVN', 'EEFT', 'OLED', 'MORN', 'CHKP', 'NBIX', 'IBB', 'SHV'] 
#stock_list = ['MSFT', 'VRTS', 'JBHT', 'AMED', 'SBNY', 'NICE', 'VTWV', 'LSTR']
#stock_list = ['VONV', 'EMB', 'PEP', 'ICPT', 'SPLK', 'CTXS', 'MBB', 'GFNCP']
#stock_list = ['KALU', 'TXN', 'PFPT', 'DWAQ', 'IEF', 'TROW', 'CIVEC', 'CBOE']
#stock_list = ['FIZZ', 'PRAH', 'BMRN', 'WRLD', 'NTRS', 'PXUS', 'HAS', 'VRTSP']
#stock_list = ['WIX', 'DVY', 'BCPC', 'ROST', 'PTC', 'CCMP', 'ESRX', 'HSKA']
#stock_list = ['KLAC', 'ESGG', 'AVAV', 'CHRW', 'BOKF', 'IRBT', 'LHCG', 'PODD']
#stock_list = ['NVEC', 'SWKS', 'COLM', 'TSCO', 'UBNT', 'VCLT', 'UTMD', 'PLUS']
#stock_list = ['PTH', 'ADI', 'HSIC', 'SAFT', 'LECO', 'HQY', 'CATC', 'SHY']
#stock_list = ['VCIT', 'CELG', 'NVEE', 'CDW', 'NATH', 'JACK', 'ALNY', 'CNBKA']
#stock_list = ['WTFC', 'SLAB', 'ISHG', 'DLTR', 'VTC', 'UAL', 'NDAQ', 'LBRDK']
#stock_list = ['LBRDA', 'INDB', 'RYAAY', 'PYPL', 'BMRC', 'FSV', 'NXPI', 'SBNYW']

#stock_list = ['ATVI', 'VCSH', 'PSCC', 'RGLD', 'USLM', 'GOLD', 'SSB', 'FISV', 'IBKC']
#stock_list = ['TCBIW', 'NTAP', 'PNRG', 'TCBI', 'XLNX', 'JOUT', 'PSMT', 'PSCT', 'RMR']
#stock_list = ['DLBL', 'MEOH', 'VWOB', 'MKSI', 'ENTA', 'PVAC', 'CBPO', 'MGPI', 'BNDW']
stock_list = ['FCFS', 'BLKB', 'NXST', 'GILD', 'QLYS', 'WBA', 'QTEC', 'YY', 'SGEN']
#stock_list = ['JCOM', 'QRVO', 'DTYL', 'POPE', 'CHCO', 'CNMD']




seed_modifier = len(stock_list)

#When we
transmit_counter = 0
transmit_counter_end = len(stock_list)

#The maximum amount of money per action allowed
trade_amount = 10000

predict_list = ['upperb_60', 'rsi_60', 'close60to3900', 'close_60_sma', 'close_180_sma', 'close_3900_sma', 'close_390_sma']

stock_long_q = PriorityQueue()
stock_short_q = PriorityQueue()

stock_dict = {}

stock_priority_dict = {}
stock_model_dict = {}
stock_rev_reqId_dict = {}

stock_priority_dict = {
	'AAPL' : [18, 5],
	'INTC' : [7, 8],
	'MSFT' : [5, 2],
	'AMZN' : [9, 11],
	'GILD' : [15, 13],
	'FOXA' : [12, 12],
	'NFLX' : [3, 18],
	'FIVE' : [99, 99],
	'NVDA' : [1, 20],
	'ADBE' : [13, 19],
	'BKNG' : [8, 14],
	'ISRG' : [11, 3],
	'REGN' : [6, 17],
	'BIIB' : [20, 1],
	'ALGN' : [4, 6],
	'CHTR' : [2, 7],
	'MELI' : [10, 21],
	'BIDU' : [21, 4]
}

stock_threshold_dict = {
	'AAPL' : [0.6, 0.5],
	'INTC' : [0.2, 0.8],
	'MSFT' : [0.2, 0.3],
	'AMZN' : [1.0, 0.8],
	'GILD' : [0.9, 0.2],
	'FOXA' : [0.9, 0.3],
	'NFLX' : [0.2, 0.7],
	'NVDA' : [0.2, 0.3],
	'ADBE' : [0.2, 1.0],
	'FIVE' : [5, 0.8],
	'BKNG' : [0.2, 0.8],
	'ISRG' : [0.2, 0.7],
	'REGN' : [0.8, 0.8],
	'BIIB' : [1.0, 0.2],
	'ALGN' : [0.3, 0.4],
	'CHTR' : [0.4, 0.2],
	'MELI' : [0.3, 0.9],
	'BIDU' : [1.0, 0.2]
}

stock_long_or_short_priority_dict = {
	'AAPL' : 'short',
	'INTC' : 'long_only',
	'MSFT' : 'long',
	'FIVE' : 'short_only',
	'AMZN' : 'long_only',
	'GILD' : 'long_only',
	'FOXA' : 'long_only',
	'NFLX' : 'long_only',
	'NVDA' : 'long_only',
	'ADBE' : 'long_only',
	'BKNG' : 'long_only',
	'ISRG' : 'long',
	'REGN' : 'long_only',
	'BIIB' : 'short_only',
	'ALGN' : 'short',
	'CHTR' : 'long_only',
	'MELI' : 'long_only',
	'BIDU' : 'short_only'
}

class Stock:
	def __init__(self, name, seed):
		self.name = name
		self.clientID = seed
		self.reqID = seed + seed_modifier
		self.volume = 0
		self.penalty = 0
		self.count = 0
		self.tickerID = seed + (2*seed_modifier)
		self.long_model = 0
		#Can disable these for model testing
		'''
		self.thresholds = stock_threshold_dict[name]
		self.lspriority = stock_long_or_short_priority_dict[name]
		'''
		###
		self.short_model = 0
		self.dataframe = 0
		self.execID = seed + (3*seed_modifier)

class Stock_Order:
	def __init__(self, name, quantity, price, exit_time, placed_time):
		self.name = name
		self.quantity = quantity
		self.price = price
		self.exit_time = exit_time
		self.placed_time = placed_time

for counter, stock in enumerate(stock_list, 1):
	stock_dict[stock] = Stock(stock, counter)
	stock_rev_reqId_dict[counter + (2*seed_modifier)] = stock


#For debugging the stock objects
'''
for k, v in stock_dict.items():
    print(vars(v))
'''



	
