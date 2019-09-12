class STOCK:
	def __init__(self, csv_row):
		self.symbol = csv_row[0]
		self.date = csv_row[1]
		self.open = csv_row[2]
		self.high = csv_row[3]
		self.low = csv_row[4]
		self.close = csv_row[5]
		self.volume = csv_row[6]

def target_classifier(row, threshold):
	
	#Classifying by trend
	if row['percent_change'] > threshold:
		val = 1
	elif row['percent_change'] < -threshold:
		val = -1
	else:
		val = 0

	return val

	'''
	#Classifying by gain
	val = int(round(row['percent_change']))

	print(row['percent_change'])
	return val
	'''




def RSI_70_30_classifier(row, name):
	
	if row[name] > 70:
		val = 1
	elif row[name] < 30:
		val = -1
	else:
		val = 0

	return val

def martySignal(row):
	
	#Classifying by trend
	if (row['farVol'] > row['closeVol']) and (row['farClose'] > row['closeClose']) and (row['distantClose'] > row['farClose']):
		val = 1
	else:
		val = 0

	return val


def closeWindowLooseSignal(row):
	val = 0
	if row['farClose'] == 0:
		val = 0
	else:
		if (row['distantClose'] / row['farClose']) > 1.1:
			val = 1
		if (row['distantClose'] / row['farClose']) > 1.2:
			val = 2
		if (row['distantClose'] / row['farClose']) > 1.3:
			val = 3
		if (row['distantClose'] / row['farClose']) > 1.4:
			val = 4
		if (row['distantClose'] / row['farClose']) > 1.5:
			val = 5
		if (row['distantClose'] / row['farClose']) > 1.7:
			val = 7
		if (row['distantClose'] / row['farClose']) > 2.0:
			val = 10


	return val


def closeWindowTightSignal(row):
	val = 0
	if row['tightClose'] == 0:
		val = 0
	else:
		if (row['farClose'] / row['tightClose']) > 1.1:
			val = 1
		if (row['farClose'] / row['tightClose']) > 1.2:
			val = 2
		if (row['farClose'] / row['tightClose']) > 1.3:
			val = 3
		if (row['farClose'] / row['tightClose']) > 1.4:
			val = 4
		if (row['farClose'] / row['tightClose']) > 1.5:
			val = 5
		if (row['farClose'] / row['tightClose']) > 1.7:
			val = 7
		if (row['farClose'] / row['tightClose']) > 2.0:
			val = 10
	return val


def volumeWindowSignal(row):
	val = 0
	if row['closeVol'] == 0:
		val = 0
	else:
		if (row['farVol'] / row['closeVol']) > 1.1:
			val = 1
		if (row['farVol'] / row['closeVol']) > 1.2:
			val = 2
		if (row['farVol'] / row['closeVol']) > 1.3:
			val = 3
		if (row['farVol'] / row['closeVol']) > 1.4:
			val = 4
		if (row['farVol'] / row['closeVol']) > 1.5:
			val = 5
		if (row['farVol'] / row['closeVol']) > 1.7:
			val = 7
		if (row['farVol'] / row['closeVol']) > 2.0:
			val = 10
	return val

def voluminousDropSignal(row):
	val = 0
	#Classifying by trend
	if row['historicVolume'] == 0:
		val = 0
	else:
		if (row['minOvernightChange'] < -0.5) and ((row['timeVolume'] / row['historicVolume']) > 1.1):
			val = 1
		if (row['minOvernightChange'] < -1.1) and ((row['timeVolume'] / row['historicVolume']) > 1.2):
			val = 3
		if (row['minOvernightChange'] < -1.4) and ((row['timeVolume'] / row['historicVolume']) > 1.3):
			val = 5
		if (row['minOvernightChange'] < -1.7) and ((row['timeVolume'] / row['historicVolume']) > 1.4):
			val = 7
		if (row['minOvernightChange'] < -2) and ((row['timeVolume'] / row['historicVolume']) > 1.5):
			val = 10
		else:
			val = 0

	return val

def voluminousRiseSignal(row):
	val = 0
	if row['historicVolume'] == 0:
		val = 0
	else:
	#Classifying by trend
		if (row['maxOvernightChange'] > 0.5) and ((row['timeVolume'] / row['historicVolume']) > 1.1):
			val = 1
		if (row['maxOvernightChange'] > 1.1) and ((row['timeVolume'] / row['historicVolume']) > 1.2):
			val = 3
		if (row['maxOvernightChange'] > 1.4) and ((row['timeVolume'] / row['historicVolume']) > 1.3):
			val = 5
		if (row['maxOvernightChange'] > 1.7) and ((row['timeVolume'] / row['historicVolume']) > 1.4):
			val = 7
		if (row['maxOvernightChange'] > 2) and ((row['timeVolume'] / row['historicVolume']) > 1.5):
			val = 10
		else:
			val = 0

	return val




		