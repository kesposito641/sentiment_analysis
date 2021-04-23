import pickle

def lr_preds(t, code="t"):
	"""
	Returns the stock price predictions for the linear regression models
	Inputs:
	- t: the test data
	- code: "t" or "r", meaning reddit or twitter models
	Outputs:
	- l: the long term prediction
	- m: the medium term prediction
	- s: the short term prediction
	"""
	if code=="t": # look at twitter models
		lrLong = pickle.load(open("lr_long.sav", "rb"))
		l = lrLong.predict(t)
		lrMed = pickle.load(open("lr_med.sav", "rb"))
		m = lrMed.predict(t)
		lrShort = pickle.load(open("lr_Short.sav", "rb"))
		s = lrShort.predict(t)
	elif code=="r": # look at reddit models
		l = pickle.load(open("lr_long_red.sav", "rb"))
		m = pickle.load(open("lr_med_red.sav", "rb"))
		s = pickle.load(open("lr_short_red.sav", "rb"))

	return l, m, s

def pct_preds(t, code="t"):
	"""
	Returns the stock return percentage predictions for the linear regression models
	Inputs:
	- t: the test_data
	- code: "t" or "r", meaning reddit or twitter models
	Outputs:
	- l: the long term prediction
	- m: the medium term prediction
	- s: the short term prediction
	"""
	if code=="t":
		lrLong = pickle.load(open("lr_long_pct.sav", "rb"))
		l = lrLong.predict(t)
		lrMed = pickle.load(open("lr_med_pct.sav", "rb"))
		m = lrMed.predict(t)
		lrShort = pickle.load(open("lr_short_pct.sav", "rb"))
		s = lrShort.predict(t)
	elif code=="r": # look at reddit models
		l = pickle.load(open("lr_long_red_pct.sav", "rb"))
		m = pickle.load(open("lr_med_red_pct.sav", "rb"))
		s = pickle.load(open("lr_short_red_pct.sav", "rb"))


	return l, m, s

def xg_preds(t, code="t"):
	"""
	Returns the classication predictions from the xgboost models
	Inputs:
	- t: the test_data
	- code: "t" or "r", meaning reddit or twitter models
	Outputs:
	- l: the long term prediction
	- m: the medium term prediction
	- s: the short term prediction
	"""
	if code=="t": # twitter models
		xgLong = pickle.load(open("xgLong.dat", "rb"))
		l = xgLong.predict(t)
		xgMed = pickle.load(open("xgMed.dat", "rb"))
		m = xgMed.predict(t)
		xgShort = pickle.load(open("xgShort.dat", "rb"))
		s = xgShort.predict(t)
	elif code=="r": # reddit models
		l = pickle.load(open("xgLongReddit.dat", "rb"))
		m = pickle.load(open("xgMedReddit.dat", "rb"))
		s = pickle.load(open("xgShortReddit.dat", "rb"))

	return l, m, s

def next_day_preds(t):
	"""
	Returns predictions for tomorrow's stock price
	Inputs:
	- t: the test_data
	Outputs:
	- l: the long term prediction
	- m: the medium term prediction
	- s: the short term prediction
	"""
	model = pickle.load(open("nextday.sav", "rb"))
	ret = model.predict(t)

	return ret
