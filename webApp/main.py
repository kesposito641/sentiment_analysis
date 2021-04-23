import faulthandler; faulthandler.enable()
from flask import Flask
from flask import request
import pandas as pd 
from updated import *
from models import *

# Create the web app
app = Flask(__name__)

def highlight(value):
	'''
	Determines which cells need to be highlighted green (bullish) or red (bearish) in the data tables
	based off the given sentiment
	Inputs:
	- value: 0 or 1 classification
	Outputs:
	- a string of the html td tag to highlight the cell red or green
	'''
	if value == 1:
		return '<td style=background-color:#006400>Bullish</td>'
	elif value == 0:
		return '<td style=background-color:#ff5c5c>Bearish</td>'

def create_table(items):
	'''
	Creates the html text needed to create a table given the items inside it
	Inputs:
	- items: a list containing dictionaries that contain the values for each column in the table (short, mid, long)
	Outputs:
	- table_html: a string containing all of the html text required to display the table
	'''
	table_html = """
	<table class=styled-table>
  	<tr>
    <td></td>
    <th scope="col">Short Term ~ 2 Weeks</th>
    <th scope="col">Mid Term ~ 3 Months</th>
    <th scope="col">Long Term ~ 1 Year</th>
  	</tr>
	"""
	
	# iterate through the items in the dictionary
	for i in range(0, len(items)):
		dic = items[i]
		new_html = "<tr>"
		if i == 0: # looking at the values for stock prediction for linear regression
			new_html = new_html + '<th scope="row">Linear Regression Prediction</th>'
			new_html = new_html + '<td>' + "${:,.2f}".format(dic.get('short')) + '</td>' + '<td>' + "${:,.2f}".format(dic.get('mid')) + '</td>' + '<td>' + "${:,.2f}".format(dic.get('long')) + '</td>' + "</tr>"
		elif i == 1: # percent return prediction
			new_html = new_html + '<th scope="row">Linear Regression Return Percentage</th>'
			new_html = new_html + '<td>' + "{:.0%}".format(dic.get('short')) + '</td>' + '<td>' + "{:.0%}".format(dic.get('mid')) + '</td>' + '<td>' + "{:.0%}".format(dic.get('long')) + '</td>' + "</tr>"
		elif i == 2: # classification prediction
			new_html = new_html + '<th scope="row">Bullish or Bearish?</th>'
			new_html = new_html + highlight(dic.get('short')) + highlight(dic.get('mid')) + highlight(dic.get('long')) + "</tr>"

		table_html = table_html + new_html

	table_html = table_html + "</table>"
	return table_html

def make_header_table(data, sent):
	"""
	Creates the html text to display the current stock price, tomorrow's prediction, and the current investor sentiment
	Inputs:
	- data: TickerInfo object of the current ticker
	- sent: the average sentiment score for the current ticker
	Ouputs:
	- table_html: a string containing all of the html required to display the table 
	"""
	table_html = """
	<table class=styled-table-two>
	<tr>
	<th scope="row">Latest Close Price</th><td>""" + "${:,.2f}".format(data.most_recent_price()) + """
	</td>
	</tr>
	<tr>
	<th scope='row'>Tomorrow's Predicted Price</th><td>""" + "${:,.2f}".format(next_day_preds([data.ticker(), data.most_recent_price()]) + """
	</td>
	</tr>
	<tr>
	<th scope='row'>Current Investor Sentiment</th><td>""" + str(sent) + """
	</td>
	</tr>
	</table><p></p>"""
	return table_html

def get_data(ticker, data):
	'''
	Getting all the data formatted properly from the csvs and returning all of the html text to make that table
	Inputs:
	- ticker: the stock ticker
	- data: the dataframe we are extracting data from
	Outputs:
	- table: all of the html text needed to create the tables
	'''
	table = ""

	# create the proper shape of the test data
	sentiment = data.get_avg_sentiment()
	recent_sentiment = sentiment[3]
	table = table + make_header_table(data, recent_sentiment)
	prices = list(data.past_five_days_return()["adjclose"])
	for price in prices:
		sentiment.append(price)
	sentiment.append(data.most_recent_price())
	test_data = pd.DataFrame(sentiment)

	# get values from the models we trained using twitter
	lr_1yr_pred, lr_3mth_pred, lr_2wk_pred = lr_preds(test_data.T)
	pct_1yr_pred, pct_3mth_pred, pct_2wk_pred = pct_preds(test_data.T)
	xg_1yr_pred, xg_3mth_pred, xg_2wk_pred = xg_preds(test_data.T)
	# print("lr_1yr", lr_1yr_pred[0][0])

	twitter_items = [dict(short=lr_2wk_pred[0][0], mid=lr_3mth_pred[0][0], long=lr_1yr_pred[0][0]),
	dict(short=pct_2wk_pred[0][0], mid=pct_3mth_pred[0][0], long=pct_1yr_pred[0][0]),
	dict(short=1, mid=0, long=1)]
	# dict(short=xg_2wk_pred, mid=xg_3mth_pred, long=xg_1yr_pred)] uncomment this for submission

	# add to the html we are building
	table = table + "<h2>Showing Results for " + ticker.upper() + " from Twitter data: </h2>"
	table = table + create_table(twitter_items)

	# get values from the models we trained using reddit
	lr_1yr_pred, lr_3mth_pred, lr_2wk_pred = lr_preds(test_data.T, "r")
	pct_1yr_pred, pct_3mth_pred, pct_2wk_pred = pct_preds(test_data.T, "r")
	xg_1yr_pred, xg_3mth_pred, xg_2wk_pred = xg_preds(test_data.T, "r")
	# print("lr_1yr", lr_1yr_pred[0][0])

	twitter_items = [dict(short=lr_2wk_pred[0][0], mid=lr_3mth_pred[0][0], long=lr_1yr_pred[0][0]),
	dict(short=pct_2wk_pred[0][0], mid=pct_3mth_pred[0][0], long=pct_1yr_pred[0][0]),
	dict(short=1, mid=0, long=1)]

	# add to the html we are building
	table = table + "<h2>Showing Results for " + ticker.upper() + " from Reddit data: </h2>"
	table = table + create_table(twitter_items)

	return table



@app.route("/")
def index():
    ticker = request.args.get("ticker", "")
    # print(k)
    
    if ticker:
    	data = TickerInfo(ticker)
    	twitter_table = get_data(ticker, data)
    else:
        ticker="AAPL"
        data = TickerInfo(ticker)
        twitter_table = get_data(ticker, data)
    return (
	"""<div class="content"><h1>Stock Market Predictions Through Sentiment Analysis</h1>"""+
    """<form action="" method="get">
                Search for a Stock Ticker: <input type="text" name="ticker">
                <input type="submit" value="Get Predictions!">
            </form>"""
            + twitter_table
            +"""<style>
            		content {
  						max-width: 500px;
  						margin: auto;
					}
            		body {
  						font-family: Verdana, sans-serif;
  						color: #ffffff;
  						background-color: #aed1ca;
  						text-align: center
					}
					th {
  						background-color: #00755e;
  						color: white;
					}
					th, td {
  						padding: 15px;
  						text-align: left;
  						border-bottom: 1px solid #ddd;
					}
					.styled-table {
					    border-collapse: collapse;
					    border-spacing: 50px 0;
  						width: 70%;
  						height: 40%;
					    margin-left: auto;
					    margin-right: auto;
					    font-size: 1.5em;
					    font-family: sans-serif;
					    min-width: 400px;
					    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
					}
					.styled-table tr {
					    background-color: #009879;
					    color: #ffffff;
					    text-align: left;
					}
					.styled-table th,
					.styled-table td {
					    padding: 12px 15px;
					}
					.styled-table tbody tr {
					    border-bottom: 1px solid #dddddd;
					}

					.styled-table tbody tr:nth-of-type(even) {
					    background-color: #f3f3f3;
					    color: #009879;
					}

					.styled-table tbody tr:last-of-type {
					    border-bottom: 2px solid #009879;
					}
					.styled-table tbody tr.active-row {
					    font-weight: bold;
					    color: #009879;
					}
					.styled-table-two {
						border-collapse: collapse;
					    border-spacing: 50px 0;
  						width: 70%;
  						height: 15%;
					    margin-left: auto;
					    margin-right: auto;
					    font-size: 1.5em;
					    font-family: sans-serif;
					    min-width: 400px;
					    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
					}


					"""
					+ 
				"""</style></div>
					<div><p><p></p></p></div>
					<div style="background-color:#00755e;color:white;padding:30px;">
  					<h2>Further Information:</h2>
  					<p>The purpose of this dashboard is to allow users to view various prediction metrics about stock performance calculated using sentiment data and financial data. 
  					The financial data was compiled from YahooFinance's API and the sentiment data was derived using tweets from Twitter and posts from various investing-related subreddits on Reddit.
   					A user can look up information about a specific stock using the search tool at the top (most of the valid tickers are from the S&P500) and the tables will populate with the relevant information.</p>
   					<p>For the various data sources, we calculated a stock prediction, return percentage, sentiment score, and outlook. Each of these metrics were calculated for different timespans, ranging from 2 weeks to 1 year to consider different investing strategies. 
   					For added convenience, descriptions of each of the metrics are found below:
   					<ul>
  					<li><span class="bold">Prediction</span> - The predicted stock price </li>
 					<li><span class="bold">Return Percentage</span> - The predicted percent increase or decrease of stock value </li>
  					<li><span class="bold">Sentiment Score</span> - A value ranging from MIN to MAX where a lower number corresponds to negative investor sentiment and a higher number corresponds to positive investor sentiment</li>
  					<li><span class="bold">Bearish or Bullish?</span> - The overall outlook on whether or not an investor should feel inclined to buy, sell, or hold the stock </li>
  					<p></p>
  					</ul>
  					<style>
  					.bold {
						font-weight:bold;
					}
					</style>
					</div> """
	
    )



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
