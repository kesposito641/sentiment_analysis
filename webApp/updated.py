import os
import pandas as pd
import tweepy
import re
import string
from textblob import TextBlob
import preprocessor as p
import csv
import sys

import pandas_datareader as pdr
import datetime as dt
from datetime import timedelta, date
from datetime import datetime

from yahoo_fin.stock_info import get_data
from get_all_tickers import get_tickers as gt

from transformers import AutoModelForSequenceClassification
from finbert import *
import utils as tools

# import pre-trained finbert model
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

# twitter authentification information
api_key="yyzrTZGD3aocjx6m0DvhS0Ayt"
api_secret_key="ZXovJpvs1hlaaJl6simAIw09DzQPwKNdVYyZ6uDMcyTRTkLTBe"
access_token="1362158646180933632-OseGDkAYIcrUaOVxT0Urkoloq9XRbf"
access_token_secret="gkgi6WO4Jw3ZBMLhH2UZLp2lpIDEgjd2tHQvNMcsCtCj5"
bearer_token="AAAAAAAAAAAAAAAAAAAAAIkMOwEAAAAA3bmz0snOFAryYITXFvwy3TJ5c0g%3D1JWOMMOsj6XMff243jHwcC25OnnLVSNG5hKBGDny1WamTyfPqI"

# pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)


def tweetSentPredict(text, model):
    """
    Returns sentiment information derived from the finbert model
    Inputs:
    - text: the twitter or reddit data we will be passing through the finbert model
    - model: the finbert model
    Outputs:
    - results: the sentiment information
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    label_list = ['positive', 'negative', 'neutral']
    result = pd.DataFrame(columns=['Negative', 'Neutral', 'Positive', 'sentiment_score'])
    for batch in chunks(text, 5):
        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]
        features = convert_examples_to_features(examples, label_list, 64, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
	
        with torch.no_grad():
            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            logits = softmax(np.array(logits.cpu()))
            sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
            batch_result = {'sentiment_score': sentiment_score,'Positive': list(logits)[0][2],'Neutral': list(logits)[0][1],'Negative': list(logits)[0][0]}
            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result, batch_result], ignore_index=True)
    return result

class TickerInfo:
    """
    A Class that helps get current information about a certain stock such as current investor sentiment and 
    recent stock returns data
    """
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.date = dt.datetime.now()
        
    def name(self):
        return self.ticker
    
    def get_date(self):
        return self.date.strftime('%Y-%m-%d')
    
    def get_avg_sentiment(self):
        query = "$"+ self.ticker + " -filter:retweets -filter:replies"
        tweets = tweepy.Cursor(api.search,q=query,lang='en',tweet_mode='extended', until=self.date.strftime('%Y-%m-%d')).items(50)
    
        lot = []
        
        for tweet in tweets:
            status = tweet._json
            tweet_text = status['full_text']
            tweet_text = re.sub(r'\W', ' ',tweet_text)
            lot.append(tweet_text)
        
        l = []
        sentiment = tweetSentPredict(lot, model)
        for s in sentiment.columns:
        	l.append(sentiment[s].mean())

        return l
        
    def past_five_days_return(self):
    	returns = get_data(self.ticker, (self.date - timedelta(days=10)).date(), self.date.date(), index_as_date=False, interval='1d')[["date", "adjclose"]]
    	returns = returns.tail(5)
    	return returns

    def most_recent_price(self):
    	returns = get_data(self.ticker, (self.date - timedelta(days=4)).date(), self.date.date(), index_as_date=False, interval='1d')[["date", "adjclose"]]
    	returns = returns.tail(1)
    	return returns.iloc[0]["adjclose"]
