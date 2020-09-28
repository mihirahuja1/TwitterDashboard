import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import settings
import itertools
import math
import base64
from flask import Flask
import os
import psycopg2
import datetime
 
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

import tweepy
import plotly.express as px
# from geopy import geocoders  
# from  geopy.geocoders import Nominatim

from opencage.geocoder import OpenCageGeocode
key = '63fec70fbe1e4b45abab3af391636427'  # get api key from:  https://opencagedata.com

geocoder = OpenCageGeocode(key)


import warnings
warnings.filterwarnings("ignore")

# import nest_asyncio
# nest_asyncio.apply()

import twint
from datetime import datetime
import re
from wordcloud import STOPWORDS

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Real-Time Twitter Monitor'

server = app.server

app.layout = html.Div(children=[
    html.H2('Real-Time Tweet Tracking for $PYPL', style={
        'textAlign': 'center'
    }),
    html.H4('(Graphs loading & refreshing..)', style={
        'textAlign': 'right'
    }),
    

    html.Div(id='live-update-graph'),
    html.Div(id='live-update-graph-bottom'),

    # Author's Words
    html.Div(
        className='row',
        children=[ 
            dcc.Markdown("__Author's Words__: This dashboard is real-time pipeline for tweet collection on topics and provide insights for the users. Sometimes the graph won't load due to API issues, in that case return after a while! "),
        ],style={'width': '35%', 'marginLeft': 70}
    ),
    html.Br(),
    
    # ABOUT ROW
    html.Div(
        className='row',
        children=[
            html.Div(
                className='three columns',
                children=[
                    html.P(
                    'Data extracted from:'
                    ),
                    html.A(
                        'Twitter API',
                        href='https://developer.twitter.com'
                    )                    
                ]
            ),
            html.Div(
                className='three columns',
                children=[
                    html.P(
                    'Code avaliable at:'
                    ),
                    html.A(
                        'GitHub',
                        href='https://github.com/mihirahuja1'
                    )                    
                ]
            ),
            html.Div(
                className='three columns',
                children=[
                    html.P(
                    'Made with:'
                    ),
                    html.A(
                        'Dash / Plot.ly',
                        href='https://plot.ly/dash/'
                    )                    
                ]
            ),
            html.Div(
                className='three columns',
                children=[
                    html.P(
                    'Author:'
                    ),
                    html.A(
                        'Mihir Ahuja',
                        href='https://www.linkedin.com/in/mihir-ahuja/'
                    )                    
                ]
            )                                                          
        ], style={'marginLeft': 70, 'fontSize': 16}
    ),

    dcc.Interval(
        id='interval-component-slow',
        interval=1*45000000, # in milliseconds
        n_intervals=0
    )
    ], style={'padding': '20px'})



# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'children'),
              [Input('interval-component-slow', 'n_intervals')])
def update_graph_live(n):

	# config = twint.Config()
	# config.Search = 'PYPL'
	# config.Limit = 1000
	# config.Lang = "en"
	# if datetime.now().minute % 2 == 0:
	# 	config.Since = "2020-04-29"
	# else:
	# 	config.Since = "2020-04-28"

	# config.Until = "2020-04-30"
	# config.Custom["created_at"] = ["stamp"]#running search
	# config.Pandas = True
	# twint.run.Search(config)
	# dump = twint.storage.panda.Tweets_df


	consumer_key = "I6fZIQiiDsMKYBEbFMt0Xh2nL"
	consumer_secret = "pRXCiv4qfuoEZqM7y23Xvvh1ZbdHPzUMimxfdZq6ChyhzseRRJ"
	access_token = "390663782-OFgaApZzw5ox8knGxsLWeAjkBOIPkX3ZJDr2r6ri"
	access_token_secret = "CWjYjMYrvKXztDJG228JI3MTF33Lzq9ki7VwaznMwfEX3"
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth,wait_on_rate_limit=True)

	text_query = 'PYPL'
	count = 500
	try:
 # Creation of query method using parameters
		tweets = tweepy.Cursor(api.search,q=text_query,lang='en').items(count)
		tweets_list = [[tweet.created_at, tweet.id, tweet.text, tweet.user.location] for tweet in tweets]
		tweets_df = pd.DataFrame(tweets_list)
	except BaseException as e:
		print('failed on_status,',str(e))

	tweets_df.columns = ['date','tweet_id','tweet','location']	


	df = tweets_df
	finalized_dataframe = df[['date','tweet','location']]

	def clean_text(txt):
	#Remove URL
		txt = txt.lower()
		txt = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt, flags=re.MULTILINE)
		#Remove special characters
		txt = re.sub('[^A-Za-z0-9]+', ' ', txt)
		return txt


	finalized_dataframe['tweet'] = finalized_dataframe['tweet'].apply(lambda x: clean_text(x))
	finalized_dataframe['hour_mark'] = finalized_dataframe['date'].astype(str).apply(lambda x: x[11:13])

	def getPolarity(text):
		return TextBlob(text).sentiment.polarity


	finalized_dataframe['polarity'] = finalized_dataframe['tweet'].apply(lambda x: getPolarity(x))	

	def polarity_label_function(pol):
		if pol>0.5:
			return 'Pos'
		elif pol<-0.5:
			return 'Neg'
		else:
			return 'Neu'
    

	finalized_dataframe['polarity_label'] = finalized_dataframe['polarity'].apply(lambda x: polarity_label_function(x))


	finalized_dataframe.groupby(['polarity_label','hour_mark'])['polarity_label'].count().reset_index(name='count')

	temp = finalized_dataframe.groupby([pd.Grouper(key='hour_mark'), 'polarity_label']).count().unstack(fill_value=0).stack().reset_index()

	temp = temp[['hour_mark','polarity_label','date']]

	pos_df = temp.loc[temp['polarity_label']=='Pos']
	neg_df = temp.loc[temp['polarity_label']=='Neg']
	neu_df = temp.loc[temp['polarity_label']=='Neu']
	neg_df['date'] = neg_df['date'].apply(lambda x: x*-1)


    # Loading data from Heroku PostgreSQL
    # DATABASE_URL = os.environ['DATABASE_URL']
    # conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    # query = "SELECT id_str, text, created_at, polarity, user_location, user_followers_count FROM {}".format(settings.TABLE_NAME)
    # df = pd.read_sql(query, con=conn)


    # Convert UTC into PDT
    # df['created_at'] = pd.to_datetime(df['created_at']).apply(lambda x: x - datetime.timedelta(hours=7))

    # # Clean and transform data to enable time series
    # result = df.groupby([pd.Grouper(key='created_at', freq='10s'), 'polarity']).count().unstack(fill_value=0).stack().reset_index()
    # result = result.rename(columns={"id_str": "Num of '{}' mentions".format(settings.TRACK_WORDS[0]), "created_at":"Time"})  
    # time_series = result["Time"][result['polarity']==0].reset_index(drop=True)

    # min10 = datetime.datetime.now() - datetime.timedelta(hours=7, minutes=10)
    # min20 = datetime.datetime.now() - datetime.timedelta(hours=7, minutes=20)

    # neu_num = result[result['Time']>min10]["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==0].sum()
    # neg_num = result[result['Time']>min10]["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==-1].sum()
    # pos_num = result[result['Time']>min10]["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==1].sum()
    
    # # Loading back-up summary data
    # query = "SELECT daily_user_num, daily_tweets_num, impressions FROM Back_Up;"
    # back_up = pd.read_sql(query, con=conn)  
    # daily_tweets_num = back_up['daily_tweets_num'].iloc[0] + result[-6:-3]["Num of '{}' mentions".format(settings.TRACK_WORDS[0])].sum()
    # daily_impressions = back_up['impressions'].iloc[0] + df[df['created_at'] > (datetime.datetime.now() - datetime.timedelta(hours=7, seconds=10))]['user_followers_count'].sum()
    # cur = conn.cursor()

    # PDT_now = datetime.datetime.now() - datetime.timedelta(hours=7)
    # if PDT_now.strftime("%H%M")=='0000':
    #     cur.execute("UPDATE Back_Up SET daily_tweets_num = 0, impressions = 0;")
    # else:
    #     cur.execute("UPDATE Back_Up SET daily_tweets_num = {}, impressions = {};".format(daily_tweets_num, daily_impressions))
    # conn.commit()
    # cur.close()
    # conn.close()

    # Percentage Number of Tweets changed in Last 10 mins

    # Create the graph 



    #PIE CHARTS
	polarity_agg = finalized_dataframe.groupby('polarity_label')['polarity_label'].count().reset_index(name='count')
	neg_count = polarity_agg.loc[polarity_agg['polarity_label']=='Neg']['count']
	pos_count = polarity_agg.loc[polarity_agg['polarity_label']=='Pos']['count']
	neu_count = polarity_agg.loc[polarity_agg['polarity_label']=='Neu']['count']

	if len(neg_count) == 0:
		neg_count.at[0] = 0

	labels = ['Positive','Negative','Neutral']
	values = [pos_count.values[0], neg_count.values[0], neu_count.values[0]]

	#Creating Unigrams

	def generate_ngrams(text,n_gram=1):
		token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
		ngrams = zip(*[token[i:] for i in range(n_gram)])
		return [' '.join(ngram) for ngram in ngrams]
	unigrams = dict()
	for tweet in finalized_dataframe['tweet']:
		for w in generate_ngrams(tweet):
			try:
				unigrams[w] +=1
			except:
				unigrams[w] = 1

	unigrams_df = pd.DataFrame([unigrams]).T.reset_index().sort_values(by=0,ascending=False)
	unigrams_df = unigrams_df[unigrams_df['index']!='twitter']
	unigrams_df = unigrams_df[unigrams_df['index']!='pypl']
	unigrams_df = unigrams_df[unigrams_df['index']!='https']
	unigrams_df = unigrams_df[[(len(x) > 3) for x in unigrams_df['index']]]
	unigrams_df = unigrams_df.iloc[:15]

	#Engineering for Retweet/Likes TS plot
	#ƒnretweets_likes_df = finalized_dataframe.groupby('hour_mark')['nlikes','nretweets'].sum().reset_index()
	retweets_likes_df = pd.DataFrame({'hour_mark':[1],"nlikes":[2],"nretweets":[3]})

	#Barplot
	fig = go.Figure(go.Bar(x=unigrams_df['index'], y=unigrams_df[0], marker_color='lightblue'))

	fig.update_layout(
		title="Most Frequent Terms associated from the Twitter feed",
		xaxis_title="Terms",
		yaxis_title="Count",
		)


	#gn = geocoders.GeoNames(username='map_python')
	def get_lat(place):

		results = geocoder.geocode(place)
		try:
			return results[0]['geometry']['lat']
		except:
			return ""

	def get_long(place):

		results = geocoder.geocode(place)
		try:
			return results[0]['geometry']['lng']
		except:
			return ""

	list_of_places = finalized_dataframe['location'].unique()

	locs = pd.DataFrame(list_of_places)

	locs.columns = ['Location']

	locs['Lat'] = locs['Location'].apply(lambda x: get_lat(x))

	locs['Lon'] = locs['Location'].apply(lambda x: get_long(x))

	fig_map = px.density_mapbox(locs, lat='Lat', lon='Lon', radius=10,center=dict(lat=0, lon=180), zoom=0,mapbox_style="stamen-terrain")

	fig.update_layout(
		title="Tweet origins",
		)


	children = [
               
    		html.Div([
    				html.Div([

    					dcc.Graph(
    						id='crossfilter-indicator-scatter',
    						figure={
    						'data':[
    							go.Scatter(
    								x=neu_df['hour_mark'],
    								y=neu_df['date'],
    								fill='tozeroy',
    								name='Neutral',
    								line=dict(width=0.5, color='rgb(131, 90, 241)')
    								),
    							go.Scatter(
    								x=neg_df['hour_mark'],
    								y=neg_df['date'],
    								fill='tozeroy',
    								name='Negative',
    								line=dict(width=0.5, color='rgb(255, 50, 50)')
    								),
    							go.Scatter(
    								x=pos_df['hour_mark'],
    								y=pos_df['date'],
    								fill='tozeroy',
    								name='Positive',
    								line=dict(width=0.5, color='rgb(184, 247, 212)')
    								)

    						],
    						'layout':{
    						'title':'Polarity TimeSeries',
    						'xaxis_title':'Hour',
    						'yaxis_title':'Count'
    						}
    						}
    						)	

    					], style={'width': '73%', 'display': 'inline-block', 'padding': '0 0 0 20'}),

    				html.Div([
    					dcc.Graph(
    						id='pie-chart',
    						figure={
    						'data':[
    						go.Pie(labels=labels,values=values,hole=.54,marker_colors=['rgba(184, 247, 212, 0.6)','rgba(255, 50, 50, 0.6)','rgba(131, 90, 241, 0.6)'])
    						],
    						'layout':{
    						'showlegend':False,
    						'title':'Tweets Pie Chart',
    						'annotations':[
    						dict(
    							text='{0:.1f}K'.format((pos_count.values[0]+neg_count.values[0]+neu_count.values[0])/1000),
    							font=dict(
    								size=40
    								),
    								showarrow=False
    							)
    						]
    						}

    						}
    						)

    					], style={'width':'27%','display':'inline-block'}),

    				html.Div([
    					dcc.Graph(figure=fig)
    					], style={'width':'47%','display':'inline-block'}),

    				html.Div([
    					dcc.Graph(figure=fig_map)
    					], style={'width':'47%','display':'inline-block'}),


    				html.Div([

    					dcc.Graph(
    						id='rt-time-series',
    						figure={
    						'data':[
    							go.Scatter(
    								x=retweets_likes_df['hour_mark'],
    								y=retweets_likes_df['nlikes'],
    								fill='tozeroy',
    								name='Likes',
    								line=dict(width=0.5, color='rgb(131, 90, 241)')
    								),
    							go.Scatter(
    								x=retweets_likes_df['hour_mark'],
    								y=retweets_likes_df['nretweets'],
    								fill='tozeroy',
    								name='Retweets',
    								line=dict(width=0.5, color='rgb(184, 247, 212)')
    								)
    						],
    						'layout':{
    						'title':'Likes/RT TimeSeries',
    						'xaxis_title':'Hour',
    						'yaxis_title':'Count'
    						}
    						}
    						)	

    					], style={'width': '50%', 'display': 'inline-block', 'padding': '0 0 0 20'}),

					])
            ]
	return children






if __name__ == '__main__':
    app.run_server(debug=True)