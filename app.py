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
    html.H2('Real-time Twitter Sentiment Analysis for Brand Improvement and Topic Tracking ', style={
        'textAlign': 'center'
    }),
    html.H4('(Last updated: Aug 23, 2019)', style={
        'textAlign': 'right'
    }),
    

    html.Div(id='live-update-graph'),
    html.Div(id='live-update-graph-bottom'),

    # Author's Words
    html.Div(
        className='row',
        children=[ 
            dcc.Markdown("__Author's Words__: Dive into the industry and get my hands dirty. That's why I start this self-motivated independent project. If you like it, I would appreciate for starring⭐️ my project on [GitHub](https://github.com/Chulong-Li/Real-time-Sentiment-Tracking-on-Twitter-for-Brand-Improvement-and-Trend-Recognition)!✨"),
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
                        href='https://github.com/Chulong-Li/Real-time-Sentiment-Tracking-on-Twitter-for-Brand-Improvement-and-Trend-Recognition'
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
                        'Chulong Li',
                        href='https://www.linkedin.com/in/chulong-li/'
                    )                    
                ]
            )                                                          
        ], style={'marginLeft': 70, 'fontSize': 16}
    ),

    dcc.Interval(
        id='interval-component-slow',
        interval=1*10000, # in milliseconds
        n_intervals=0
    )
    ], style={'padding': '20px'})



# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'children'),
              [Input('interval-component-slow', 'n_intervals')])
def update_graph_live(n):

	def fetch_tweets(query,count,start_time,stop_time):
	    config = twint.Config()
	    config.Search = query
	    config.Limit = count
	    config.Lang = "en"
	    config.Since = start_time
	    config.Until = stop_time
	    config.Custom["created_at"] = ["stamp"]#running search
	    #config.Store_csv = True
	    #config.Output = "none"
	    config.Pandas = True
	    twint.run.Search(config)
    	return twint.storage.panda.Tweets_df

    dump = fetch_tweets('PYPL',100,"2019-04-29","2020-04-30")
    df = dump
    finalized_dataframe = df[['date','tweet','nretweets','nlikes']]



    def clean_text(txt):
    #Remove URL
    	txt = txt.lower()
    	txt = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt, flags=re.MULTILINE)
    
    #Remove special characters
    	txt = re.sub('[^A-Za-z0-9]+', ' ', txt)
    
    	return txt


    finalized_dataframe['tweet'] = finalized_dataframe['tweet'].apply(lambda x: clean_text(x))

    finalized_dataframe['hour_mark'] = finalized_dataframe['date'].apply(lambda x: x[11:13])

    def getPolarity(text):
    
    	return TextBlob(text).sentiment.polarity

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
    children = [
               

    		html.Div([
    				html.Div([

    					dcc.graph(
    						id='crossfilter-indicator-scatter',
    						figure={
    						'data':[
    							go.scatter(
    								x=neu_df['hour_mark'],
    								y=neu_df['date'],
    								fill='tozeroy',
    								name='Neutral',
    								line=dict(width=0.5, color='rgb(131, 90, 241)')
    								),
    							go.scatter(
    								x=neg_df['hour_mark'],
    								y=neg_df['date'],
    								fill='tozeroy',
    								name='Negative',
    								line=dict(width=0.5, color='rgb(255, 50, 50)')
    								),
    							go.scatter(
    								x=pos_df['hour_mark'],
    								y=pos_df['date'],
    								fill='tozeroy',
    								name='Positive',
    								line=dict(width=0.5, color='rgb(184, 247, 212)')
    								)

    						]
    						}
    						)	

    					])

    			]style={'width': '73%', 'display': 'inline-block', 'padding': '0 0 0 20'})

            ]
    return children






if __name__ == '__main__':
    app.run_server(debug=True)