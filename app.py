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

import nest_asyncio
nest_asyncio.apply()
import twint
from datetime import datetime
import pandas as pd
import numpy
import re
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import itertools    

stopwords = set(STOPWORDS)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Real-Time Twitter Monitor'

server = app.server

app.layout = html.Div(children = [
		html.H2('Real-time Twitter Sentiment Analysis', style = {
			'textAlign':'center'
			}),
		html.H4('Dev by', style={
		'textAlign':'right'
		})

		html.Div(id='live-update-graph'),

		dcc.Interval(
        id='interval-component-slow',
        interval=1*10000, # in milliseconds
        n_intervals=0
    )
	],style={'padding':'20px'})






if __name__ == '__main__':
    app.run_server(debug=True)