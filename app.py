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

    # Loading data from Heroku PostgreSQL
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    query = "SELECT id_str, text, created_at, polarity, user_location, user_followers_count FROM {}".format(settings.TABLE_NAME)
    df = pd.read_sql(query, con=conn)


    # Convert UTC into PDT
    df['created_at'] = pd.to_datetime(df['created_at']).apply(lambda x: x - datetime.timedelta(hours=7))

    # Clean and transform data to enable time series
    result = df.groupby([pd.Grouper(key='created_at', freq='10s'), 'polarity']).count().unstack(fill_value=0).stack().reset_index()
    result = result.rename(columns={"id_str": "Num of '{}' mentions".format(settings.TRACK_WORDS[0]), "created_at":"Time"})  
    time_series = result["Time"][result['polarity']==0].reset_index(drop=True)

    min10 = datetime.datetime.now() - datetime.timedelta(hours=7, minutes=10)
    min20 = datetime.datetime.now() - datetime.timedelta(hours=7, minutes=20)

    neu_num = result[result['Time']>min10]["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==0].sum()
    neg_num = result[result['Time']>min10]["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==-1].sum()
    pos_num = result[result['Time']>min10]["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==1].sum()
    
    # Loading back-up summary data
    query = "SELECT daily_user_num, daily_tweets_num, impressions FROM Back_Up;"
    back_up = pd.read_sql(query, con=conn)  
    daily_tweets_num = back_up['daily_tweets_num'].iloc[0] + result[-6:-3]["Num of '{}' mentions".format(settings.TRACK_WORDS[0])].sum()
    daily_impressions = back_up['impressions'].iloc[0] + df[df['created_at'] > (datetime.datetime.now() - datetime.timedelta(hours=7, seconds=10))]['user_followers_count'].sum()
    cur = conn.cursor()

    PDT_now = datetime.datetime.now() - datetime.timedelta(hours=7)
    if PDT_now.strftime("%H%M")=='0000':
        cur.execute("UPDATE Back_Up SET daily_tweets_num = 0, impressions = 0;")
    else:
        cur.execute("UPDATE Back_Up SET daily_tweets_num = {}, impressions = {};".format(daily_tweets_num, daily_impressions))
    conn.commit()
    cur.close()
    conn.close()

    # Percentage Number of Tweets changed in Last 10 mins

    count_now = df[df['created_at'] > min10]['id_str'].count()
    count_before = df[ (min20 < df['created_at']) & (df['created_at'] < min10)]['id_str'].count()
    percent = (count_now-count_before)/count_before*100
    # Create the graph 
    children = [
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='crossfilter-indicator-scatter',
                            figure={
                                'data': [
                                    go.Scatter(
                                        x=time_series,
                                        y=result["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==0].reset_index(drop=True),
                                        name="Neutrals",
                                        opacity=0.8,
                                        mode='lines',
                                        line=dict(width=0.5, color='rgb(131, 90, 241)'),
                                        stackgroup='one' 
                                    ),
                                    go.Scatter(
                                        x=time_series,
                                        y=result["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==-1].reset_index(drop=True).apply(lambda x: -x),
                                        name="Negatives",
                                        opacity=0.8,
                                        mode='lines',
                                        line=dict(width=0.5, color='rgb(255, 50, 50)'),
                                        stackgroup='two' 
                                    ),
                                    go.Scatter(
                                        x=time_series,
                                        y=result["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==1].reset_index(drop=True),
                                        name="Positives",
                                        opacity=0.8,
                                        mode='lines',
                                        line=dict(width=0.5, color='rgb(184, 247, 212)'),
                                        stackgroup='three' 
                                    )
                                ]
                            }
                        )
                    ], style={'width': '73%', 'display': 'inline-block', 'padding': '0 0 0 20'}),
                    
                    html.Div([
                        dcc.Graph(
                            id='pie-chart',
                            figure={
                                'data': [
                                    go.Pie(
                                        labels=['Positives', 'Negatives', 'Neutrals'], 
                                        values=[pos_num, neg_num, neu_num],
                                        name="View Metrics",
                                        marker_colors=['rgba(184, 247, 212, 0.6)','rgba(255, 50, 50, 0.6)','rgba(131, 90, 241, 0.6)'],
                                        textinfo='value',
                                        hole=.65)
                                ],
                                'layout':{
                                    'showlegend':False,
                                    'title':'Tweets In Last 10 Mins',
                                    'annotations':[
                                        dict(
                                            text='{0:.1f}K'.format((pos_num+neg_num+neu_num)/1000),
                                            font=dict(
                                                size=40
                                            ),
                                            showarrow=False
                                        )
                                    ]
                                }

                            }
                        )
                    ], style={'width': '27%', 'display': 'inline-block'})
                ]),
                
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            children=[
                                html.P('Tweets/10 Mins Changed By',
                                    style={
                                        'fontSize': 17
                                    }
                                ),
                                html.P('{0:.2f}%'.format(percent) if percent <= 0 else '+{0:.2f}%'.format(percent),
                                    style={
                                        'fontSize': 40
                                    }
                                )
                            ], 
                            style={
                                'width': '20%', 
                                'display': 'inline-block'
                            }

                        ),
                        html.Div(
                            children=[
                                html.P('Potential Impressions Today',
                                    style={
                                        'fontSize': 17
                                    }
                                ),
                                html.P('{0:.1f}K'.format(daily_impressions/1000) \
                                        if daily_impressions < 1000000 else \
                                            ('{0:.1f}M'.format(daily_impressions/1000000) if daily_impressions < 1000000000 \
                                            else '{0:.1f}B'.format(daily_impressions/1000000000)),
                                    style={
                                        'fontSize': 40
                                    }
                                )
                            ], 
                            style={
                                'width': '20%', 
                                'display': 'inline-block'
                            }
                        ),
                        html.Div(
                            children=[
                                html.P('Tweets Posted Today',
                                    style={
                                        'fontSize': 17
                                    }
                                ),
                                html.P('{0:.1f}K'.format(daily_tweets_num/1000),
                                    style={
                                        'fontSize': 40
                                    }
                                )
                            ], 
                            style={
                                'width': '20%', 
                                'display': 'inline-block'
                            }
                        ),

                        html.Div(
                            children=[
                                html.P("Currently tracking \"Facebook\" brand (NASDAQ: FB) on Twitter in Pacific Daylight Time (PDT).",
                                    style={
                                        'fontSize': 25
                                    }
                                ),
                            ], 
                            style={
                                'width': '40%', 
                                'display': 'inline-block'
                            }
                        ),

                    ],
                    style={'marginLeft': 70}
                )
            ]
    return children






if __name__ == '__main__':
    app.run_server(debug=True)