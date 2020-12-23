import sys
from Corpus import Corpus
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

corpus = Corpus()
corpus.read('resources/corpus_mini.txt')
corpus.preprocess()
# corpus.read_word2vec()

df = pd.DataFrame(corpus.data_sent_segment, columns = ['sentences'])

# # đếm theo tiếng
# df['len_tieng'] = df['sentences'].str.split()
# df['len_tieng'] = df['len_tieng'].apply(len)

# so_tieng = df['len_tieng'].sum()
# cau_tu_tieng = pd.DataFrame(np.array([len(corpus.data_sent_segment), len(corpus.vocab.keys()), so_tieng]))
# print(cau_tu_tieng)

# Độ dài câu theo tiếng
df['sentence_split'] = df['sentences'].str.split()
df['len_sentence'] = df['sentence_split'].apply(len)

# Độ dài câu theo từ
len_sen_by_words = []
for temp in corpus.data_word_segment:
    len_sen_by_words.append(len(temp))
df['len_sentence_by_words'] = np.array(len_sen_by_words)

# Fig 1
fig1 = px.histogram(df, x="len_sentence", nbins = 30,
                    title = 'Phân phối độ dài câu (theo tiếng)',
                    labels={
                     "len_sentence": "Độ dài các câu"
                    }, 
                    histnorm='probability density',
                    marginal= 'violin',
                    color_discrete_sequence= ['#4F2992'],
                    height = 500).update(layout=dict(title=dict(x=0.5)))

# Fig 1_2
def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

x, y = ecdf(df['len_sentence_by_words'])
df_ecdf = pd.DataFrame({'len' :x, 'ecdf': y})

fig1_2 = px.scatter(df_ecdf, 'len', 'ecdf', 
                 title = 'Phân phối độ dài câu (theo tiếng)',
                 color= 'ecdf',
                 color_continuous_scale= 'AgSunset',
                 labels= {
                     'len': 'Độ dài câu'
                 }, height = 500)

fig1_2.update(layout=dict(title=dict(x=0.5)))
fig1_2.update_layout(showlegend = False)


@app.callback(
    Output("fig1", "figure"), 
    Input("sen_choose", 'value')
)

def update_graph1(value):
    if value == 0:
        fig1 = px.histogram(df, x="len_sentence", nbins = 30,
                    title = 'Phân phối độ dài câu (theo tiếng)',
                    labels={
                     "len_sentence": "Độ dài các câu"
                    }, 
                    histnorm='probability density',
                    marginal= 'box',
                    color_discrete_sequence= ['#4F2992'],
                    height = 500).update(layout=dict(title=dict(x=0.5)))
    
    elif value == 1:
        fig1 = px.histogram(df, x="len_sentence_by_words",
            title = 'Phân phối độ dài câu (theo từ)',
            labels={
                "len_sentence_by_words": "Độ dài các câu"
            }, 
            histnorm='probability density',
            marginal= 'box',
            color_discrete_sequence= ['#4F2992'],
            height = 500).update(layout=dict(title=dict(x=0.5)))

    return fig1

@app.callback(
    Output("fig1_2", "figure"), 
    Input("sen_choose", 'value')
)
def update_graph1_2(value):
    if value == 1:
        x, y = ecdf(df['len_sentence_by_words'])
        df_ecdf = pd.DataFrame({'len' :x, 'ecdf': y})
        fig1_2 = px.scatter(df_ecdf, 'len', 'ecdf', 
                title = 'Phân phối độ dài câu (theo từ)',
                color= 'ecdf',
                color_continuous_scale= 'AgSunset',
                labels= {
                    'len': 'Độ dài câu'
                }, height = 500)
    else:
        x, y = ecdf(df['len_sentence'])
        df_ecdf = pd.DataFrame({'len' :x, 'ecdf': y})
        fig1_2 = px.scatter(df_ecdf, 'len', 'ecdf', 
                title = 'Phân phối độ dài câu (theo tiếng)',
                color= 'ecdf',
                color_continuous_scale= 'AgSunset',
                labels= {
                    'len': 'Độ dài câu'
                }, height = 500)
    fig1_2.update(layout=dict(title=dict(x=0.5)))
    fig1_2.update_layout(showlegend = False)
    
    return fig1_2

# fig2---------------------------------------------
tong_tieng = df['len_sentence'].sum()
tong_tu = df['len_sentence_by_words'].sum()
tong_cau = len(corpus.data_sent_segment)

fig2 = go.Figure([go.Bar(x = ['Tổng câu', 'Tổng từ', 'Tổng tiếng'], 
                y = [tong_cau, tong_tu, tong_tieng],
                marker_color = ['#F89378', '#B1339E', '#642A98'])])

fig2.update_layout(
    title="Tổng số câu / từ / tiếng", 
    height = 500
)
fig2.update(layout=dict(title=dict(x=0.5)))


# fig 3----------------------------------------------
from collections import Counter
import itertools

def itertools_chain(a):
    return list(itertools.chain.from_iterable(a))

vocab = itertools_chain(corpus.data_word_segment)
vocab = dict(Counter(vocab))
sample_space = sum(vocab.values())

# Create dataframe
vocab_df = pd.DataFrame.from_dict(vocab, orient = 'index', columns = ['count'])

# Sort
vocab_df = vocab_df.sort_values(by = 'count', ascending = False)

vocab_df.loc[:, 'count'] = vocab_df['count'] / sample_space

top_10_vocab = vocab_df.head(10)

def plotBar(df, xs, ys, title):
    return px.bar(df, x = xs, y = ys, 
                title = title,
                color_discrete_sequence= ['#4F2992']
                ).update(layout=dict(title=dict(x=0.5)))


fig3 = plotBar(top_10_vocab, top_10_vocab.index, 'count',
              title = 'Top 10 từ phổ biến nhất (n - grams)')


# fig 3_2
vocab_df = vocab_df.reset_index()
vocab_df['gram_len'] = vocab_df['index'].apply(lambda x: len(x.split()))

top_10_vocab_bigrams = vocab_df[vocab_df['gram_len'] == 2]
top_10_vocab_bigrams = top_10_vocab_bigrams.head(10)

fig3_2 = plotBar(top_10_vocab_bigrams, 'index', 'count', title = 'Top 10 từ phổ biến nhất (bi-grams)')

# fig 3_3
top_10_vocab_trigrams = vocab_df[vocab_df['gram_len'] == 3]
top_10_vocab_trigrams = top_10_vocab_trigrams.head(10)

fig3_3 = plotBar(top_10_vocab_trigrams, 'index', 'count', title = 'Top 10 từ phổ biến nhất (tri-grams)')

statistics = [

    dbc.Col(html.H1("PHÂN TÍCH VÀ THỐNG KÊ", className = 'text-center'), 
        width = 12, style = {'background': '#4F2992', 'color': 'white', 'padding': 8, 'marginBottom': 5}),
    # graph 1 + 2
    dbc.Row([

        # graph 1
        dbc.Col([
            dcc.Graph(
                id='fig1'
            ), 
            dcc.RadioItems(
                    id='sen_choose',
                    options= [
                        {'label': 'tiếng', 'value': 0},
                        {'label': 'từ', 'value': 1}
                    ],
                    value= 0,
                    labelStyle={'display': 'inline-block', 'paddingLeft': 10, 'textAlign': 'center'}
            )
        ], style = {'textAlign': 'center'}, width = 6), 

        # graph 1_2
        dbc.Col(
            dcc.Graph(
                id='fig1_2',
                figure = fig1_2
            ))
    ], style = {'marginLeft': -52, 'marginRight': -52}),

    # Graph 2
    dbc.Row(        
        dbc.Col(
            dcc.Graph(
                id='fig2',
                figure = fig2
            )
        ), style = {'marginLeft': -38, 'marginRight': -38}),

    # Graph 3
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig3',
                figure = fig3
            )
        )
    ], style = {'marginTop': 50}),

    # Graph 3_2
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig3_2',
                figure = fig3_2
            )
        )
    ], style = {'marginTop': 50}),

    # Graph 3_3
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig3_3',
                figure = fig3_3
            )
        )
    ], style = {'marginTop': 50})
]


app.layout = dbc.Container(
    statistics
)

if __name__ == '__main__':
    app.run_server(debug=True, threaded = True)