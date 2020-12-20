import sys
from Corpus import Corpus
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go

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
len_sen_by_words = [len(x) for x in corpus.data_word_segment]
df['len_sentence_by_words'] = np.array(len_sen_by_words)

fig1 = px.histogram(df, x="len_sentence",
                    title = 'Phân phối độ dài câu (theo tiếng)',
                    labels={
                     "len_sentence": "Độ dài các câu"
                    }, 
                    height = 500).update(layout=dict(title=dict(x=0.5)))



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
                    height = 500).update(layout=dict(title=dict(x=0.5)))
    
    elif value == 1:
        fig1 = px.histogram(df, x="len_sentence_by_words",
            title = 'Phân phối độ dài câu (theo từ)',
            labels={
                "len_sentence_by_words": "Độ dài các câu"
            }, 
            height = 500).update(layout=dict(title=dict(x=0.5)))

    return fig1


# fig2---------------------------------------------
tong_tieng = df['len_sentence'].sum()
tong_tu = df['len_sentence_by_words'].sum()
tong_cau = len(corpus.data_sent_segment)

fig2 = go.Figure([go.Bar(x = ['Tổng câu', 'Tổng từ', 'Tổng tiếng'], 
                y = [tong_cau, tong_tu, tong_tieng])])

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

vocab_df = pd.DataFrame.from_dict(vocab, orient = 'index', columns = ['count'])

vocab_df = vocab_df.sort_values(by = 'count', ascending = False)
top_10_vocab = vocab_df.head(10)


fig3 = px.bar(top_10_vocab, x = top_10_vocab.index, y = 'count', 
                title = 'Top 10 từ phổ biến nhất (không tính stop words)').update(layout=dict(title=dict(x=0.5)))




statistics = [
    dbc.Row([
        dbc.Col(html.H1("PHÂN TÍCH VÀ THỐNG KÊ", className = 'text-center text-primary, mb-4'), width = 12)
    ]),
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
        ], style = {'textAlign': 'center'}),

        # Graph 2
        dbc.Col([
            dcc.Graph(
                id='fig2',
                figure = fig2
            )
        ])
    ], style = {'marginLeft': -38, 'marginRight': -38}),
    
    # Graph 3
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig3',
                figure = fig3
            )
        )
    ], style = {'marginTop': 50})
]


app.layout = dbc.Container(
    statistics
)

if __name__ == '__main__':
    app.run_server(debug=True, threaded = True)