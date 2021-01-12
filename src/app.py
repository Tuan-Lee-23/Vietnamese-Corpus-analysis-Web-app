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
from statsmodels.graphics.gofplots import qqplot

import pandas as pd
import numpy as np

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

corpus = Corpus() # khởi tạo
dirr = "resources/corpus_mini.txt"  # chọn đường dẫn corpus
corpus.read(dirr)  # đọc đường dẫn
corpus.preprocess()   # Tiền xử lí
corpus.read_word2vec()  # Đọc model word2vec để tìm từ đồng nghĩa
corpus.read_ner()   # Đọc dữ liệu đã xử lí tên thực thể và từ loại

def searchAmbiguous(corpus, input):
   temp = corpus.search_ambiguous('miền Trung', case = 1)
   result = []
   for res in temp[:10]:
       result.append(res)
   return result
result_SA = searchAmbiguous(corpus, 1)
@app.callback(Input('label', 'value'),
            Output('RSA', 'result_SA')
)

def findSimilar(corpus, input):
    result = corpus.find_similar(input)
    return result

def findByPost(corpus, input):
    temp = corpus.search_by_pos('Việt', 'N')
    result = []
    for res in temp:
        result.append(res)
    return result
    
def findByNer(corpus, input, typeIn):
    temp = corpus.search_by_ner(input, typeIn)
    result = []
    for res in temp:
        result.append(res)
    return result


title = html.Div([
    html.H1('Vietnamese-Corpus-analysis-Web-app', className = 'display-5')
], style={'marginBottom': 50, 'marginTop': 25, 'justify': 'center', 'textAlign': 'center'})

input = html.Div([
    html.Div(["Input: ",
              dcc.Input(id='my-input', value='initial value', type='text')]),
    html.Br(),
    html.Div(id='my-output', style = {'marginBottom': 20, 'marginTop': 25,'justify': 'center'}),

    html.Div([
    dcc.RadioItems(
                id = 'pos',
                options = [
                    {'label' : ' Noun ', 'value' : 'N'},
                    {'label' : ' Word ', 'value' : 'W'},
                    {'label' : ' Adj ', 'value' : 'A'},
                    {'label' : ' Morpheme ', 'value' : 'M'},
                    {'label' : ' Verb ', 'value' : 'V'},
                ],
                value = 'N'
            )
])
], style={"width": "100%", 'textAlign': 'center', 'justify': 'center' })



def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

card = dbc.Card(
    dbc.ListGroup(
        id = 'RSA'
        ,
        flush=True,
    ),
    style={"width": "100%", 'textAlign': 'center', 'justify': 'center'},
)







app.layout = html.Div([title, input, card])

if __name__ == "__main__":
    app.run_server(host= '127.0.0.1', port = 80, debug= True)