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
dirr = "resources/vn_express.txt"  # chọn đường dẫn corpus
corpus.read(dirr)  # đọc đường dẫn
corpus.preprocess()   # Tiền xử lí
corpus.read_word2vec()  # Đọc model word2vec để tìm từ đồng nghĩa
corpus.read_ner()   # Đọc dữ liệu đã xử lí tên thực thể và từ loại

def genResult(res):
    result = [html.P(children = sen, style = {'backgroundColor': 'white', 'borderBottom': '2px solid #4F2992', 'margin': '30px', 'padding': '10px'}) for sen in res]
    res.append(html.Br())
    return result


@app.callback(Output('RSA', 'children'),
            Input('pos', 'value'),
            Input('input_val', 'value'),
)
def handle(pos, input_val):
    if pos == 'AM':
        res = corpus.search_ambiguous(input_val)
    elif pos == 'N':
        temp = corpus.search_by_pos(input_val, 'N')
        res = [x[0] for x in temp]
    elif pos == 'V':
        temp = corpus.search_by_pos(input_val, 'V')
        res = [x[0] for x in temp]
    elif pos == 'A':
        temp = corpus.search_by_pos(input_val, 'A')
        res = [x[0] for x in temp]
    elif pos == 'FW':
        temp = corpus.search_by_pos(input_val, 'FW')
        res = [x[0] for x in temp]
    elif pos == 'PER':
        temp = corpus.search_by_ner(input_val, 'PER')
        res = [x[0] for x in temp]
    elif pos == 'LOC':
        temp = corpus.search_by_ner(input_val, 'LOC')
        res = [x[0] for x in temp]
    elif pos == 'ORG':
        temp = corpus.search_by_ner(input_val, 'ORG')
        res = [x[0] for x in temp]


    if len(input_val) == 0:
        return genResult(res[:30])
    elif len(res) == 0:
        return ""
    else:
        # return [html.P(children = res[0]), html.P("I am going to the beach baby")]
        return genResult(res)

@app.callback(Output('stats', 'children'),
            Input('pos', 'value'),
            Input('input_val', 'value'),
)
def handle2(pos, input_val):
    if pos == 'AM':
        res = corpus.search_ambiguous(input_val)
    elif pos == 'N':
        temp = corpus.search_by_pos(input_val, 'N')
        res = [x[0] for x in temp]
    elif pos == 'V':
        temp = corpus.search_by_pos(input_val, 'V')
        res = [x[0] for x in temp]
    elif pos == 'A':
        temp = corpus.search_by_pos(input_val, 'A')
        res = [x[0] for x in temp]
    elif pos == 'FW':
        temp = corpus.search_by_pos(input_val, 'FW')
        res = [x[0] for x in temp]
    elif pos == 'PER':
        temp = corpus.search_by_ner(input_val, 'PER')
        res = [x[0] for x in temp]
    elif pos == 'LOC':
        temp = corpus.search_by_ner(input_val, 'LOC')
        res = [x[0] for x in temp]
    elif pos == 'ORG':
        temp = corpus.search_by_ner(input_val, 'ORG')
        res = [x[0] for x in temp]

    if len(input_val) == 0 or len(res) == 0:
        return '0/' + str(len(corpus.data))
    else:
        return str(len(res)) + '/' + str(len(corpus.data))
    
@app.callback(Output('similar', 'children'),
            Input('input_val', 'value'),
)

def handle3(input_val):
    if len(input_val) == 0:
        return ''
    else:
        top10_similar = corpus.find_similar(input_val)
        top10_similar = dict(top10_similar)

        for key in top10_similar: 
            top10_similar[key] = round(top10_similar[key], 3) 

        res = str(top10_similar).strip('{}')
        return res


title = dbc.Col(html.H1("TRA CỨU", className = 'text-center'), 
        width = 12, style = {'background': '#4F2992', 'color': 'white', 'padding': 8, 'marginBottom': 5})


input = html.Div([
    html.Div([
        dcc.Input(id='input_val', value='', type='text', style = {'padding': '10px 20px', 'marginTop': '20vh', 'marginBottom': '5vh', 'width': '25vw', 'border': '0', 'borderBottom': '3.5px solid #4F2992'})]),
    html.Br(),  
    html.Div([
        html.Div(id = 'similar', style = {'marginBottom': '20px'}),
        dcc.RadioItems(
                    id = 'pos',
                    options = [
                        {'label' : ' Ambiguous ', 'value' : 'AM'},
                        {'label' : ' Noun ', 'value' : 'N'},
                        {'label' : ' Verb ', 'value' : 'V'},
                        {'label' : ' Adj ', 'value' : 'A'},
                        {'label' : 'Foreign words', 'value' : 'FW'},
                        {'label' : 'Person', 'value' : 'PER'},
                        {'label' : 'Location', 'value' : 'LOC'},
                        {'label' : 'Org', 'value' : 'ORG'},
                    ],
                    labelStyle = {'marginRight': '10px'},
                    value = 'AM')
    ])
], style={"width": "100%", 'textAlign': 'center', 'justify': 'center' })



def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

card = dbc.Card([
    # dbc.ListGroup(
    #     id = 'RSAS'
    #     ,
    #     flush=True,
    # ),
    # style={"width": "100%", 'textAlign': 'center', 'justify': 'center'},
    html.P(id = 'stats', style = {'textAlign': 'center', 'color': '#4F2992', 'fontSize': '18px', 'marginTop': '10px', 'fontWeight': 'bold'}),
    html.P(id = 'RSA', style={"width": "100%", 'textAlign': 'center', 'justify': 'center'})]
    , style = {'backgroundColor': '#ededed'}
)







app.layout = html.Div([title, input, card])

if __name__ == "__main__":
    app.run_server(debug= True)