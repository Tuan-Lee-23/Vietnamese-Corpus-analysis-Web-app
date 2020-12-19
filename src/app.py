import dash_bootstrap_components as dbc 
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output
import dash

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
corpus = ['cau 1 cau 1 cau 1 cau 1',
          'cau 2 cau 2 cau 2 cau 2',
          'cau 3 cau 3 cau 3 cau 3',
          'cau 4 cau 4 cau 4 cau 4',
          'cau 5 cau 5 cau 5 cau 5',
          'cau 6 cau 6 cau 6 cau 6']

def getSentences(corpus):
    result = [dbc.ListGroupItem(sen) for sen in corpus]
    return result

def returnHighlight(corpus, a):
    index = corpus.find(a)
    return index #tra ve vi tri de in
    
# checkbox = html.Div([
#     dcc.RadioItems(
#                 options = [
#                     {'label' : ' Noun ', 'value' : 'Noun'},
#                     {'label' : ' Word ', 'value' : 'Word'},
#                     {'label' : ' Adj ', 'value' : 'Adj'},
#                     {'label' : ' Morpheme ', 'value' : 'Morpheme'},
#                     {'label' : ' Verb ', 'value' : 'Verb'},
#                 ],
#                 value = 'Noun'
#             )
# ])

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
                options = [
                    {'label' : ' Noun ', 'value' : 'Noun'},
                    {'label' : ' Word ', 'value' : 'Word'},
                    {'label' : ' Adj ', 'value' : 'Adj'},
                    {'label' : ' Morpheme ', 'value' : 'Morpheme'},
                    {'label' : ' Verb ', 'value' : 'Verb'},
                ],
                value = 'Noun'
            )
])
], style={"width": "100%", 'textAlign': 'center', 'justify': 'center' })

# @app.callback(
#     Output(component_id='my-output', component_property='children'),
#     Input(component_id='my-input', component_property='value')
# )


def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

card = dbc.Card(
    dbc.ListGroup(
        [
            dbc.ListGroupItem("Item 1: aaaaaaaaaaaaaaaaaaaaa"),
            dbc.ListGroupItem("Item 2: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            dbc.ListGroupItem("Item 3: ccccccccccccccccccccccccccccccccccccc"),
        ],
        flush=True,
    ),
    style={"width": "100%", 'textAlign': 'center', 'justify': 'center'},
)

card = dbc.Card(
    dbc.ListGroup(
        getSentences(corpus[:10]),
        flush=True,
    ),
    style={"width": "100%", 'justify': 'center', 'textAlign': 'center'},
)





app.layout = html.Div([title, input, card])

if __name__ == "__main__":
    app.run_server(debug=True)