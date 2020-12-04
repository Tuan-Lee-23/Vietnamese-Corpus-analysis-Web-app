import dash_bootstrap_components as dbc 
import dash_html_components as html 
import dash

corpus = ['cau 1 cau 1 cau 1 cau 1',
          'cau 2 cau 2 cau 2 cau 2',
          'cau 3 cau 3 cau 3 cau 3',
          'cau 4 cau 4 cau 4 cau 4',
          'cau 5 cau 5 cau 5 cau 5',
          'cau 6 cau 6 cau 6 cau 6']

def getSentences(corpus):
    result = [dbc.ListGroupItem(sen) for sen in corpus]
    return result


title = html.Div([
    html.H1('Example Div', className = 'display-5')
], style={'marginBottom': 50, 'marginTop': 25, 'justify': 'center', 'textAlign': 'center'})

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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([title, card])

if __name__ == "__main__":
    app.run_server(debug=True)