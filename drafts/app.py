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



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

corpus = Corpus()
corpus.read('resources/corpus_mini.txt')
corpus.preprocess()
corpus.read_ner()
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
df = df[df['len_sentence_by_words'] > 0]

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

# Graph 1_3


fig1_3 = go.Figure()

@app.callback(
    Output("fig1_3", "figure"), 
    Input("sen_choose", 'value')
)
def update_graph1_3(value):
    if value == 0:
        qqplot_data= qqplot(df['len_sentence'], line = 's').gca().lines
        fig1_3 = go.Figure()
    else:
        qqplot_data = qqplot(df['len_sentence_by_words'], line = 's').gca().lines
        fig1_3 = go.Figure()
    
    fig1_3.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#4F2992'
        }
    })

    fig1_3.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#B1339E'
        }

    })


    fig1_3['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 1000,
        'height': 700,
    })
   
    fig1_3.update(layout=dict(title=dict(x=0.5)))

    return fig1_3




# fig2---------------------------------------------
stop_words = set()
with open("resources/stop_words.txt", encoding = 'utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.add(line.rstrip('\n'))

tong_cau = len(corpus.data_sent_segment)
tong_tieng = df['sentence_split'].to_numpy()
tong_tieng = np.array([np.array(x) for x in tong_tieng])
temp = []

for vec in tong_tieng:
    for word in vec:
        # if word not in stop_words:
            # temp.append(word.lower())
        temp.append(word.lower())

tong_tieng = len(set(temp))

# print(len(temp))
# print(len(set(temp)))

# Step of couting tieng: vẽ scatter plot show ra tiến trình nhận được số tiếng giảm dần
step_count = 0
step_np = []
tieng_temp = set()
for i, tieng in enumerate(temp):
    if tieng not in tieng_temp:
        tieng_temp.add(tieng)
        step_count += 1
        step_np.append(np.array([i + 1, step_count]))
step_np = np.array(step_np)
step_df = pd.DataFrame({'so luong tieng': step_np[:, 0], 'so luong tieng (ko trung)': step_np[:, 1]})

#----------------

temp = []
for vec in corpus.data_word_segment:
    for word in vec:
        temp.append(word)
dictionary = temp
tong_tu = len(set(temp))

step_count = 0
step_np_tu = []
tu_temp = set()
for i, tu in enumerate(temp):
    if tu not in tu_temp:
        tu_temp.add(tu)
        step_count += 1
        step_np_tu.append(np.array([i + 1, step_count]))
step_np_tu = np.array(step_np_tu)
step_df_tu = pd.DataFrame({'so luong tu': step_np_tu[:, 0], 'so luong tu (ko trung)': step_np_tu[:, 1]})




fig2 = go.Figure([go.Bar(x = ['Tổng câu', 'Tổng từ', 'Tổng tiếng'], 
                y = [tong_cau, tong_tu, tong_tieng],
                marker_color = ['#F89378', '#B1339E', '#642A98'])])

fig2.update_layout(
    title="Tổng số câu / từ / tiếng", 
    height = 500
)
fig2.update(layout=dict(title=dict(x=0.5)))

# fig2_1
fig2_1 = px.line(step_df, 'so luong tieng', 'so luong tieng (ko trung)', 
                title = 'Sự tăng tiến về số lượng của tiếng',
                color_discrete_sequence = ['#4F2992'],
                height = 500,
                labels = {'so luong tieng': 'Số lượng tiếng',
                          'so luong tieng (ko trung)': 'Số lượng tiếng (không trùng)'}).update(layout=dict(title=dict(x=0.5)))

fig2_1_tu = px.line(step_df_tu, 'so luong tu', 'so luong tu (ko trung)', 
                title = 'Sự tăng tiến về số lượng của từ',
                color_discrete_sequence = ['#4F2992'],
                height = 500,
                labels = {'so luong tu': 'Số lượng từ',
                          'so luong tu (ko trung)': 'Số lượng từ (không trùng)'}).update(layout=dict(title=dict(x=0.5)))

# fig 2_2
fig2_2 = px.scatter(df, 'len_sentence','len_sentence_by_words', color = 'len_sentence_by_words',
                    title = 'Độ dài câu theo tiếng vs độ dài câu theo từ',
                    height = 500,
                    labels = {'len_sentence': 'Độ dài câu theo tiếng', 
                              'len_sentence_by_words': 'Độ dài câu theo từ'}).update(layout=dict(title=dict(x=0.5)))

# fig 2_3
temp = df.groupby('len_sentence')['len_sentence_by_words'].mean().reset_index()
fig2_3 = px.scatter(temp, 'len_sentence', 'len_sentence_by_words', color = 'len_sentence_by_words',
                    title = 'Độ dài câu theo tiếng vs trung bình độ dài câu theo từ',
                    height = 500,
                    labels = {'len_sentence': 'Độ dài câu theo tiếng', 
                              'len_sentence_by_words': 'Trung bình độ dài câu theo từ'}).update(layout=dict(title=dict(x=0.5)))



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
                color_discrete_sequence= ['#4F2992'],
                labels = {'index': 'Từ', 
                              'count': str('Tần suất xuất hiện (' + str(sample_space)+')')
                        }).update(layout=dict(title=dict(x=0.5)))


fig3 = plotBar(top_10_vocab, top_10_vocab.index, 'count',
              title = 'Top 10 từ phổ biến nhất (n - tiếng)')


# fig 3_2
vocab_df = vocab_df.reset_index()
vocab_df['gram_len'] = vocab_df['index'].apply(lambda x: len(x.split()))

top_10_vocab_bigrams = vocab_df[vocab_df['gram_len'] == 2]
top_10_vocab_bigrams = top_10_vocab_bigrams.head(10)

fig3_2 = plotBar(top_10_vocab_bigrams, 'index', 'count', title = 'Top 10 từ phổ biến nhất (2 tiếng)')

# fig 3_3
top_10_vocab_trigrams = vocab_df[vocab_df['gram_len'] == 3]
top_10_vocab_trigrams = top_10_vocab_trigrams.head(10)

fig3_3 = plotBar(top_10_vocab_trigrams, 'index', 'count', title = 'Top 10 từ phổ biến nhất (3 tiếng)')

# fig 3_3
top_10_vocab_4grams = vocab_df[vocab_df['gram_len'] == 4]
top_10_vocab_4grams = top_10_vocab_4grams.head(10)

fig3_4 = plotBar(top_10_vocab_4grams, 'index', 'count', title = 'Top 10 từ phổ biến nhất (4 tiếng)')

# tu dai nhat
longest_word = max(dictionary, key = len)
len_longest = len(corpus.search_ambiguous(longest_word))


# fig 4
stop_words_dict = corpus.stopwords
# print(sorted(stop_words_dict))
stop_words_df = pd.DataFrame.from_dict(stop_words_dict, orient = 'index', columns = ['count'])
stop_words_df = stop_words_df.sort_values('count', ascending= False).reset_index().head(10)

sample_space_sw = stop_words_df['count'].sum()
stop_words_df['count'] = stop_words_df['count'] / sample_space_sw

fig4_1 = plotBar(stop_words_df, 'index', 'count', 'Top 10 stop-words')
fig4_1.update_layout(
    xaxis_title = 'Từ',
    yaxis_title = 'Xác suất xuất hiện (' + str(sample_space_sw) + ')'
)

# fig 5
def getPOS(data, type):
    pos_container = []
    pos_count = []
    for i, sen in enumerate(data):
        count = 0
        # print(sen)
        if sen: 
            for word in sen:
                if word[1] == type:
                    pos_container.append(word[0].lower())
                    count += 1
        # print(count)
        pos_count.append(count)
    
    return pos_container, pos_count

nouns, nouns_count = getPOS(corpus.data_pos_tagged, 'N')
verbs, verbs_count = getPOS(corpus.data_pos_tagged, 'V')
adj, adj_count = getPOS(corpus.data_pos_tagged, 'A')

nouns_set = set(nouns)
verbs_set = set(verbs)
adj_set = set(adj)


pos_df = pd.DataFrame({'type': ['danh từ', 'động từ', 'tính từ'], 
                    'count': [len(nouns_set), len(verbs_set), len(adj_set)]})

noun_hist = pd.DataFrame({'count': nouns_count})

fig5_1 = plotBar(pos_df, 'type', 'count', 'Tổng số danh từ, động từ, tính từ')
fig5_1.update_layout(
    xaxis_title = 'Từ loại',
    yaxis_title = 'Số lượng'
)
# fig5_1.show()


fig5_2 = px.histogram(noun_hist, x = "count", nbins = int(np.ceil(np.sqrt(len(nouns_count))).astype('int')),
                    title = 'Phân phối số lượng danh từ trong 1 câu',
                    labels={
                     "count": "số lượng"
                    }, 
                    histnorm='probability density',
                    marginal= 'box',
                    color_discrete_sequence= ['#4F2992'],
                    height = 500).update(layout=dict(title=dict(x=0.5)))


# fig5_2.show()

#fig5_3
verb_hist = pd.DataFrame({'count': verbs_count})

fig5_3 = px.histogram(verb_hist, x = "count", nbins = int(np.ceil(np.sqrt(len(verbs_count))).astype('int')),
                    title = 'Phân phối số lượng động từ trong 1 câu',
                    labels={
                     "count": "số lượng"
                    }, 
                    histnorm='probability density',
                    marginal= 'box',
                    color_discrete_sequence= ['#4F2992'],
                    height = 500).update(layout=dict(title=dict(x=0.5)))

#fig5_4
adj_hist = pd.DataFrame({'count': adj_count})

fig5_4 = px.histogram(adj_hist, x = "count", nbins = int(np.ceil(np.sqrt(len(adj_count))).astype('int')),
                    title = 'Phân phối số lượng tính từ trong 1 câu',
                    labels={
                     "count": "số lượng"
                    }, 
                    histnorm='probability density',
                    marginal= 'box',
                    color_discrete_sequence= ['#4F2992'],
                    height = 500).update(layout=dict(title=dict(x=0.5)))

# fig 6_1
df = pd.DataFrame(corpus.data_sent_segment, columns = ['sentences'])

# Độ dài câu theo tiếng
df['sentence_split'] = df['sentences'].str.split()
df['len_sentence'] = df['sentence_split'].apply(len)

df['nouns_count'] = nouns_count
df['verbs_count'] = verbs_count
df['adj_count'] = adj_count

# Độ dài câu theo từ
len_sen_by_words = []
for temp in corpus.data_word_segment:
    len_sen_by_words.append(len(temp))
df['len_sentence_by_words'] = np.array(len_sen_by_words)

df = df[df['len_sentence_by_words'] > 0]


fig6_1 = px.scatter(df, 'len_sentence_by_words', 'nouns_count', color = 'len_sentence_by_words',
                    title = 'Độ dài câu theo từ vs số lượng danh từ',
                    height = 500,
                    labels = {'len_sentence_by_words': 'Độ dài câu theo từ', 
                              'nouns_count': 'Số lượng danh từ'}).update(layout=dict(title=dict(x=0.5)))


# fig 6_2
temp = df.groupby('len_sentence_by_words')['nouns_count'].median().reset_index()
fig6_2 = px.scatter(temp, 'len_sentence_by_words', 'nouns_count', color = 'len_sentence_by_words',
                    title = 'Độ dài câu theo từ vs trung bình số lượng danh từ',
                    height = 500,
                    labels = {'len_sentence_by_words': 'Độ dài câu theo từ', 
                              'nouns_count': 'Trung bình số lượng danh từ'}).update(layout=dict(title=dict(x=0.5)))


# fig 6_3
fig6_3 = px.scatter(df, 'len_sentence_by_words', 'verbs_count', color = 'len_sentence_by_words',
                    title = 'Độ dài câu theo từ vs số lượng động từ',
                    height = 500,
                    labels = {'len_sentence_by_words': 'Độ dài câu theo từ', 
                              'nouns_count': 'Số lượng động từ'}).update(layout=dict(title=dict(x=0.5)))

# fig 6_4
temp = df.groupby('len_sentence_by_words')['verbs_count'].median().reset_index()
fig6_4 = px.scatter(temp, 'len_sentence_by_words', 'verbs_count', color = 'len_sentence_by_words',
                    title = 'Độ dài câu theo từ vs trung bình số lượng động từ',
                    height = 500,
                    labels = {'len_sentence_by_words': 'Độ dài câu theo từ', 
                              'nouns_count': 'Trung bình số lượng động từ'}).update(layout=dict(title=dict(x=0.5)))

# fig 6_5
fig6_5 = px.scatter(df, 'len_sentence_by_words', 'adj_count', color = 'len_sentence_by_words',
                    title = 'Độ dài câu theo từ vs số lượng động từ',
                    height = 500,
                    labels = {'len_sentence_by_words': 'Độ dài câu theo từ', 
                              'nouns_count': 'Số lượng động từ'}).update(layout=dict(title=dict(x=0.5)))

# # fig 6_6
# temp = df.groupby('len_sentence_by_words')['adj_count'].median().reset_index()
# fig6_6 = px.scatter(temp, 'len_sentence_by_words', 'adj_count', color = 'len_sentence_by_words',
#                     title = 'Độ dài câu theo từ vs trung bình số lượng tính từ',
#                     height = 500,
#                     labels = {'len_sentence_by_words': 'Độ dài câu theo từ', 
#                               'nouns_count': 'Trung bình số lượng tính từ'}).update(layout=dict(title=dict(x=0.5)))


# fig 6_7: histogram danh từ, động từ, tính từ có trung bình bao nhiêu tiếng
nouns_len = np.array([len(x.split()) for x in nouns])
nouns_df = pd.DataFrame({'count': nouns_len})
sample = len(nouns_len)

fig6_7 = px.histogram(nouns_df, x = "count", nbins = 10,
                    title = 'Phân phối số lượng tiếng của danh từ ' + '(' + str(sample) + ')',
                    histnorm='probability density',
                    color_discrete_sequence= ['#4F2992'],
                    height = 500).update(layout=dict(title=dict(x=0.5)))

# fig 6_8
verbs_len = np.array([len(x.split()) for x in verbs])
verbs_df = pd.DataFrame({'count': verbs_len})
sample = len(verbs_len)

fig6_8 = px.histogram(verbs_df, x = "count", nbins = 10,
                    title = 'Phân phối số lượng tiếng của động từ ' + '(' + str(sample) + ')',
                    histnorm='probability density',
                    color_discrete_sequence= ['#4F2992'],
                    height = 500).update(layout=dict(title=dict(x=0.5)))


# fig 6_9
adj_len = np.array([len(x.split()) for x in adj])
adj_df = pd.DataFrame({'count': adj_len})
sample = len(adj_len)

fig6_9 = px.histogram(adj_df, x = "count", nbins = 4,
                    title = 'Phân phối số lượng tiếng của tính từ ' + '(' + str(sample) + ')',
                    histnorm='probability density',
                    color_discrete_sequence= ['#4F2992'],
                    height = 500).update(layout=dict(title=dict(x=0.5)))

#fig 7 top ten thuc the
def getNER(data, type):
    ner_container = []
    for sen in data:
        # print(sen)
        if sen: 
            for word in sen:
                if type in word[3]:
                    ner_container.append(word[0])

    return ner_container

loc = getNER(corpus.data_pos_tagged, 'LOC')
per = getNER(corpus.data_pos_tagged, 'PER')
org = getNER(corpus.data_pos_tagged, 'ORG')

loc = set(loc)
per = set(per)
org = set(org)

total_ner = len(loc) + len(per) + len(org)
len_loc = len(loc) / total_ner 
len_per = len(per) / total_ner 
len_org = len(org) / total_ner

ner_df = pd.DataFrame({"ner": ['location', 'person', 'organization'], "count": [len_loc, len_per, len_org]})
print(ner_df)


fig7 = plotBar(ner_df, 'ner', 'count', 'Top 10 stop-words')
fig7.update_layout(
    xaxis_title = 'NER',
    yaxis_title = 'Xác suất xuất hiện (' + str(total_ner) + ')'
)

fig7.show()



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

    # graph 1_3
    dbc.Row([
        dcc.Graph(
            id = 'fig1_3',
            figure = fig1_3
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),


    # Graph 2
    
    dbc.Row([
        dcc.Graph(
            id='fig2',
            figure = fig2
        )]
    , style = {'marginLeft': -38, 'marginRight': -38, 'justifyContent': 'center'}),

    # Graph 2_1, 2_1_tu
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id = 'fig2_1',
                figure = fig2_1
            )
        ], width = 6), 
        dbc.Col(
            dcc.Graph(
                id = 'fig2_1_tu',
                figure = fig2_1_tu
            )
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),

    # Graph 2_2, 2_3
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig2_2',
                figure = fig2_2
            )
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig2_3',
                figure = fig2_3
            )
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),

    # Graph 3
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig3',
                figure = fig3
            )
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),

    # Graph 3_2
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig3_2',
                figure = fig3_2
            )
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),

    # Graph 3_3
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig3_3',
                figure = fig3_3
            )
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),

    # Graph 3_4
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig3_4',
                figure = fig3_4
            )
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),

    # Graph 4_1
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'fig4_1',
                figure = fig4_1
            )
        )
    ], style = {'marginLeft': -52, 'marginRight': -52, 'justifyContent': 'center'}),


]


app.layout = dbc.Container(
    statistics
)

# if __name__ == '__main__':
    # app.run_server(host= '127.0.0.1', port = 80, debug= True)