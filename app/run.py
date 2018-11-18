import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
table_name = 'Messages'
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table(table_name, engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # All Source Category
    category_counts = list(df[df.columns[4:]].sum())
    category_names = df.columns[4:]

    # Direct Source categories
    temp = df[df['genre'] == 'direct'].copy()
    category_direct_counts = list(temp[temp.columns[4:]].sum())

    category_total = sum(category_direct_counts)
    category_direct_percentage = list(temp[temp.columns[4:]].sum() / category_total)


    # News Source categories
    temp = df[df['genre'] == 'news'].copy()
    category_news_counts = list(temp[temp.columns[4:]].sum())

    category_total = sum(category_news_counts)
    category_news_percentage = list(temp[temp.columns[4:]].sum() / category_total)

    # Social Source categories
    temp = df[df['genre'] == 'social'].copy()
    category_social_counts = list(temp[temp.columns[4:]].sum())

    category_total = sum(category_social_counts)
    category_social_percentage = list(temp[temp.columns[4:]].sum() / category_total)
    # create visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Number Message for each Sources',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Source"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Number Message for each Category',
                'yaxis': {
                    'title': "Count",
                    'tickangle': -35
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -35
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_direct_counts,
                    name='Direct'
                ),
                Bar(
                    x=category_names,
                    y=category_news_counts,
                    name='News'
                    ),
                Bar(
                    x=category_names,
                    y=category_social_counts,
                    name='Social'
                    )
                ],

            'layout': {
                'title': 'Number of Messages for Each Category Grouped by Source',
                'yaxis': {
                    'title': "Count",
                    'tickangle': -35
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -35
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_direct_percentage,
                    name='Direct'
                ),
                Bar(
                    x=category_names,
                    y=category_news_percentage,
                    name='News'
                ),
                Bar(
                    x=category_names,
                    y=category_social_percentage,
                    name='Social'
                )
            ],
            'layout': {
                'title': 'Percentage of Messages for Each Category Grouped by Source',
                'yaxis': {
                    'title': "Relative percentage",
                    'tickangle': -35
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -35
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()