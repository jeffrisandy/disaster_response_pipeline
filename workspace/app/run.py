import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline,  FeatureUnion
from sklearn.multiclass import OneVsRestClassifier

import pickle
import re

# import sys
# sys.path.insert(0, '../models')

# from train_classifier import 

app = Flask(__name__)

def tokenize(text):
    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    #tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    #iterate for each tokens
    clean_tokens = []
    for tok in tokens:

        if tok not in stopwords.words('english'):
            # lemmatize, normalize case, and remove leading/trailing white space
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()

            clean_tokens.append(clean_tok)

    return clean_tokens



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
def load(filename):
    return pickle.load(open(filename, 'rb'))

#model = joblib.load("../models/classifier.pkl")
model = load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.drop(['id','message','original','genre'], axis=1).sum()
    category_names = list(category_counts.index)
    
    text = pd.Series(' '.join(df['message']).lower().split())
    most_words_counts = text[~text.isin(stopwords.words("english"))].value_counts()[:5]
    most_words_names = list(most_words_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
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
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=most_words_names,
                    y=most_words_counts
                )
            ],

            'layout': {
                'title': 'Most Frequent Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
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