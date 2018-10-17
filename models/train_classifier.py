import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import re

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline,  FeatureUnion
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler

import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df['message'].values
    Y = df.drop(columns=['id', 'message', 'original','genre'])
    category_names = Y.columns.tolist()
    return X,Y, category_names


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


def build_model():

    msg_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2))),
        ('nlp_union', FeatureUnion([
            ('tfidf_pl', Pipeline([
                ('tfidf', TfidfTransformer()),
                ('dim_red', SelectKBest(chi2, 100))
            ])),
            ('svd_pl', Pipeline([
                ('tfidf_svd', TfidfTransformer()),
                ('truncated_svd', TruncatedSVD(500))
            ]))
        ]))
    ])



    pipeline = Pipeline([
        ('features', msg_pipeline),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression(random_state=42), n_jobs=-1))

    ])

    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    """ Print out classification report """
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print("\nClassification report:\n", class_report)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
