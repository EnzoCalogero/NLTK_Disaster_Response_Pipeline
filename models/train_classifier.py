# import libraries
import pickle
import re
import sys

import nltk
import pandas as pd
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

def load_data(database_filepath):
    """
     connect to the sqlite database and export the mesages dataframe
    --
    Inputs:
        database_filepath: sqlite database
    Outputs:
        X: Independent Variables
        Y: Dependent Variables
        category_names : List variables/ Features
    """
    table_name = 'Messages'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, con=engine)

    category_names = df.columns[4:]

    X = df['message']
    Y = df[category_names]

    return X, Y, category_names


def tokenize(text):
    """
        The input text is cleaned, normalizated and lemmatizated
        --
        Inputs:
            text: input text
        Outputs:
            clean_tokens: list of tokens
        """

    # Replace any URL with  urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # remove punctuation and Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    # Remove stop-words
    stop_words = stopwords.words("english")
    tokens = [token for token in tokens if token not in stop_words]

    # lemmitize
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Define the model pipeline

    Returns:
    cv: gridsearch cv model object. Gridsearchcv object with optimaize parameters.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC(multi_class="crammer_singer"), n_jobs=1))

    ])
    parameters = {
        'clf__estimator__C': [0.8, 1.0, 1.2, 1.4],
        'clf__estimator__max_iter': [500, 1000, 1200, 1500],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_weighted', verbose=1, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function print out the classification performance for the test subset

    :param model: model object
    :param X_test: Independent Variables for the test set
    :param Y_test: Dependent Variables for the test set
    :param category_names: list of the features related to the disaster categories
    :return: None
    """
    model_predict = model.predict(X_test)
    print(classification_report(Y_test, model_predict, target_names=category_names))


def save_model(model, model_filepath):
    """
    The function save the model parameter and weights to the given
    file path in the pincle file format.
    :param model: model object
    :param model_filepath: pickle file path
    :return: none
    """
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
        print('Please provide the filepath of the disaster messages database ' 
              'as the first argument and the filepath of the pickle file to ' 
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()