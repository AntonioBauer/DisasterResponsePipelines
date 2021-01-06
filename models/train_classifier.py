# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import pickle


def load_data(database_filepath):
    '''reads in a database and then assigns the applicable part of the dataframe to each variable.

        Parameters
        ----------
        database_filepath: path of database
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    
    # read Database table into DataFrame
    df = pd.read_sql_table('DisasterData', con=engine)
    # assign feature 'message' to variable X
    X = df['message']
    # assign all categoy features to variable Y
    Y = df.iloc[:, 4:]
    
    # get all column headers for variable Y
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """Tokenize and lemmatize each word in a given text"""

    # remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # Tokenize the string text and initiate the lemmatizer
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    
    for token in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    ''' creates pipeline to build model and finds the best parameters by using GridSearchCV.'''
    
    # build the pipeline for the text transformation and then for the estimator instance
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters are set to reduce the size of the pickle file, since my first files were larger than 1GB.
    parameters = {
        'clf__estimator__n_estimators': [5],
        'clf__estimator__min_samples_split': [2],
    }
    
    model = GridSearchCV(pipeline, param_grid = parameters, cv = 3, verbose = 2, n_jobs = 4)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate model against test set and print the classification_report.

        Parameters
        ----------
        model: trained machine learning model
        X_test: test features
        Y_test: test targets
        category_names: column names
    '''
    
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, Y_pred, target_names = category_names, digits = 2))


def save_model(model, model_filepath):
    '''save improved model to a pickle file.

        Parameters
        ----------
        model: trained machine learning model
        model_filepath: path were pickle file is stored
    '''
        
    with open(model_filepath, 'wb') as pkl_file:
        pickle.dump(model, pkl_file)
    pkl_file.close()


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