import sys
import pandas as pd
from sqlalchemy import *


def load_data(messages_filepath, categories_filepath):
    '''reads in two csv files based on their paths and then merges and stores files in a DataFrame.
     Then returns DataFrame.

        Parameters
        ----------
        messages_filepath: path of csv file messages
        categories_filepath: path of csv file categories 
    '''
    #read in csv file
    messages = pd.read_csv(messages_filepath)
    
    #read in csv file
    categories = pd.read_csv(categories_filepath)
    
    #merge DataFrame messages and categories into DataFrame df based on their id
    df = messages.merge(categories, how='inner', on='id')
    
    return df


def clean_data(df):
    '''reads in merged DataFrame and splits categories in separate columns. 
    Use column values as headers and drop duplicates.

        Parameters
        ----------
        df: merge DataFrame 
    '''
    
    #split column categories into separate columns delimited by semicolon
    categories = df['categories'].str.split(';', expand=True)
    #get the string values expect the last two characters of the first row and create DatFrame headers
    categories.columns = categories.iloc[0].str[:-2]   
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] =  pd.to_numeric(categories[column])
    
    # replace '2' values inlcuded in feature 'related' with '1'
    categories['related'] = categories['related'].replace(2, 1)
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''saves cleaned DataFrame to a SQL database. 

        Parameters
        ----------
        df: claned DataFrame
        database_filename: name of database
    '''
        
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterData', engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()