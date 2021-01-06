# Disaster Response Pipelines
![](https://ibb.co/BgmHjnQ)

## Table of Contents
1. [Installation](#Installation)
2. [Project Motivation](#Project)
3. [File Descriptions](#File)
4. [Results](#Results)
5. [Licensing, Authors, and Acknowledgements](#License)

## <a name="Installation"></a>Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.
Run the following commands in the project's root directory to set up the database and model.

1. run ETL pipeline that cleans data and stores in database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. run ML pipeline that trains classifier and saves it as a pickle file: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run the flask web app: `python run.py`

4. Go to http://0.0.0.0:3001/

## <a name="Project"></a>Project Motivation
This project is part of my Data Scientist Nanodegree Program by Udacity. The goal of this submission is to create a machine learning pipeline to categorize real disaster event messages in order to triage them to the appropriate disaster relief agency. For 

### 1. Business Understanding
The disaster response data is provided by [appen](https://appen.com "appen") and consists of two csv files. The 'messages.csv' file contains the id, the text translated into english and the original text for each message. The second 'categories.csv' file contains the corresponding categories to each message, represented by 0 (false) and 1 (true). 

### 2. Data Understanding
The data consists of two csv files 'messages.csv' and 'categories.csv'.
The 'messages.csv' file has 26,248 rows and 4 columns. Only one column ('original') contains NULL values. The distribution of the data types is as follows:
Object data type: 3, int64 data type: 1.
The 'categories.csv' file has 26,386 rows and 36 columns. There aren't any NULL values in this file. The distribution of the data types is as follows:
int64 data type: 36.

### 3. ETL Pipeline Preparation
The files 'messages.csv' and 'categories.csv' were loaded into separate DataFrames and then merged into a single DataFrame by using an inner join. Then the column 'categories' is split into separate columns and headers were created accordingly. The next step was to convert category values to numbers (0 or 1) in order to use machine learning models later. There was a data quality issues with column 'related'. This column included some 2 values, so these values were replaced by 1 values. Last but not least 171 duplicates were dropped and the DataFrame was stored as a table in the DisasterResponse.db file.

### 4. Data Modeling
The table 'DisasterData' included in DisasterResponse.df was loaded into a DataFrame and then split into two DataFrames. Then a machine learning pipeline was created which includes the CountVectorizer which makes use of the tokenize function. Then the TfidfTransformer is used to then use a MultiOutputClassifier with a RandomForestClassifier estimator to label each message to one or many categories. In order to optimize the machine learning model, GridSearchCV was used to identify the best parameters.
Then the pickle module was used to store the model as a python object.

### 5. Evaluation
The variables used were sufficient to use descriptive statistics to gain interesting insights regarding the market value. For the k-nearest-neighbor (KNN) classifier, I used 3 features to predict the league of a player. The accuracy is around 40% which is not too high but acceptable for my first machine learning model.

### 6. Deployment
The machine learning pipeline can be used in a user-friendly way through a flask web app which was given by Udacity.

## <a name="License"></a>Licensing, Authors, Acknowledgements
Must give credit to Figure Eight for the data. I used the help of the stackoverflow community, the pandas and scikit-learn documentation along with the Udacity Data Science courses software engineering and data engineering. 