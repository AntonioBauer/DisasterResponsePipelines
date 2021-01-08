# Disaster Response ETL and ML Pipelines
![](https://i.ibb.co/s5frggQ/Disaster-Header.png)

## Table of Contents
1. [Folder Structure](#FolderStructure)
2. [Installation](#Installation)
3. [Project Motivation](#Project)
4. [Business Understanding](#BusinessUnderstanding)
5. [Data Understanding](#DataUnderstanding)
6. [ETL Pipeline](#ETLPipeline)
7. [ML Pipeline](#MLPipeline)
8. [Evaluation](#Evaluation)
9. [Deployment](#Deployment)
10. [Licensing, Authors, Acknowledgements](#License)

## <a name="FolderStructure"></a>Folder Structure
-   app  
    | - template  
    | |- master.html # main page of web app  
    | |- go.html # classification result page of web app  
    |- run.py # Flask file that runs app
-   data  
    |- disaster_categories.csv # data to process  
    |- disaster_messages.csv # data to process  
    |- process_data.py  
    |- InsertDatabaseName.db # database to save clean data to
-   models  
    |- train_classifier.py  
    |- classifier.pkl # saved model
-   README.md

## <a name="Installation"></a>Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.
Run the following commands in the project's root directory to set up the database and model.

1. run ETL pipeline that cleans data and stores in database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. run ML pipeline that trains classifier and saves it as a pickle file: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run the flask web app: `python run.py`

4. Go to http://0.0.0.0:3001/

### <a name="Project"></a>Project Motivation
This project is part of my Data Scientist Nanodegree Program by Udacity. The goal of this submission is to create a machine learning pipeline to categorize real disaster event messages in order to triage them to the appropriate disaster relief agency.

### <a name="BusinessUnderstanding"></a>Business Understanding
The disaster response data is provided by [appen](https://appen.com "appen") and consists of two csv files. The 'messages.csv' file contains the english text of the message and the second 'categories.csv' file contains the corresponding categories for each message, represented by 0 (false) and 1 (true). 

### <a name="DataUnderstanding"></a>Data Understanding
The data consists of two csv files 'messages.csv' and 'categories.csv'.
The 'messages.csv' file has 26,248 rows and 4 columns. Only one column ('original') contains NULL values. The distribution of the data types is as follows:
Object data type: 3, int64 data type: 1.
The 'categories.csv' file has 26,386 rows and 36 columns. There aren't any NULL values in this file. The distribution of the data types is as follows:
int64 data type: 36.

### <a name="ETLPipeline"></a>ETL Pipeline
The files 'messages.csv' and 'categories.csv' were loaded into separate DataFrames and then merged into a single DataFrame by using an inner join. Then the column 'categories' is split into separate columns and headers were created accordingly. The next step was to convert category values to numbers (0 or 1) in order to use machine learning models later. There was a data quality issues with column 'related'. This column included some 2 values, so these values were replaced by 1 values. Last but not least 171 duplicates were dropped and the DataFrame was stored as a table in the DisasterResponse.db file.

### <a name="MLPipeline"></a>ML Pipeline
The table 'DisasterData' included in DisasterResponse.df was loaded into a DataFrame and then split into two DataFrames. Then a machine learning pipeline was created which includes the CountVectorizer which makes use of the tokenize function. Then the TfidfTransformer is used to then use a MultiOutputClassifier with a RandomForestClassifier estimator to label each message to one or many categories. In order to optimize the machine learning model, GridSearchCV was used to identify the best parameters. Then a classification report is printed to display the results of the test set. Finally a pickle module was used to store the model as a python object.

### <a name="Evaluation"></a>Evaluation
Overall the model is doing quite okay, but there are some major imbalances in this dataset. For example labels like shops and tools have a really small sample size and therefore the model is performing really bad in these cases.

                             precision    recall   f1-score   support

               electricity       0.82      0.07      0.12       136
                     tools       0.00      0.00      0.00        37
                 hospitals       0.00      0.00      0.00        71
                     shops       0.00      0.00      0.00        29
               aid_centers       0.00      0.00      0.00        73
      other_infrastructure       0.25      0.01      0.01       297
           weather_related       0.84      0.58      0.69      1842
                    floods       0.90      0.33      0.48       551

A bigger sample size, a generalization or reduction of features could also improve the performance. Also a more refined hyperparameter tuning might improve the output, but I was really struggeling with computing power, so I had to stick to a simple model.

### <a name="Deployment"></a>Deployment
The machine learning pipeline can be used in a user-friendly way through a flask web app which was given by Udacity. See section [Installation](#Installation).

## <a name="License"></a>Licensing, Authors, Acknowledgements
Must give credit to Figure Eight for the data. I used the help of the stackoverflow community, the pandas and scikit-learn documentation along with the Udacity Data Science courses Software Engineering and Data Engineering.