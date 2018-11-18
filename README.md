# Disaster Response Pipeline Project

### Project Overview
This project aim  is to classified text messages and identify those related to a disaster.
The model is tested and evaluated with dataset provide by "Figure Eight".

### Motivation

This repository contains the python source code to create a flask web app.
which coul dbe used by an emergency agency during a disaster event (e.g. an flood or storm) to receive information realtime from social media sourecs.
The application would receive text messages as input, and consider only the messages disaster related and it will classify the message into several categories. 
Each category represent information needed to investigate specic case of disasterand therefore the application can redirect the meesage to 
 to the appropriate aid agencies.

### Overview of the dataset

The data set used for the nltk training has been provided by "Figure Eight" it is composed of messages from 3 different sources ('direct', 'news', 'social').
These message are a mix of generic message and disaster related messages (total messages 25825, disaster related messages 19688). The disaster message are divided into 35 categories.

### Instructions:

These steps allow to rebuild and run the flask application.

1. unzip the file ./data/data.7z on the same folder (./data/), it will extract the two files needed to populate the database (disaster_categories.csv and disaster_message.csv).

2. Run the following commands in the project's root directory to set up the database.
   it will clean data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
3. Run the following commands in the project's root directory to set up the model
    It will run the ML pipeline that trains the classifier and saves teh model.
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Run the following command in the app's directory to run your web app.
    `python run.py`

5. Go to http://0.0.0.0:3001/

### Overview application user interface

The interface charts show the dataset used for the training.
The Overview is compose of the following diagrams:
1. Histogram of the number of messages for each genre;
![alt text](https://github.com/EnzoCalogero/NLTK_Disaster_Response_Pipeline/blob/master/message_per_source.png "Histogram of the number of messages for each genre")

2. Histogram of the number of messages for each category;
![alt text](https://github.com/EnzoCalogero/NLTK_Disaster_Response_Pipeline/blob/master/Message_per_category.png "Histogram of the number of messages for each category")

3. Histogram of the number of messages for each category grouped by genre;
![alt text](https://github.com/EnzoCalogero/NLTK_Disaster_Response_Pipeline/blob/master/message_category_grouped.png "Histogram of the number of messages for each category grouped by genre")

4. Histogram of the percentage of messages for each category grouped by genre;
![alt text](https://github.com/EnzoCalogero/NLTK_Disaster_Response_Pipeline/blob/master/percentage_category_grouped.png "Percentage of messages for each category grouped by genre")

### Files
**data/process_data.py**: The nltk ETL pipeline used for clean the message/data and saved them into the database.

**models/train_classifier.py**: The Machine Learning pipeline used create and export the model to a Python pickle.

**app/templates/master.html**: HTML templates for the flask web app.

**app/run.py**: Start the Python server for the web app and prepare visualizations.

**ETL Pipeline Preparation.ipynb**: jupiter notebook used to create process_data.py file.

**ML Pipeline Preparation.ipynb**: jupiter notebook used to create train_classifier.py file.

### Required libraries

. Flask 1.0.2

. plotly 3.4.1

. nltk 3.3.0

. numpy 1.15.2

. pandas 0.23.4

. scikit-learn 0.20.0
 
. sqlalchemy 1.2.14
