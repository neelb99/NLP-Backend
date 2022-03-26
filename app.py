from urllib import request
import numpy as np
import pandas as pd
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask import jsonify

app = Flask(__name__)
CORS(app)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test = test.set_index('id', drop = True)

# Dropping all rows where text column is NaN
train.dropna(axis=0, how="any", thresh=None, subset=['text'], inplace=True)
test = test.fillna(' ')

# Checking length of each article
length = []
[length.append(len(str(text))) for text in train['text']]
train['length'] = length

# Removing outliers, it will reduce overfitting
train = train.drop(train['text'][train['length'] < 50].index, axis = 0)

# Secluding labels in a new pandas dataframe for supervised learning
train_labels = train['label']
# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(train['text'], train_labels, test_size=0.1, random_state=0)

# Setting up Term Frequency - Inverse Document Frequency Vectorizer
tfidf = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
# Fit and transform training set and transform test set
tfidf_train = tfidf.fit_transform(x_train) 
tfidf_test = tfidf.transform(x_test)
tfidf_test_final = tfidf.transform(test['text'])

# Setting up Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter = 50)
# Fitting on the training set
pac.fit(tfidf_train, y_train)
# Predicting on the test set
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

@app.route('/predict', methods = ['POST'])
def predict():
    if len(request.json["msg"].split(" ")) <= 50 and not "https://" in request.json["msg"]:
        return jsonify("Please enter a url or more than 50 words")
    tfidf_test = tfidf.transform([request.json["msg"]]) 
    y_pred = pac.predict(tfidf_test)
    response = ""
    if y_pred[0] == 0:
        response = "Fake"
    else:
        response = "Not Fake"
    return jsonify(response)