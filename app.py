from urllib import request
import pandas as pd
import pandas as pd
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
train.dropna(axis=0, how="any", thresh=None, subset=['text'], inplace=True)
length = []
[length.append(len(str(text))) for text in train['text']]
train['length'] = length
train = train.drop(train['text'][train['length'] < 50].index, axis = 0)
train_labels = train['label']
x_train, x_test, y_train, y_test = train_test_split(train['text'], train_labels, test_size=0.1, random_state=0)
tfidf = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
tfidf_train = tfidf.fit_transform(x_train) 
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

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