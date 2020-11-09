from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/analysis', methods=['POST'])
def sentiment_ana():
    result = []
    if request.method == 'POST':
        message = request.form['message']
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(message)
        result = vs["compound"]
    return render_template('result.html', prediction=result)


@app.route('/countInfoWord', methods=['POST'])
def Info_word():
    nltk.download('stopwords')
    nltk.download('punkt')
    wordbag = []
    len_text = 0
    if request.method == 'POST':
        text = request.form['message']
        text_tokens = word_tokenize(text)
        infoword = [word for word in text_tokens if not word in stopwords.words()]
        wordbag = infoword
        len_text = len(text_tokens)
    return render_template('wcResult.html', words=wordbag, length = len_text)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
