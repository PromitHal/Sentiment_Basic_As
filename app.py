  
from flask import Flask, render_template,request
#Importing necessary modules
from google_play_scraper import app,Sort,reviews_all
import pandas as pd
import numpy as np
from google_play_scraper import Sort, reviews
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input,LSTM,Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import re
import nltk 
import re
import pickle 
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score

ps=PorterStemmer()
app = Flask(__name__) #creating the Flask class object   


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        user_input = request.json['text']  # Get the input from the JSON data
        # Apply your function to the user input (replace this with your function)
        processed_output = process_input(user_input)
        return jsonify(result=processed_output)

# Define your processing function here
def process_input(input_text):
    def predict_review_sentiment(input_text,tk,model):
        pat = r'[^a-zA-z0-9]'
        review_cur=re.sub(pat,' ',input_text)
        review_cur=review_cur.lower()
        review_cur=review_cur.split()
        review_cur=[ps.stem(word) for word in review_cur if word not in stopwords.words('english')]
        review_cur=' '.join(review_cur)
        review=[]
        sentence_len=500
        review.append(review_cur)
        tokenized_review=pad_sequences(
        loaded_tokenizer.texts_to_sequences(review),
        maxlen=sentence_len,
        padding="pre"
        )
        pred=model(tokenized_review)
        temp=pred
        pred=tf.squeeze(pred)
        if pred>=0.5:
            pred=1
        else:
            pred=0
        return pred,temp

    pred,temp=predict_review_sentiment(input_text,loaded_tokenizer,model)
    temp=float(temp)*100
    temp=round(temp,ndigits=2)
    
    if pred==1: 
        print(temp)
        return ['Thanks for your positive response!',temp]
    else: 
        print(temp)
        return " Seems you are upset with the App!"

if __name__ =='__main__':  
    app.run(debug = True) 
