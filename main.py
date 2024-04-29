import random
import json
import pickle
import bcrypt
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

from flask import Flask, redirect, render_template, request, session, flash, jsonify

app = Flask(__name__)
app.secret_key = 'fa1d9f0aeb0148117c0b5622ba892f8d'



@app.route("/")
def home():
    return render_template('index.html')


@app.route('/suggestion', methods=['POST'])
def suggestion():
    email = request.form.get('uemail')
    suggestion_message = request.form.get('message')

    suggestion_data = {'email': email, 'message': suggestion_message}
    print(suggestion_data)
    flash('Your suggestion has been successfully sent!', 'success')
    return redirect('/')






lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    print("Predicted intents:", return_list)
    return return_list


def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



@app.route("/get")
def get_bot_response():
    message = request.args.get('msg') 
    print("Input message:", message)
    bow = bag_of_words(message)
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot response:", res)
    return str(res)
    

if __name__ == "__main__":
    app.run(debug=True)