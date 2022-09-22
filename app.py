import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash,session
import pickle
from werkzeug.utils import secure_filename
import os
from emotion_classify import emotion_model
emo = emotion_model()

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.secret_key = "super secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

pred_obj = None


def init_model():
   global pred_obj
   #model loaded from the app directory once at the beginning of the app
   pred_obj = emotion_model(path='result/mpl_classifier.model')
   pred_obj.load_model()
   print("model loaded")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def Classify():
   global pred_obj
   if request.method == 'POST':
      f = request.files['audio']
      name = f.filename
      f.save((f.filename))
      #pred_result = pred_obj((f.filename))
      pred_result=emo.predict(f.filename) 
      return render_template("index.html", n= "The Emotion Recognized in Speech is {}".format(pred_result))
        

if __name__ == "__main__": 
    app.run(debug=True)
    