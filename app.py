import keras_cv

from flask_cors import CORS, cross_origin
from flask import Flask, request, send_file
import tensorflow as tf
from tensorflow import keras
from keras_cv.models import StableDiffusion
import numpy as np
from PIL import Image
from translate import Translator

import keras_cv
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
translator = Translator(from_lang="ko", to_lang="en")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins":"*"}})


@app.route('/')
def home():
    print("helo")
   
    return 'This is Home!'

@cross_origin()

@app.route('/text2img', methods=["POST"])
def react_to_flask():
    print("텍스트를 받습니다.")
    text = request.form.to_dict()
    for key in text:
        text = key
        break
    print(text)
    translation = translator.translate(text)
    file_name = './test.jpg'
    return {
        "success": True,
        "img_url" : "http://localhost:5000/static/test.jpg",
        "translation" : translation
    }

if __name__ == '__main__':  
    app.debug = True
    app.run('0.0.0.0',port=5000,debug=True)