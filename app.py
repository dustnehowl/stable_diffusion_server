import keras_cv

from flask_cors import CORS, cross_origin
from flask import Flask, request, send_file
import tensorflow as tf
from tensorflow import keras
from keras_cv.models import StableDiffusion
import numpy as np
from PIL import Image
from translate import Translator

# import module.txt2img as t2i
import keras_cv
import matplotlib.pyplot as plt
import os, glob

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
translator = Translator(from_lang="ko", to_lang="en")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins":"*"}})


@app.route('/')
def home():
    print("helo")
   
    return 'This is Home!'

@cross_origin()

@app.route('/img2img', methods=["POST"])
def react_to_flask2():
    print("img2img style transfer")
    parsed_request = request.files.get('file')
    fileName = request.form.get("fileName")
    text = request.form.get("prompt")
    translation = translator.translate(text)
    prompt = translation

    filePath = "./static2/"
    filePath += fileName
    parsed_request.save(filePath)

    device = "cuda"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, use_auth_token=True
    ).to(device)

    init_image = Image.open("./static2/test.png") 
    init_image = init_image.resize((512, 512))

    images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images
    images[0].save("static/" + str(translation) + ".jpg")

    return {
        "success": True,
        "img_url" : "http://localhost:5000/static/" + str(translation) + ".jpg",
        "translation" : translation
    }

@app.route('/text2img', methods=["POST"])
def react_to_flask():
    print("텍스트를 받습니다.")
    text = request.form.to_dict()
    for key in text:
        text = key
        break
    print(text)
    translation = translator.translate(text)
    prompt = translation

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to(device)
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5).images[0]  
    
    rm_forder()
    image.save("static/" + str(translation) + ".jpg")

    return {
        "success": True,
        "img_url" : "http://localhost:5000/static/" + str(translation) + ".jpg",
        "translation" : translation
    }

def rm_forder():
    dir = 'static'
    filelist = glob.glob(os.path.join(dir, "*"))
    for f in filelist:
        os.remove(f)

if __name__ == '__main__':  
    app.debug = True
    app.run('127.0.0.1',port=5000,debug=True)