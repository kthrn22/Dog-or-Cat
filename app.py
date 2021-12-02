import os
from os import name
import flask
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = tf.keras.models.load_model('./model/cat_vs_dog_classifier.h5', 
                    custom_objects = {'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m})

@app.route("/", methods = ['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('main_page.html')

@app.route('/prediction<filename>')
def prediction(filename):
    img_dir = os.path.join('uploads', filename)
    image = load_img(img_dir, target_size = (150, 150))
    image_array = img_to_array(image) / 255
    image_array = image_array.reshape([1, 150, 150, 3])
    prediction = model.predict(image_array)
    if prediction > 0.5:
        result = "It's a dog 🐶"
    else:
        result = "It's a cat 🐱"
    return render_template('predict.html', result = result)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
