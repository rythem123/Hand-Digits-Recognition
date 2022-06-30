import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, flash, render_template, request, send_from_directory, abort
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import models,layers
from PIL import Image
from logging import FileHandler,WARNING

Batch_Size=32
channels=3
EPOCHS=30
data_set=tf.keras.preprocessing.image_dataset_from_directory("E:\\zcomplete web devlopment\\hand_digits\\captured_images",seed=123,shuffle=True,image_size=(340,380)
                                                             ,batch_size=Batch_Size)
class_name=data_set.class_names
model = load_model("E:\\zcomplete web devlopment\\hand_digits\\model\\VSK.h5")
def predict(model,img):
    img_array=tf.keras.preprocessing.image.img_to_array(img)
    img_array=tf.expand_dims(img_array,0)
    predictions=model.predict(img_array)
    predicted_class=class_name[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence

app = Flask(__name__)
# app = Flask(__name__, template_folder = 'template')
app.config["MNIST_BAR"] = "generated_image"
app.config["IMAGES"] = "upload"

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/mnist')
def mnist_home():
    return render_template('mnist.html')


@app.route('/mnistprediction/', methods=['GET', 'POST'])
def mnist_prediction():
    if request.method == "POST":
        if not request.files['file'].filename:
            flash("No File Found")
        else:
            image =  request.files['file']
            image.save("uploads/"+image.filename)
            # image1=Image.open("E:\\zcomplete web devlopment\\hand_digits\\uploads\\28.png")
            # tf_image = image.load_img("uploads/"+image.filename, grayscale=True, color_mode='rgb', target_size=(340,380), interpolation='nearest')
            tf_image = Image.open("uploads/"+image.filename)
            img_batch=np.expand_dims(tf_image,0)
            predictions=model.predict(img_batch)
            pc=class_name[np.argmax(predictions[0])]
            return pc
            # return "yes"
@app.route("/get-mnist-image/<image_name>")
def get_mnist_image(image_name):
    try:
        return send_from_directory(app.config["MNIST_BAR"], filename=image_name)
    except FileNotFoundError:
        abort(404)

if __name__=="__main__":
    app.run(debug=True)
 