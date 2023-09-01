from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)


DATASET_PATH = './dataset'
class_labels = ['Cercospora Leaf Spot','Common Rust','Nothern Blight','Healthy']

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        img_file = request.files['imgFiles']
        img_path = './static/assets/upload-test.jpg' 
        img_file.save(img_path)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)
        #img = np.vstack([img])
        model_path = './model/model_efficientnet.h5'
        model = load_model(model_path)
        predict = model.predict(img)

        result = list(predict[0])
        max_value = max(result)
        index = result.index(max_value)
        predict_class = class_labels[index]

        return render_template("index.html", img_path=True,class_labels=class_labels, result=result, max_value=max_value, predict_class=predict_class)
    else:
        result = [0, 0, 0, 0]
        return render_template("index.html", class_labels=class_labels, result=result, img_path=False)


if __name__ == '__main__':
    app.run(debug=True)
