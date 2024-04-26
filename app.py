import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)

model = load_model("model.h5")

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/avulusion.html')
def avulusion():
    return render_template("avulusion.html")
@app.route('/compression.html')
def compression():
    return render_template("compression.html")
@app.route('/greenstick.html')
def greenstick():
    return render_template("greenstick.html")

@app.route('/hairline.html')
def hairline():
    return render_template("hairline.html")
@app.route('/pathological.html')
def pathological():
    return render_template("pathological.html")
@app.route('/spiral.html')
def spiral():
    return render_template("spiral.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        test_image = image.load_img(filepath, target_size=(200, 200))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        img_data = preprocess_input(test_image)
        result = np.argmax(model.predict(img_data), axis=1)
        index = ['Avulsion fracture','Compression-Crush fracture',
                  'Greenstick fracture','Hairline Fracture',
                  'Pathological fracture','Spiral Fracture']
        text = str(index[result[0]])
        return render_template("output.html", prediction=text)

if __name__ == '__main__':
    app.run()