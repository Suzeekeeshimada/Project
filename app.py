from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)

# Load model once at startup
model = load_model('model/cnn_model.h5')
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1. Define helper function at the top level
def save_to_history(filename, label, confidence):
    file_exists = os.path.isfile('history.csv')
    with open('history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, label, confidence])

# 2. Main Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Predict first, THEN save
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        save_to_history(file.filename, CLASSES[class_idx], round(confidence * 100, 2))
        
        return render_template('results.html', 
                               label=CLASSES[class_idx], 
                               confidence=round(confidence * 100, 2),
                               img_path=filepath)
    return render_template('index.html')

@app.route('/history')
def view_history():
    history = []
    if os.path.isfile('history.csv'):
        with open('history.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader, None) 
            history = list(reader)
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)