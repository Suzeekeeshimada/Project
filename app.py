from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv
import time
from datetime import datetime

app = Flask(__name__)

# Load model once at startup to keep the app fast
model = load_model('model/cnn_model.h5')
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Helper function to log validations to CSV
def save_to_history(filename, label, confidence):
    file_exists = os.path.isfile('history.csv')
    with open('history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, label, confidence])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return "No file uploaded"
        file = request.files['file']
        if file.filename == '': return "No file selected"
        
        # 1. Use a unique filename (timestamp + original name) to prevent file collisions
        unique_filename = str(int(time.time())) + "_" + file.filename
        filepath = os.path.join('static/uploads', unique_filename)
        file.save(filepath)
        
        # 2. Process image for the model
        img = image.load_img(filepath, target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Predict first
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # 4. Save to history CSV
        save_to_history(unique_filename, CLASSES[class_idx], round(confidence * 100, 2))
        
        # 5. Render results page
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
            next(reader, None) # Skip header
            history = list(reader)
    return render_template('history.html', history=history)

if __name__ == '__main__':
    # Debug must be False for Render/Gunicorn
    app.run(debug=False)