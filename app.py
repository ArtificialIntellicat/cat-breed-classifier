from flask import Flask, request, render_template
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import csv

app = Flask(__name__)

model = load_model('models/cat-breed-classifier.h5')
label_encoder = joblib.load('src/label_encoder.joblib')

# Get all breed names for feedback selection
all_breeds = label_encoder.classes_.tolist()

# A route for the main page
@app.route('/')
def index():
    return render_template('index.html', cat_breeds=all_breeds)


# A route to handle image uploads and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        filestr = request.files['image'].read()
        # Convert string data to numpy array
        npimg = np.fromstring(filestr, np.uint8)
        # Convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocess the image
        img = cv2.resize(img, (224, 224))  # Example for VGG16
        img = img.astype('float32') / 255  # Scaling pixel values

        # Reshape and normalize the image
        img = np.expand_dims(img, axis=0)  # Reshape to (1, 224, 224, 3)

        # Make a prediction
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Convert class index to a readable label (you need to define this mapping)
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

        return render_template('index.html', prediction=predicted_label, cat_breeds=all_breeds)

    return 'No image found', 400

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    feedback = request.form['feedback']
    selected_breed = request.form['correctLabel']

    # Check if "Other" was selected
    if selected_breed == "other":
        selected_breed = request.form['otherBreedName']

    with open('feedback.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([feedback, selected_breed])

    return 'Thank you for your feedback!'


if __name__ == '__main__':
    app.run(debug=True)  # Starts the Flask web server
