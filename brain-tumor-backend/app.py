from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the trained model
model = load_model('xception.h5')  # Replace with your actual model path

# Define your classes (make sure these match your training classes)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Example classes, update as necessary

# Preprocessing function for image input
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((299, 299))  # Resize to the model's input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess as Xception expects
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Save the uploaded file temporarily
    temp_path = os.path.join('uploads', file.filename)
    file.save(temp_path)
    
    # Preprocess the image and make prediction
    img_array = preprocess_image(temp_path)
    prediction = model.predict(img_array)
    
    # Get class with maximum probability
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(round(np.max(prediction) * 100, 2))  # Convert to native Python float
    
    # Remove temporary image
    os.remove(temp_path)
    
    # Return the prediction and confidence
    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence
    })

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Run Flask app
    app.run(debug=True)
