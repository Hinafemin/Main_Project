from flask import Flask, render_template, request
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained models
image_model = load_model('resnet_image_classifier_model.h5')
video_model = load_model('resnet_video_classifier_model.h5')

# Function to preprocess the image similar to your training data preprocessing
def preprocess_image(image):
    # Resize the image
    resized_image = cv2.resize(image, (64, 64))
    # Normalize pixel values
    processed_image = resized_image / 255.0
    return processed_image

# Function to preprocess the video frames similar to your training data preprocessing
def preprocess_video_frame(frame):
    # Resize the frame
    resized_frame = cv2.resize(frame, (64, 64))
    # Normalize pixel values
    processed_frame = resized_frame / 255.0
    return processed_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        # Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name
            file.save(temp_file_name)

        # Check if the uploaded file is an image or a video
        if file.filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read the uploaded image file
            image = cv2.imread(temp_file_name)
            processed_image = preprocess_image(image)
            processed_image = np.expand_dims(processed_image, axis=0)
            # Make prediction using the image model
            prediction = image_model.predict(processed_image)
            prediction_result = 'FAKE' if prediction.mean() > 0.5 else 'REAL'
            return render_template('result.html', image_prediction=prediction_result)
        elif file.filename.endswith(('.mp4', '.avi')):
            # Read the uploaded video file
            cap = cv2.VideoCapture(temp_file_name)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = preprocess_video_frame(frame)
                frames.append(processed_frame)
            cap.release()
            video = np.array(frames)
            # Make prediction using the video model
            prediction = video_model.predict(video.reshape(-1, 64, 64, 3))
            prediction_result = 'FAKE' if prediction.mean() > 0.5 else 'REAL'
            return render_template('result.html', video_prediction=prediction_result)
        else:
            return "Unsupported file format!"

if __name__ == '__main__':
    app.run(debug=True)
