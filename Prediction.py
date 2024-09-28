import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = 'deepfake_detection_model.keras'  # Change this path to the saved model file
model = load_model(model_path)


# Function to read and preprocess video frames with temporal pooling
def preprocess_video(video_path, num_frames_per_video=10, input_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(num_frames_per_video):
        frame_idx = int(i * total_frames / num_frames_per_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, input_size)
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    # Temporal pooling: Take average of frames to get a fixed number of frames
    frames = np.mean(frames, axis=0)
    return frames


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded video file from the user
        uploaded_file = request.files['video']
        if uploaded_file.filename != '':
            video_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(video_path)

            frames = preprocess_video(video_path)
            frames = np.expand_dims(frames, axis=0)  # Add a batch dimension for the model

            prediction = model.predict(frames)
            is_fake = "Fake" if prediction[0][0] >= 0.7 else "Real"
            confidence = prediction[0][0]

            os.remove(video_path)

            return render_template('index.html', result={'video_name': uploaded_file.filename, 'is_fake': is_fake,
                                                        'confidence': round(confidence*100, 2)})

    return render_template('index.html', result=None)


if __name__ == '__main__':
    app.run(debug=True)
