import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'deepfake_detection_model.keras'  # Change this path to the saved model file
model = load_model(model_path)
count = 0


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


# Folder containing the videos to be tested
test_folder = 'test_videos/'  # Change this to the path of the folder with test videos

# Iterate through videos in the folder and make predictions
for video_name in os.listdir(test_folder):
    video_path = os.path.join(test_folder, video_name)
    frames = preprocess_video(video_path)
    frames = np.expand_dims(frames, axis=0)  # Add a batch dimension for the model

    prediction = model.predict(frames)
    is_fake = "Fake" if prediction[0][0] >= 0.5 else "Real"
    count = count + 1
    print(f"{count}. Video: {video_name}, Prediction: {is_fake}, Confidence: {prediction[0][0]}")
