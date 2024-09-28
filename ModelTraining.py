import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the JSON file
json_file_path = 'train_videos/metadata.json'

with open(json_file_path, 'r') as f:
    data = json.load(f)

# Step 2: Prepare the data
data_folder = 'train_videos/'
video_filenames = os.listdir(data_folder)
num_frames_per_video = 10  # You can adjust this number depending on your hardware constraints

X = []  # List to store the frames
y = []  # List to store the labels (0: real, 1: fake)
count = 0

for video_name, info in data.items():
    count = count + 1
    print(str(count) + "." + video_name, end=" ")
    if count % 10 == 0:
        print()
    video_path = os.path.join(data_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames_per_video):
        frame_idx = int(i * total_frames / num_frames_per_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            X.append(cv2.resize(frame, (224, 224)))  # You can resize frames to the desired input size
            y.append(1 if info['label'] == 'FAKE' else 0)

    cap.release()

X = np.array(X) / 255.0  # Normalize pixel values to [0, 1]
y = np.array(y)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and train the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Step 5: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {round((test_acc * 100), 2)}')

# Step 6: Calculate and print the classification report and confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.7).astype(int)  # Convert probabilities to binary predictions (0 or 1)

acc_s = accuracy_score(y_test, y_pred) * 100
print("Accuracy Score: {} %".format(round(acc_s, 2)))

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Step 7: Save the trained model in the recommended format
# model.save('deepfake_detection_model.h5')

# Save the model in the TensorFlow format
# model.save('deepfake_detection_model', save_format='tf')

# or save it using the newer Keras format
model.save('deepfake_detection_model.keras')
