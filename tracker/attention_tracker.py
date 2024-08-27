import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

uploaded = files.upload()
file_path = next(iter(uploaded.keys()))
annotations = pd.read_csv(file_path)
annotations.head()

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from google.colab import drive
drive.mount('/content/drive')

img_height, img_width = 128, 128
image_dir = '/content/drive/MyDrive/My Drive/samples/'  # Path to the extracted images

# Create lists to store the images and labels
images = []
labels = []

# Load your annotations DataFrame (replace this with your actual DataFrame loading code)
annotations = pd.read_csv('/content/_annotations.csv')

# Define the binary label conversion function
def convert_label(label):
    return 0 if label == "Normal" else 1

# Load and preprocess the images
for idx, row in annotations.iterrows():
    img_path = os.path.join(image_dir, row['filename'])
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(convert_label(row['class']))
    else:
        print(f"Image {img_path} not found.")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Check the number of images and labels
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")

# Ensure there are enough samples
if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images or labels found. Please check the annotations and image paths.")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_val = X_val / 255.0

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save('/content/model.h5')
model.save('/content/drive/MyDrive/My Drive/model.h5')


from google.colab import files
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

uploaded = files.upload()
sample_image_path = next(iter(uploaded.keys()))

model = load_model('/content/model.h5')

# Step 3: Preprocess the Sample Image
img_height, img_width = 128, 128
sample_img = load_img(sample_image_path, target_size=(img_height, img_width))
sample_img_array = img_to_array(sample_img)
sample_img_array = sample_img_array / 255.0  # Normalize pixel values
sample_img_array = np.expand_dims(sample_img_array, axis=0)  # Add batch dimension

# Step 4: Make Predictions
predictions = model.predict(sample_img_array)
prediction_class = 'Normal' if predictions[0][0] < 0.5 else 'Other'

print(f'Prediction: {prediction_class}')

from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# Define a function to extract frames from the video
def extract_frames(video_path, interval=30):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return frames

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        print("Error: Frame rate is zero.")
        cap.release()
        return frames

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, frame_count, int(frame_rate * interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()
    return frames

# Define a function to preprocess the frame
def preprocess_frame(frame, img_height=128, img_width=128):
    frame = cv2.resize(frame, (img_height, img_width))
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Define a function to classify frames and calculate the percentage of "Normal" frames
def classify_frames(frames, model):
    normal_count = 0
    for frame in frames:
        processed_frame = preprocess_frame(frame)
        predictions = model.predict(processed_frame)
        if predictions[0][0] < 0.5:
            normal_count += 1

    if len(frames) == 0:
        return 0

    normal_percentage = (normal_count / len(frames)) * 100
    return normal_percentage

# Load the pre-trained model
model_path = '/content/drive/My Drive/My Drive/model.h5'  # Adjust the path to the model file as needed
model = load_model(model_path)

# Path to the video file on Google Drive
video_path = '/content/drive/My Drive/My Drive/sample_vid.mp4'

# Extract frames from the video
frames = extract_frames(video_path)

# Classify the frames and calculate the percentage of "Normal" frames
if frames:
    normal_percentage = classify_frames(frames, model)
    print(f"Percentage of 'Normal' frames: {normal_percentage}%")
else:
    print("No frames extracted from the video.")


import cv2

def extract_frames(video_path, interval=30):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return frames

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0 or frame_rate is None:
        print("Error: Frame rate is zero or could not be determined.")
        cap.release()
        return frames

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, frame_count, int(frame_rate * interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


from keras.models import load_model

# Load the pre-trained model
model_path = '/content/model.h5'  # Adjust the path to the model file as needed
model = load_model(model_path)

def preprocess_frame(frame, img_height=128, img_width=128):
    frame = cv2.resize(frame, (img_height, img_width))
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def classify_frame(model, frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    return predictions[0][0] < 0.5  # True if classified as "Normal"


import cv2
import numpy as np
from keras.models import load_model
from google.colab import drive
from scipy.spatial import distance as dist
from google.colab.patches import cv2_imshow
import time

# Mount Google Drive
drive.mount('/content/drive')

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Pre-trained model path
model_path = '/content/model.h5'  # Adjust the path to the model file as needed
model = load_model(model_path)

# Define the CentroidTracker class
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Define the detect_faces function
def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Define the preprocess_frame function
def preprocess_frame(frame, img_height=128, img_width=128):
    frame = cv2.resize(frame, (img_height, img_width))
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Define the classify_frame function
def classify_frame(model, face_img):
    processed_frame = preprocess_frame(face_img)
    predictions = model.predict(processed_frame)
    return predictions[0][0] < 0.5

# Define a function to calculate attention percentage for each user in 2-minute clips
def calculate_attention_percentage(video_path, clip_length=120, pause_length=120, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = int(frame_rate * clip_length)
    pause_frames = int(frame_rate * pause_length)
    frame_step = int(frame_rate * frame_interval)
    tracker = CentroidTracker()
    attention_counts = {}
    total_counts = {}

    frame_number = 0
    while frame_number < total_frames:
        if frame_number % (clip_frames + pause_frames) < clip_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame)
            rects = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
            objects = tracker.update(rects)

            for (objectID, centroid) in objects.items():
                if objectID not in attention_counts:
                    attention_counts[objectID] = 0
                    total_counts[objectID] = 0

                for (x, y, w, h) in faces:
                    if abs(centroid[0] - (x + w // 2)) < w // 2 and abs(centroid[1] - (y + h // 2)) < h // 2:
                        face_img = frame[y:y+h, x:x+w]
                        if classify_frame(model, face_img):
                            attention_counts[objectID] += 1
                        total_counts[objectID] += 1

            frame_number += frame_step
        else:
            frame_number += pause_frames

    cap.release()

    attention_percentages = {objectID: (attention_counts[objectID] / total_counts[objectID]) * 100
                             for objectID in attention_counts}

    return attention_percentages

# Path to the video file (needs to be adjusted)
video_path = '/content/drive/MyDrive/My Drive/sample_vid.mp4'

# Calculate attention percentages
start_time = time.time()
attention_percentages = calculate_attention_percentage(video_path)
end_time = time.time()

# Print the attention percentages
print("Attention Percentages for each user:")
for user_id, percentage in attention_percentages.items():
    print(f"User {user_id}: {percentage:.2f}%")

print(f"Total Processing Time: {end_time - start_time} seconds")


import json

# Your existing code to calculate attention percentages
attention_percentages = calculate_attention_percentage(video_path)

# Convert the attention percentages to JSON format
attention_data_json = json.dumps(attention_percentages)

print("Attention Percentages for each user:")
for user_id, percentage in attention_percentages.items():
    print(f"User {user_id}: {percentage:.2f}%")

print(f"Total Processing Time: {end_time - start_time} seconds")


print(attention_percentages)


import cv2
import numpy as np
from keras.models import load_model
from google.colab import drive
from scipy.spatial import distance as dist
from google.colab.patches import cv2_imshow
import time

# Mount Google Drive
drive.mount('/content/drive')

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Pre-trained model path
model_path = '/content/drive/MyDrive/My Drive/model.h5'  # Adjust the path to the model file as needed
model = load_model(model_path)

# Define the CentroidTracker class
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Define the detect_faces function
def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Define the preprocess_frame function
def preprocess_frame(frame, img_height=128, img_width=128):
    frame = cv2.resize(frame, (img_height, img_width))
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Define the classify_frame function
def classify_frame(model, face_img):
    processed_frame = preprocess_frame(face_img)
    predictions = model.predict(processed_frame)
    return predictions[0][0] < 0.5

# Define a function to calculate attention percentage for each user in 2-minute clips
def calculate_attention_percentage(video_path, clip_length=120, pause_length=120, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = int(frame_rate * clip_length)
    pause_frames = int(frame_rate * pause_length)
    frame_step = int(frame_rate * frame_interval)
    tracker = CentroidTracker()
    attention_counts = {}
    total_counts = {}

    frame_number = 0
    while frame_number < total_frames:
        if frame_number % (clip_frames + pause_frames) < clip_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame)
            rects = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
            objects = tracker.update(rects)

            for (objectID, centroid) in objects.items():
                if objectID not in attention_counts:
                    attention_counts[objectID] = 0
                    total_counts[objectID] = 0

                for (x, y, w, h) in faces:
                    if abs(centroid[0] - (x + w // 2)) < w // 2 and abs(centroid[1] - (y + h // 2)) < h // 2:
                        face_img = frame[y:y+h, x:x+w]
                        if classify_frame(model, face_img):
                            attention_counts[objectID] += 1
                        total_counts[objectID] += 1

            frame_number += frame_step
        else:
            frame_number += pause_frames

    cap.release()

    attention_percentages = {objectID: (attention_counts[objectID] / total_counts[objectID]) * 100
                             for objectID in attention_counts}

    return attention_percentages

# Path to the video file (needs to be adjusted)
video_path = '/content/drive/MyDrive/My Drive/sample_vid.mp4'

# Calculate attention percentages
start_time = time.time()
attention_percentages = calculate_attention_percentage(video_path)
end_time = time.time()

# Print the attention percentages
print("Attention Percentages for each user:")
for user_id, percentage in attention_percentages.items():
    print(f"User {user_id}: {percentage:.2f}%")

print(f"Total Processing Time: {end_time - start_time} seconds")

