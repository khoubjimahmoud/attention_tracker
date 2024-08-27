import cv2
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
import json
import os
import time
import pygetwindow as gw
import pyautogui
import shutil

model_path = 'model.h5'
clips_folder = 'clips'
images_folder = 'static/images'
data_folder = 'static/data'

model = load_model(model_path)

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

def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))
    return faces

def preprocess_frame(frame, img_height=128, img_width=128):
    frame = cv2.resize(frame, (img_height, img_width))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def classify_frame(model, face_img):
    processed_frame = preprocess_frame(face_img)
    predictions = model.predict(processed_frame)
    return predictions[0][0] < 0.5 

def get_zoom_window():
    """Function to get the Zoom window object."""
    try:
        zoom_windows = [win for win in gw.getWindowsWithTitle('Zoom Meeting') if not win.isMinimized]
        if zoom_windows:
            return zoom_windows[0]
    except Exception as e:
        print(f"Error finding Zoom window: {e}")
    return None

def clean_previous_data():
    """Deletes old clips, images, and data files before starting a new session."""
    for folder in [clips_folder, images_folder, data_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

def record_clip(filename, duration=120, fps=20):
    """Records a video clip from the Zoom window."""
    zoom_window = get_zoom_window()
    if zoom_window is None:
        print("Zoom window not found. Exiting recording...")
        return False
    
    x, y, width, height = zoom_window.left, zoom_window.top, zoom_window.width, zoom_window.height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    start_time = time.time()
    while (time.time() - start_time) < duration:
        zoom_window = get_zoom_window()
        if zoom_window is None:
            print("Zoom window closed during recording. Exiting...")
            out.release()
            return False
        
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        out.write(frame)
        time.sleep(1 / fps)

    out.release()
    return True

def process_clips_and_calculate_attention():
    attention_counts = {}
    total_counts = {}
    user_frames = {}
    tracker = CentroidTracker()

    for clip_file in os.listdir(clips_folder):
        if not clip_file.endswith('.mp4'):
            continue

        cap = cv2.VideoCapture(os.path.join(clips_folder, clip_file))

        while cap.isOpened():
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

                        if objectID not in user_frames:
                            user_frames[objectID] = face_img

        cap.release()

    attention_percentages = {f"User {objectID}": (attention_counts[objectID] / total_counts[objectID]) * 100
                             for objectID in attention_counts if total_counts[objectID] > 0}

    with open(os.path.join(data_folder, 'attention_data.json'), 'w') as f:
        json.dump(attention_percentages, f)

    for objectID, frame in user_frames.items():
        user_image_path = os.path.join(images_folder, f'User_{objectID}.jpg')
        cv2.imwrite(user_image_path, frame)

    return attention_percentages

def main():
    clean_previous_data()  

    while True:
        clip_filename = os.path.join(clips_folder, f"clip_{int(time.time())}.mp4")
        print(f"Recording clip: {clip_filename}")
        if not record_clip(clip_filename):
            break

        print("Pausing before the next recording...")
        time.sleep(120)

        print("Processing clips and calculating attention percentages...")
        attention_percentages = process_clips_and_calculate_attention()
        print("Attention percentages:", attention_percentages)

if __name__ == "__main__":
    main()
