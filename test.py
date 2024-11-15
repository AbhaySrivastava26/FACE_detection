from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time 
from datetime import datetime
from win32com.client import Dispatch
def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)
def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    """Resize image maintaining aspect ratio."""
    h, w = image.shape[:2]
    if target_width is None and target_height is None:
        return image
    if target_width is None:
        aspect_ratio = target_height / h
        target_width = int(w * aspect_ratio)
    else:
        aspect_ratio = target_width / w
        target_height = int(h * aspect_ratio)
    return cv2.resize(image, (int(target_width), int(target_height)))

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# Load the trained model data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Initialize and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load and prepare background image
imageBackground = cv2.imread("imageBackground.png")
if imageBackground is None:
    raise ValueError("Background image not found!")

# Set desired dimensions for the final display
DISPLAY_WIDTH = 800  # Adjust as needed
DISPLAY_HEIGHT = 880  # Adjust as needed

# Resize background to fill the display while maintaining aspect ratio
imageBackground = resize_with_aspect_ratio(imageBackground, 
                                         target_width=DISPLAY_WIDTH, 
                                         target_height=DISPLAY_HEIGHT)
COl_Names=['NAME','TIME']
while True:
    ret, frame = video.read()
    if not ret:
        break
        

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist=os.path.isfile("Attendance/Attendance_"+date+".csv")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 225), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance=[str(output[0]),str(timestamp)]
    # Prepare the background for display
    display_img = imageBackground.copy()
    
    # Calculate frame size and position (ensure integers)
    frame_height = int(display_img.shape[0] // 1.5)  # Adjust divisor to change frame size
    aspect_ratio = frame.shape[1] / frame.shape[0]
    frame_width = int(frame_height * aspect_ratio)
    
    # Resize frame while maintaining aspect ratio
    frame_resized = cv2.resize(frame, (frame_width, int(frame_height)))
    
    # Calculate position to place frame on right side (ensure integers)
    x_offset = int(display_img.shape[1] - frame_width - 20)  # 20 pixels padding from right
    y_offset = int((display_img.shape[0] - frame_height) // 2)  # Center vertically
    
    try:
        
        roi = display_img[y_offset:y_offset+frame_resized.shape[0], 
                         x_offset:x_offset+frame_resized.shape[1]]
        
        # Check if ROI and frame shapes match before overlaying
        if roi.shape == frame_resized.shape:
            display_img[y_offset:y_offset+frame_resized.shape[0], 
                       x_offset:x_offset+frame_resized.shape[1]] = frame_resized
    except ValueError as e:
        print(f"Error with image overlay: {e}")
        continue

  
    cv2.imshow("Face Detection", display_img)
    k=cv2.waitKey(1)
    if k==ord('o'):
        speak('Attendance Taken .')
        time.sleep(2)
        if exist:
          with open("Attendance/Attendance_"+date+".csv",'+a') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
          csvfile.close()
        else:
            with open("Attendance/Attendance_"+date+".csv",'+a') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COl_Names)
                writer.writerow(attendance)
            csvfile.close()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()