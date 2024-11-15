# import cv2
# face_cap=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# faces_data=[]
# i=0
# video_capture=cv2.VideoCapture(0)
# while True:
#     abhay,video_data=video_capture.read()
#     col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
#     #detect multiscale detects the face
#     faces=face_cap.detectMultiScale(col,1.3,5)
#     #     col,
#     #     scaleFactor=1.1,
#     #     minNeighbors=5,
#     #     minSize=(30,30),
#     #     flags=cv2.CASCADE_SCALE_IMAGE
#     # )
#     #width and height detects the box and x and y are the coordinates
#     for(x,y,w,h) in faces:
#         crop_image= video_data[y:y+h,x:x+w,:]
#         resized_img=cv2.resize(crop_image,(50,50))
#         if len(faces_data)<=100 and i%10==0:
#          faces_data.append(resized_img) 
#          i=i+1
#          cv2.putText(video_data,str(len(faces_data)) ,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,225),1)
          
#          cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
#          cv2.imshow("ABHAY KA Camera",video_data)
#     if cv2.waitKey(1)==ord("z") or len(faces_data)==100:
#      break
# video_capture.release()
# cv2.destroyAllWindows()
import cv2
import pickle 
import numpy as np
import os #check the inside data folder

# Load the Haar cascade for face detection
face_cap = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cap2=cv2.CascadeClassifier("haarcascade_eye.xml")
faces_data = []
i = 0
name=input("enter the username from the user")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, video_data = video_capture.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cap.detectMultiScale(gray, 1.3, 5)
    
    # Loop through detected faces
    for (x, y, w, h) in faces:
        crop_image = video_data[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_image, (50, 50))
        
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        
        # Draw rectangle around the face
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if len(faces_data) < 100 and i % 10 == 0:
        cv2.putText(video_data, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 225), 1)
    
    # Display the resulting frame
    cv2.imshow("Face Capture", video_data)
    
    # Increment the frame counter
    i += 1
    
    # Break the loop if 'z' is pressed or 100 faces have been collected
    if cv2.waitKey(1) == ord("z") or len(faces_data) == 100:
        break

# Release the capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
#this changes converts the data into numpy as asarray which is useful for operation 
faces_data=np.asarray(faces_data)
#now it has 100 rows with each row containing all pixcle values 
faces_data=faces_data.reshape(100,-1)
if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl') as f:
        pickle.dump(names,f)
else:
    with open('data/names.pkl') as f:
     names=pickle.load(f)
     