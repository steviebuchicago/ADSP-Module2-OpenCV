from djitellopy import Tello
import cv2
import time

# Initialize the Tello drone
tello = Tello('YOUR_TELLO_IP')
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Takeoff
tello.takeoff()
tello.move_up(30)  # Adjust as necessary

try:
    while True:
        # Get the image from the drone's camera
        frame = frame_read.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # For simplicity, take the first detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Face center
            face_center_x = x + w / 2
            face_center_y = y + h / 2
            
            # Move the drone based on the position of the face
            if face_center_x < frame.shape[1] / 3:
                tello.move_left(20)
            elif face_center_x > frame.shape[1] * 2 / 3:
                tello.move_right(20)
            
            if face_center_y < frame.shape[0] / 3:
                tello.move_up(20)
            elif face_center_y > frame.shape[0] * 2 / 3:
                tello.move_down(20)
            
            break  # If you want to track multiple faces, remove this line

        # Display the resulting frame
        cv2.imshow('Tello tracking...', frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture and land the drone
    tello.land()
    cv2.destroyAllWindows()
