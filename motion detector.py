import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('motion_detection.avi', fourcc, 20.0, (640, 480))

# Initialize the first frame
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Calculate the difference between the current frame and the previous frame
    delta_frame = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the areas with motion
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False  # Flag for motion detection
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Ignore small movements
            continue

        # Motion detected
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # If motion detected, save the frame to the output video file
    if motion_detected:
        out.write(frame2)

    # Display the resulting frame
    cv2.imshow('Motion Detection', frame2)

    # Update the previous frame to the current frame
    gray1 = gray2

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
