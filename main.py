import cv2
import time
import glob
import os
from emailing import send_email
from threading import Thread


video = cv2.VideoCapture(2)
time.sleep(1)

first_frame = None
status_list = []
count = 1


def clean_folder():
    print("clean folder function started")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("clean folder function ended")

while True:
    status = 0
    check, frame = video.read()
   
    # Grey the image and blur to reduce noise
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Capture the first frame
    if first_frame is None:
        first_frame = gray_frame_gau

    # Get the difference for current frame
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    # Apply binary thresholding, remove noise
    thresh_frame = cv2.threshold(delta_frame, 45, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)
    cv2.imshow("My video", dil_frame)
    
    # Detect contours of object
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a green rectangle for each object detected
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        if rectangle.any():
            status = 1
            # Store images only when object detected
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1
            # Only get the middle image to email
            all_images = glob.glob("images/*.png")
            index = int(len(all_images) / 2)
            image_with_object = all_images[index]
    
    # Detect when the object has exited the frame
    status_list.append(status)
    status_list = status_list[-2:]
    
    # Email only when the object has exited the frame
    if status_list[0] == 1 and status_list[1] == 0:
        email_thread = Thread(target=send_email, args=(image_with_object, ))
        email_thread.daemon = True
        clean_thread = Thread(target=clean_folder)
        clean_thread.daemon = True

        email_thread.start()
        

    cv2.imshow("Video", frame)

    # Exit on key press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()
clean_thread.start()