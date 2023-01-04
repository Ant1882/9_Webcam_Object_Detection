import cv2
import time

video = cv2.VideoCapture(2)
time.sleep(1)

first_frame = None

while True:
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("Video", frame)

    # Exit on key press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()
