
import cv2
import time

print("Starting VisionPort camera test...")

camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

print("Camera status:", camera.isOpened())

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera opened! Waiting for frames...")

time.sleep(2)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Unable to read camera frame.")
        break

    cv2.imshow("VisionPort Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("Camera test completed.")
