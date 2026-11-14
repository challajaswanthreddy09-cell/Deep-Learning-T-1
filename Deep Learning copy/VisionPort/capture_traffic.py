
import cv2
import os

# Create folder to save captured images
os.makedirs("images", exist_ok=True)

# Open the default camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Unable to access camera.")
    print("Check your macOS camera permissions.")
    exit()

print("Camera started successfully!")
print("Press 'S' to capture the image.")
print("Press 'Q' to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Display live camera feed
    cv2.imshow("VisionPort - Traffic Image Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save image when S is pressed
    if key == ord('s'):
        image_path = "images/traffic_signal.jpg"

        cv2.imwrite(image_path, frame)

        print("Image captured successfully!")
        print("Saved at:", image_path)

    # Exit when Q is pressed
    elif key == ord('q'):
        break

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()

print("Camera closed successfully.")
