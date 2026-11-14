
import cv2
import os

# Path to the captured image
image_path = "images/camera_index1.jpg"

# Check whether the image exists
if not os.path.exists(image_path):
    print("Error: Image file not found!")
    exit()

# Read the image using OpenCV
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to read the image!")
    exit()

# Display image information
height, width, channels = image.shape

print("Image loaded successfully!")
print("Image width:", width)
print("Image height:", height)
print("Number of channels:", channels)

# Display the image
cv2.imshow("VisionPort - Traffic Image", image)

print("Press Q to close the image window.")

cv2.waitKey(0)
cv2.destroyAllWindows()
