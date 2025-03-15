import cv2
import numpy as np

# Create a black image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw a white rectangle
cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), -1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find contours
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours Found: {len(contours)}")  # Should print "Contours Found: 1"

cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
