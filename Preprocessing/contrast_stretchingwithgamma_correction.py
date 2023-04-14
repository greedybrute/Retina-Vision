import cv2
import numpy as np

# Load the image
img = cv2.imread('path/to/image.jpg')

# Apply a bilateral filter to reduce noise
img_filtered = cv2.bilateralFilter(img, 9, 75, 75)

# Apply contrast stretching
v_min, v_max = np.percentile(img_filtered, (5, 95))
img_stretched = np.uint8(np.clip((img_filtered - v_min) * 255.0 / (v_max - v_min), 0, 255))

# Apply gamma correction
gamma = 1.5
img_gamma_corrected = np.uint8(np.clip(((img_stretched / 255.0) ** (1 / gamma)) * 255.0, 0, 255))

# Display the original image and processed images
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', img_filtered)
cv2.imshow('Contrast Stretched Image', img_stretched)
cv2.imshow('Gamma Corrected Image', img_gamma_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()
