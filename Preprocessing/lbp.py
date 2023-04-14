from skimage.feature import local_binary_pattern
import numpy as np
import cv2

img = cv2.imread('path/to/image.jpg', cv2.IMREAD_GRAYSCALE)

# Define LBP parameters
radius = 3
n_points = 8 * radius

# Calculate LBP image
lbp = local_binary_pattern(img, n_points, radius, method='uniform')

# Calculate histogram of LBP image
hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

# Normalize histogram
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

# Print histogram values
print(hist)
