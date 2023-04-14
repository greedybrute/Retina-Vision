import cv2
from skimage.feature import hog
from cv2 import xfeatures2d

img = cv2.imread('path/to/image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = xfeatures2d.SIFT_create()

# Detect SIFT keypoints and extract descriptors
kp, des_sift = sift.detectAndCompute(gray, None)

# Initialize HOG descriptor
hog_desc = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, block_norm='L2-Hys')

# Concatenate SIFT and HOG descriptors
des_combined = np.concatenate((des_sift, hog_desc), axis=1)

# Display SIFT keypoints on image
img_sift = cv2.drawKeypoints(img, kp, None)

# Display HOG descriptor on image
img_hog = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)[1]
img_hog = cv2.cvtColor(img_hog, cv2.COLOR_GRAY2BGR)

# Display combined descriptors on image
img_combined = cv2.drawKeypoints(img, kp, None)
img_combined = cv2.cvtColor(img_combined, cv2.COLOR_GRAY2BGR)
img_combined[:, :, 2] = img_hog[:, :, 2]

# Show the images
cv2.imshow('SIFT Keypoints', img_sift)
cv2.imshow('HOG Descriptor', img_hog)
cv2.imshow('Combined Descriptors', img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
