from skimage.feature import greycomatrix, greycoprops
import numpy as np
from PIL import Image


img = Image.open('path/to/image.jpg').convert('L') # Convert to grayscale

# Convert the image to a numpy array
img_arr = np.array(img)

# Calculate the GLCM with a distance of 1 and an angle of 0 degrees
glcm = greycomatrix(img_arr, [1], [0], levels=256, symmetric=True, normed=True)

# Calculate contrast and correlation properties
contrast = greycoprops(glcm, 'contrast')
correlation = greycoprops(glcm, 'correlation')

print(f'Contrast: {contrast}')
print(f'Correlation: {correlation}')
