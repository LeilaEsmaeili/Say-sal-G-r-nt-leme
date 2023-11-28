import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('C:\pyprojects\image\photo.png', cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image)
resized_equalized_image = cv2.resize(equalized_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# blurred image
blurred_image = cv2.GaussianBlur(resized_equalized_image, (9, 9), 0)

# calc mask
mask = resized_equalized_image - blurred_image

k = 1.5

# calc highboost
highboost_image = cv2.addWeighted(resized_equalized_image, 1.0 + k, mask, k, 0)

# change image
f = np.fft.fft2(highboost_image)
fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.imshow(highboost_image, cmap='gray')
plt.title('Highboost Image')
plt.xticks(range(0,2001, 2000)) 
plt.yticks(range(0, 2001, 2000)) 

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Highboost Magnitude Spectrum')

plt.show()
