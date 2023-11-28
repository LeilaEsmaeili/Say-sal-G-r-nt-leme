import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('C:\pyprojects\image\photo.png', cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image)
resized_equalized_image = cv2.resize(equalized_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)


# blurred
blurred_image = cv2.GaussianBlur(resized_equalized_image, (9, 9), 0)

# calc mask
mask = resized_equalized_image - blurred_image

k = 1.5

# calc highboost image
highboost_image = cv2.addWeighted(resized_equalized_image, 1.0 + k, mask, k, 0)

# calc highboost histogram
histogram_highboost = cv2.calcHist([highboost_image], [0], None, [256], [0, 256])

# show image and histogram
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.imshow(highboost_image, cmap='gray')
plt.title('Highboost Image')
plt.xticks(range(0,2001, 2000)) 
plt.yticks(range(0, 2001, 2000)) 

plt.subplot(1, 2, 2)
plt.plot(histogram_highboost)
plt.title('Histogram of Highboost Image')

plt.show()
