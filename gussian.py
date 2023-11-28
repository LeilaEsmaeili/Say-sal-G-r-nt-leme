import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('C:\pyprojects\image\photo.png', cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image)
resized_equalized_image = cv2.resize(equalized_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)


blurred_image = cv2.GaussianBlur(resized_equalized_image, (9, 9), 0)

mask = resized_equalized_image - blurred_image

k = 1.5

highboost_image = cv2.addWeighted(resized_equalized_image, 1.0 + k, mask, k, 0)
blurred_image = cv2.GaussianBlur(highboost_image, (9, 9), 0)

kernel_size = 2001
sigma = 1
gaussian_kernel = cv2.getGaussianKernel(kernel_size , sigma)
gaussian_mask = np.outer(gaussian_kernel, gaussian_kernel.T)

gaussian_filtered_image = cv2.GaussianBlur(highboost_image, (kernel_size, kernel_size), sigma)


plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.imshow(gaussian_mask, cmap='gray')
plt.title('Gaussian Mask')

plt.subplot(1, 2, 2)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title('Gaussian Filtered Image')
plt.xticks(range(0,2001, 2000))  
plt.yticks(range(0, 2001, 2000)) 

plt.show()