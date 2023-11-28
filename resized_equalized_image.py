import cv2
from matplotlib import pyplot as plt

image = cv2.imread('C:\pyprojects\image\photo.png', cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image)

resized_equalized_image = cv2.resize(equalized_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
histogram_resized = cv2.calcHist([resized_equalized_image], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 3))

plt.subplot(1,2,1)
plt.imshow(resized_equalized_image, cmap='gray')
plt.title('Resized Equalized Image')
plt.xticks(range(0,2001, 2000))  
plt.yticks(range(0, 2001, 2000)) 


plt.subplot(1,2,2)
plt.plot(histogram_resized)
plt.title('Resized Equalized Histogram')

plt.show()

  


