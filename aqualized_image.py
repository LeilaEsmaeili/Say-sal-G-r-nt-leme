import cv2
from matplotlib import pyplot as plt

image = cv2.imread('C:\pyprojects\image\photo.png', cv2.IMREAD_GRAYSCALE)

equalized_image = cv2.equalizeHist(image)

# calc histogram
histogram = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 3))

#  show equalized image
plt.subplot(1,2,1)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.xticks(range(0, 1751, 1000))  # set x
plt.yticks(range(0, 1001, 1000))  # set y

# show histogram 
plt.subplot(1,2,2)
plt.plot(histogram)
plt.title('Equalized Histogram')


plt.show()







  


