import cv2
from matplotlib import pyplot as plt

# upload image
image = cv2.imread('C:\pyprojects\image\photo.png', cv2.IMREAD_GRAYSCALE)

# calc histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 3))

# show orginal image
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.xticks(range(0, 1751, 1000))  # set x
plt.yticks(range(0, 1001, 1000))  # set y

# show histogram
plt.subplot(1,2,2)
plt.plot(histogram)
plt.title('Orginal Histogram')

plt.show()





  


