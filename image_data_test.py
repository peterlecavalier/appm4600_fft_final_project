import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# testing loading in differnt pictures 
# img = plt.imread('testimage.png')
img = plt.imread('fig_3.jpeg')

# show image 
plt.imshow(img)
plt.show()

b = np.mean(img, -1)

plt.imshow(b)
plt.show()

# convert image to np array and show shape
d = np.array(b)
print(d.shape)

fft = np.fft.fft2(b)
#amount_compression = [0.1]
amount_compression = 0.0005
#mag = np.abs(b.reshape(-1))
sortedB = np.sort(np.abs(b.reshape(-1)))

threshold = sortedB[int(np.floor((1-amount_compression)*len(sortedB)))]

ind = np.abs(b) > threshold

b_low = b * ind

new_img = np.fft.ifft2(b_low).real
plt.imshow(new_img)
plt.show()