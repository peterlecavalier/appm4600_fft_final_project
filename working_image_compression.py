import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# testing loading in differnt pictures 
# img = plt.imread('testimage.png')
img = plt.imread('fig_3.jpeg')

# show image 
plt.imshow(img)
plt.show()

img = np.mean(img, -1)

# convert image to np array and show shape
d = np.array(img)
print(d.shape)

plt.imshow(img)
plt.show()

fft = np.fft.fft2(img)

print("The size before compression: {}".format(np.count_nonzero(fft)))

sortedfft = np.sort(np.abs(fft.reshape(-1)))

amount_compression = 0.1

threshold = sortedfft[int(np.floor((1-amount_compression)*len(sortedfft)))]

ind = np.abs(fft) > threshold

b_low = fft * ind

print("The size after compression: {}".format(np.count_nonzero(b_low)))

new_img = np.fft.ifft2(b_low).real

plt.imshow(new_img)
plt.show()

for i in [0.1, 0.05, 0.025, 0.0125]:
    threshold = sortedfft[int(np.floor((1-i)*len(sortedfft)))]

    ind = np.abs(fft) > threshold

    b_low = fft * ind

    new_img = np.fft.ifft2(b_low).real

    plt.imshow(new_img)
    plt.show()