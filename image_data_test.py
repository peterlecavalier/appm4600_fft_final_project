import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# testing loading in differnt pictures 
# img = plt.imread('testimage.png')
img = plt.imread('fig_2.jpeg')

# show image 
plt.imshow(img)
plt.show()

# convert image to np array and show shape
d = np.array(img)
print(d.shape)

