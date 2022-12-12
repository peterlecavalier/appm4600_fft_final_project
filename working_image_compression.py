import matplotlib.pyplot as plt
import numpy as np

# Load in the image you want to compress 
img = plt.imread('fig_3.jpeg')

# Make subplots and add original image
fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(img)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# This takes the average across the RGB scale converting it to a "grey scale" needed to perform the FFT
img = np.mean(img, -1)

# Add grey scale image to subplot
axs[0, 1].imshow(img)
axs[0, 1].set_title("'Grey Scale' Image")
axs[0, 1].axis('off')

# Perform a 2-D FFT on the image 
fft = np.fft.fft2(img)
fft = fft.astype(np.complex64)

# Print the size of the image after the FFT but before compression
print("The size before compression: {}".format(np.count_nonzero(fft)))
print()

# Sort the FFT by absolute value after re-sizing the matrix to make it a n x 1 matrix
sortedfft = np.sort(np.abs(fft), axis=None)[::-1]

# Loop over each amount of compression we want and make counters for subplots
amount_compression = [0.1, 0.05, 0.025, 0.0125]
subplotindex_x = 2
subplotindex_y = 0
for i in amount_compression:
    # Make a threshold matrix where any value you want will be larger than the value in this
    thresh = sortedfft[int(i*len(sortedfft))]

    # Create new logical matrix where values are 1 if over threshold and 0 if not 
    logical = np.abs(fft) > thresh

    # Dot product the FFT matrix and logical matrix to get compressed version of FFT matrix
    compressedfft = fft * logical
    print(fft.shape)
    print(logical.shape)

    # Print how large the actual compression is
    print("The size after compression of {}%: {}".format(i*100, np.count_nonzero(compressedfft)))
    print("This is {:.3f}% of the original size".format(np.count_nonzero(compressedfft)/np.count_nonzero(fft) * 100))
    print()
    # Invert back to image and show it
    new_img = np.fft.ifft2(compressedfft).real

    # Add to subplot and give labels
    axs[subplotindex_y, subplotindex_x].imshow(new_img)
    axs[subplotindex_y, subplotindex_x].set_title('Image compressed to {}%'.format(i*100))
    axs[subplotindex_y, subplotindex_x].axis('off')

    if subplotindex_y == 0:
        subplotindex_y = 1

    subplotindex_x += 1
    
    if subplotindex_x == 3:
        subplotindex_x = 0

plt.show()