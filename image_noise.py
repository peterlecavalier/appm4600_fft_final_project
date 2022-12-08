import numpy as np
import matplotlib.pyplot as plt

# Read in our image and plot it
img = plt.imread('aerial.jpg')
plt.imshow(img)
plt.title('Original image with periodic noise', fontdict={'fontsize':18})
plt.tight_layout()
plt.savefig('noise_img.png')
plt.show()

# run the fft on the image
fft_img = np.fft.fft2(img)

# Shift our FFT to represent the power spectrum
shift_fft = np.fft.fftshift(fft_img)
# Scale it as well
scaled_fft = 10*np.log10(np.abs(shift_fft))

# Show the original fft
plt.imshow(scaled_fft)
plt.title('Original Power Spectrum', fontdict={'fontsize':18})
plt.tight_layout()
plt.savefig('og_noise_spectrum.png')
plt.show()

# Show zoomed in power spectrum, and removed peaks
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(scaled_fft[290:520, 540:800])
ax[0].set_title('Original Power Spectrum (zoomed)', fontsize=18)
ax[1].imshow(scaled_fft[290:520, 540:800] >= 69)
ax[1].set_title('Periodic noise peaks', fontsize=18)
new_scaled_fft = np.copy(scaled_fft)[290:520, 540:800]
new_scaled_fft[new_scaled_fft >= 69] = 0
ax[2].imshow(new_scaled_fft)
ax[2].set_title('Spectrum with peaks removed', fontsize=18)
plt.tight_layout()
plt.savefig('noise_spectrums.png')
plt.show()

# Remove the peaks from the original shifted fft
shift_fft[scaled_fft >= 69] = 0

# Apply the inverse of our operations
inverse_scale = np.fft.ifftshift(shift_fft)
inverse_img = np.fft.ifft2(inverse_scale)

# Show the final image
plt.imshow(np.real(inverse_img))
plt.title('Image with periodic noise removed', fontdict={'fontsize':18})
plt.tight_layout()
plt.savefig('final_noise.png')
plt.show()
