import numpy as np
import matplotlib.pyplot as plt

# Numpy has a built-in fft package
# We can use it here on a simple combination of sin/cos waves
# With this equation, we have the following:
#   ->frequency=2 with an average amplitude~=4.02
#   ->frequency=5 with an average amplitude~=1.53
f = lambda x: 4*np.sin(2*x) - 1.5*np.cos(5*x) + 0.3*np.sin(-5*x) + 0.4*np.cos(2*x)

# Let's sample N points over the interval [a,b]
N = 128
a = 0
b = 4*np.pi
xs = np.linspace(a, b, N)
fxs = f(xs)

# The sample rate is essentially the number of periods our function goes through
sample_rate = (b-a)/(2*np.pi)

# Just used to finely plot the function
fine_xs = np.linspace(a, b, 10000)
fine_fxs = f(fine_xs)

# Plot everything
plt.plot(fine_xs, fine_fxs, label='Full function')
plt.scatter(xs, fxs, c='r', label='Discrete sample points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('4sin(2x) - 1.5cos(5x) + 0.3sin(0.5x) + 0.4cos(2x)')
plt.show()

# Apply the FFT to our function data
fft_result = (1/N)*np.fft.fft(fxs)

# To actually get the magnitudes of our frequencies from this,
# we can do the following to separate the real and imaginary components:
real = np.real(fft_result)
imag = np.imag(fft_result)
# And finally, to get the amplitudes of components, we can apply the following:
cos_amp = np.abs(2*real)
sin_amp = np.abs(2*imag)

# The actual amplitudes of our values from above
amps = np.sqrt(np.square(cos_amp) + np.square(sin_amp))

# The frequencies of our matched up amplitudes
freqs = np.linspace(0, N/sample_rate - 1, N)
# Let's just look at the first 20 points

#plt.plot(freqs[:50], cos_amp[:50], label='Real amplitudes')
#plt.plot(freqs[:50], sin_amp[:50], label='Imaginary amplitudes')

plt.stem(freqs[:20], amps[:20], label='Average amplitudes')
plt.legend()
plt.title('Average amplitude of each frequency value from the FFT')
plt.show()

# As expected (with slight errors due to sample size), we get the correct peaks at
# frequencies 2 and 5, with amplitudes ~4.02 and ~1.53 respectively.

# Additionally, we can add noise to our function sample points to show
# that the FFT still functions, while doing a good job of noise reduction.
# Let's randomly offset each of our samples across the normal distribution:
offsets = np.random.normal(size=fxs.size)
fxs_offset = fxs + offsets

# TODO: Modify comments/variable names below to look cleaner for noise

# Plot everything
plt.plot(fine_xs, fine_fxs, label='Full function')
plt.scatter(xs, fxs_offset, c='r', label='Discrete sample points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('4sin(2x) - 1.5cos(5x) + 0.3sin(0.5x) + 0.4cos(2x)')
plt.show()

# Apply the FFT to our function data
fft_result_offset = (1/N)*np.fft.fft(fxs_offset)

# To actually get the magnitudes of our frequencies from this,
# we can do the following to separate the real and imaginary components:
real = np.real(fft_result_offset)
imag = np.imag(fft_result_offset)
# And finally, to get the amplitudes of components, we can apply the following:
cos_amp = np.abs(2*real)
sin_amp = np.abs(2*imag)

# The actual amplitudes of our values from above
amps = np.sqrt(np.square(cos_amp) + np.square(sin_amp))

# The frequencies of our matched up amplitudes
freqs = np.linspace(0, N/sample_rate - 1, N)
# Let's just look at the first 20 points

#plt.plot(freqs[:50], cos_amp[:50], label='Real amplitudes')
#plt.plot(freqs[:50], sin_amp[:50], label='Imaginary amplitudes')

plt.stem(freqs[:20], amps[:20], label='Average amplitudes')
plt.legend()
plt.title('Average amplitude of each frequency value from the FFT')
plt.show()

# As expected (with slight errors due to sample size), we get the correct peaks at
# frequencies 2 and 5, with amplitudes ~4.02 and ~1.53 respectively.

# TODO: Calculate the inverse FFT and show that it works