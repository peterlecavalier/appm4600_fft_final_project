import numpy as np
import matplotlib.pyplot as plt

def magnitudes(fft_result):
    # To actually get the magnitudes of our frequencies,
    # we can do the following to separate the real and imaginary components:
    real = np.real(fft_result)
    imag = np.imag(fft_result)
    # And finally, to get the amplitudes of components, we can apply the following:
    cos_amp = np.abs(2*real)
    sin_amp = np.abs(2*imag)

    # The actual amplitudes of our values from above
    amps = np.sqrt(np.square(cos_amp) + np.square(sin_amp))
    return amps

"""Example FFT on clean function data"""

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
plt.figure(figsize=(10, 10))
plt.plot(fine_xs, fine_fxs, label='Full function')
plt.scatter(xs, fxs, c='r', label='Discrete sample points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('4sin(2x) - 1.5cos(5x) + 0.3sin(-5x) + 0.4cos(2x) - Sampled with 128 points')
plt.tight_layout()
plt.savefig(f'{N}sample_fft_func.png')
plt.show()

# Apply the FFT to our function data
fft_result = (1/N)*np.fft.fft(fxs)

# The actual magnitudes of our values from above
mags = magnitudes(fft_result)

# The frequencies of our matched up amplitudes
freqs = np.linspace(0, N/sample_rate - 1, N)
# Let's just look at the first 20 points
plt.figure(figsize=(10, 10))
plt.stem(freqs[:20], mags[:20], label='Average amplitudes')
plt.xlabel('Frequency value')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Average amplitude of each frequency value from the FFT')
plt.tight_layout()
plt.savefig(f'{N}sample_fft_stems.png')
plt.show()

# As expected (with slight errors due to sample size), we get the correct peaks at
# frequencies 2 and 5, with amplitudes ~4.02 and ~1.53 respectively.

"""Example FFT on noisy function data"""

# Additionally, we can add noise to our function sample points to show
# that the FFT still functions, while doing a good job of noise reduction.
# Let's randomly offset each of our samples across the normal distribution:
offsets = np.random.normal(size=fxs.size)
fxs_offset = fxs + offsets

# The code below this is nearly identical to that explained above

# Plot everything
plt.figure(figsize=(10, 10))
plt.plot(fine_xs, fine_fxs, label='Full function')
plt.scatter(xs, fxs_offset, c='r', label='Discrete sample points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('4sin(2x) - 1.5cos(5x) + 0.3sin(-5x) + 0.4cos(2x) - Sampled with 128 noisy points')
plt.tight_layout()
plt.savefig(f'{N}sample_noisy_fft_func.png')
plt.show()

# Apply the FFT to our function data
fft_result_offset = (1/N)*np.fft.fft(fxs_offset)

# Get the magnitudes of our frequencies:
offset_mags = magnitudes(fft_result_offset)

# Let's just look at the first 20 points
plt.figure(figsize=(10, 10))
plt.stem(freqs[:20], offset_mags[:20], label='Average amplitudes')
plt.xlabel('Frequency value')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Average amplitude of each frequency value from the noisy FFT')
plt.tight_layout()
plt.savefig(f'{N}sample_noisy_fft_stems.png')
plt.show()

# As expected (with slight errors due to sample size), we get the correct peaks at
# frequencies 2 and 5, with amplitudes ~4.02 and ~1.53 respectively.

# This result is almost identical to our previous result without noise.
# This highlights another useful task that Fourier Transforms accomplish,
# filtering noise in data.

"""Taking the inverse of our original transformed function data"""
# Lastly, we can take the inverse of our fft data from the previous noisy example
# to see how close to our original plot we are:
s = N*np.fft.ifft(fft_result)

plt.figure(figsize=(10, 10))
plt.plot(fine_xs, fine_fxs, label='Full function')
plt.scatter(xs, s.real, c='r', label='Discrete inversed points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('4sin(2x) - 1.5cos(5x) + 0.3sin(-5x) + 0.4cos(2x) - Inverse FFT')
plt.tight_layout()
plt.savefig(f'{N}sample_inverse_fft_func.png')
plt.show()

# Not surprisingly, the inverse gives us back our signal almost exactly!

"""Working with non-power of 2 sample size"""

# As described before, FFT can only work with data that has sample size of
# 2^n (n is an integer). To work around this with sample sizes that aren't
# powers of 2, FFT will pad the data with zeros to get to the nearest
# power of 2 before transforming.

# Let's try this on the function we've been working with:
# With this equation, we have the following:
#   ->frequency=2 with an average amplitude~=4.02
#   ->frequency=5 with an average amplitude~=1.53
f = lambda x: 4*np.sin(2*x) - 1.5*np.cos(5*x) + 0.3*np.sin(-5*x) + 0.4*np.cos(2*x)

a = 0
b = 4*np.pi
# The sample rate is essentially the number of periods our function goes through
sample_rate = (b-a)/(2*np.pi)

# Let's sample 96 points over the interval [a,b]
N = 96
xs = np.linspace(a, b, N)
fxs = f(xs)

# Apply the FFT to our function data
# Inside this function, it will pad up to make the input size 128
# Although this isn't how Numpy would usually handle this,
# this forces a padding up to 2^n.
padded_N = 128
padded_fft_result = (1/N)*np.fft.fft(fxs, n=padded_N)

padded_freqs = np.linspace(0, N/sample_rate - 1, padded_N)

padded_mags = magnitudes(padded_fft_result)

plt.figure(figsize=(10, 10))
plt.stem(padded_freqs[:20], padded_mags[:20], label='Average amplitudes')
plt.xlabel('Frequency value')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Average amplitude of each frequency value from the padded FFT')
plt.tight_layout()
plt.savefig(f'{N}sample_padded_fft_stems.png')
plt.show()