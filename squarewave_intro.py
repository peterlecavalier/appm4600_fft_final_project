import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def real_imag(fft_result):
    # To actually get the magnitudes of our frequencies,
    # we can do the following to separate the real and imaginary components:
    real = np.real(fft_result)
    imag = np.imag(fft_result)
    # And finally, to get the amplitudes of components, we can apply the following:
    cos_amp = 2*real
    sin_amp = -2*imag

    # The actual amplitudes of our values from above
    amps = np.sqrt(np.square(cos_amp) + np.square(sin_amp))
    return cos_amp,sin_amp

"""Example FFT on clean function data"""

# Numpy has a built-in fft package
# We can use it here on a simple combination of sin/cos waves
# With this equation, we have the following:
#   ->frequency=2 with an average amplitude~=4.02
#   ->frequency=5 with an average amplitude~=1.53
f = lambda x: signal.square(x)

# Let's sample N points over the interval [a,b]
N = 128
a = 0
b = 4*np.pi - (1/N)
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
plt.title('Square Wave - Sampled with 128 points')
#plt.savefig(f'{N}sample_fft_func.png')
plt.show()

# Apply the FFT to our function data
fft_result = (1/N)*np.fft.fft(fxs)

# The actual magnitudes of our values from above
real,imag = real_imag(fft_result)

# The frequencies of our matched up amplitudes
freqs = np.linspace(0, N/sample_rate - 1, N)
# Let's just look at the first 20 points
fig,ax = plt.subplots(1,3,figsize=(15, 10))
ax[0].plot(fine_xs,fine_fxs,label='Square wave signal')
ax[0].scatter(xs,fxs,c='r',label='Discrete sample points')
ax[1].stem(freqs[:20], real[:20],'g', markerfmt='go' ,label='real coefficients = An')
ax[1].set_ylim(-.06,1.334)
ax[2].stem(freqs[:20], imag[:20],'k', markerfmt='ko' ,label='imaginary coefficients = Bn')
ax[1].legend()
ax[2].legend()
fig.suptitle('Absolute value of real and imaginary coefficients from the FFT')
plt.tight_layout()
#plt.savefig(f'{N}sample_fft_stems.png')
plt.show()

# As expected (with slight errors due to sample size), we get the correct peaks at
# frequencies 2 and 5, with amplitudes ~4.02 and ~1.53 respectively.


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
plt.title('Square Wave - Inverse FFT')
#plt.savefig(f'{N}sample_inverse_fft_func.png')
plt.show()

# Not surprisingly, the inverse gives us back our signal almost exactly!
