import numpy as np
import matplotlib.pyplot as plt


# Numpy has a built-in fft package
# We can use it here on a simple combination of sin/cos waves
f = lambda x: 4*np.sin(0.5*np.pi*x)# - 1.5*np.cos(2*np.pi*x)# + 0.3*np.sin(0.5*x) + 0.4*np.cos(2*x)

# Let's sample 100 points over the interval [-2pi, 2pi]
N = 128
xs = np.linspace(0, 2, N)
fxs = f(xs)
diff = np.diff(xs)[0]

# Just used to finely plot the function
fine_xs = np.linspace(0, 2, 10000)
fine_fxs = f(fine_xs)


# Plot everything
plt.plot(fine_xs, fine_fxs, label='Full function')
plt.scatter(xs, fxs, c='r', label='Discrete sample points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('4sin(2x) - 1.5cos(5x) + 0.3sin(0.5x) + 0.4cos(2x)')
plt.show()

# Now if we apply a FFT, we should see
fft_result = (1/N)*np.fft.fft(fxs)

# To actually get the magnitudes of our frequencies from this,
# we can do the following to separate the real and imaginary components:
real = np.real(fft_result)#[i.real for i in fft_result]
imag = np.imag(fft_result)#[i.imag for i in fft_result]
# And finally, to get the amplitudes of components, we can apply the following:
cos_amp = 2*real
sin_amp = 2*imag

#freqs = np.sqrt(np.square(real) + np.square(imag))

freq_xs = np.linspace(0, N, N)

print(freq_xs[np.argmin(sin_amp)])

plt.plot(freq_xs, cos_amp, label='Cosine Amplitudes')
plt.plot(freq_xs, sin_amp, label='Sine Amplitudes')
plt.legend()
#plt.hist(freqs, 1000)
plt.show()
#print(freqs)