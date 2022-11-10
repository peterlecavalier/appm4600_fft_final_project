import numpy as np
import matplotlib.pyplot as plt
import timeit


## code to calculate the DFT
def DFT(x):

    ''' an optimed version of the discrete fourier transform '''

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    # calculating the fk sum as a dot product for effeciency
    sum = np.dot(e, x)

    return sum

# the function were sampling
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
''' whats new '''

pmax = 10
Dtime = np.zeros(pmax - 1 -3)
Ftime = np.zeros(pmax - 1 - 3)
for p in range (3,pmax):

    N = 2**p
    xs = np.linspace(a,b,N)
    fxs = f(xs)

    startd = timeit.default_timer()
    DFT_result = (1/N)*DFT(fxs)
    stopd = timeit.default_timer()
    Dtime[p-3] = stopd-startd

    startf = timeit.default_timer()
    fft_result = (1/N)*np.fft.fft(fxs)
    stopf = timeit.default_timer()
    Ftime[p-3] = stopf - startf

xtime = np.linspace(a,b,len(Dtime))
plt.plot(xtime,Dtime)
plt.plot(xtime,Ftime)
plt.show()


startd = timeit.default_timer()
DFT_result = (1/N)*DFT(fxs)
stopd = timeit.default_timer()
print('DFT time: ', stopd - startd)

startf = timeit.default_timer()
fft_result = (1/N)*np.fft.fft(fxs)
stopf = timeit.default_timer()
print('FFT time: ', stopf - startf)



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
plt.stem(freqs[:20], amps[:20], label='Average amplitudes')
plt.legend()
plt.title('Average amplitude of each frequency value from the FFT')
plt.show()

# As expected (with slight errors due to sample size), we get the correct peaks at
# frequencies 2 and 5, with amplitudes ~4.02 and ~1.53 respectively.
