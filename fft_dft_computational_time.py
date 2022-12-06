import numpy as np
import matplotlib.pyplot as plt
import timeit


## code to calculate the DFT
def DFT(x):

    ''' an optimed version of the discrete fourier transform '''

    # N is the total number of sampled points
    N = len(x)

    # make n go from 0 to N-1
    n = np.arange(N)
    # make k a column vector going from 0 to N-1
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    # calculating the fk sum as a dot product for effeciency
    sum = np.dot(e, x)

    return sum

# the function were sampling
f = lambda x: 4*np.sin(2*x) - 1.5*np.cos(5*x) + 0.3*np.sin(-5*x) + 0.4*np.cos(2*x)

''' plot function and sampled points '''

# Let our interval be from 0 to 4 pi
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

# Plot function and sampled points
plt.plot(fine_xs, fine_fxs, label='Full function')
plt.scatter(xs, fxs, c='r', label='Discrete sample points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('4sin(2x) - 1.5cos(5x) + 0.3sin(0.5x) + 0.4cos(2x)')
plt.show()


''' show the FFT is O(NlogN) and DFT is O(N^2) '''

# pmax is the max power of 2 (2^pmax)
pmax = 10
Dtime = np.zeros(pmax - 3)
Ftime = np.zeros(pmax - 3)
for p in range (3,pmax):

    # make N = 2^p data points
    Np = 2**p
    xs = np.linspace(a,b,Np)
    fxs = f(xs)

    # times DFT for each value of p
    startd = timeit.default_timer()
    DFT_result = (1/N)*DFT(fxs)
    stopd = timeit.default_timer()
    Dtime[p-3] = stopd-startd

    # times FFT for each value of p
    startf = timeit.default_timer()
    fft_result = (1/N)*np.fft.fft(fxs)
    stopf = timeit.default_timer()
    Ftime[p-3] = stopf - startf

# plot the times for DFT and FFT
xtime = np.linspace(3,pmax,len(Dtime))
plt.plot(xtime,Dtime, label='DFT Time')
plt.plot(xtime,Ftime, label='FFT Time')
plt.legend()
plt.title('Time it takes DFT and FFT for different powers of 2^p')
plt.xlabel('p')
plt.ylabel('time (s)')
plt.tight_layout()
plt.savefig('dft_fft_time')
plt.show()


#plot DFT w o(N^2) and FFT w o(NlogN)
n = np.linspace(2**3,2**pmax,1000)
n2 = n**2
nlogn = n*np.log2(n)

fig,ax = plt.subplots(1,2)

ax[0].plot(xtime,Dtime, label='DFT Time')
ax[0].plot(xtime,Ftime, label='FFT Time')
ax[0].set_xlabel('2^p')
ax[0].set_ylabel('time (s)')
ax[0].legend()
ax[0].set_title('DFT vs FFT computational time')

ax[1].plot(n,n2, label='O(N^2)')
ax[1].plot(n,nlogn, label = 'O(NlogN)')
ax[1].set_xlabel('N')
ax[1].legend()
ax[1].set_title('O(N^2) vs O(NlogN)')

plt.tight_layout()
plt.savefig('ON2_ONlogN')
plt.show()

''' Now show DFT and FFT arrive at same results '''

# plot DFT and FFT results for N = 128
N = 128
xs = np.linspace(a,b,N)
fxs = f(xs)

DFT_result = (1/N)*DFT(fxs)

fft_result = (1/N)*np.fft.fft(fxs)

realF = np.real(fft_result)
imagF = np.imag(fft_result)
realD = np.real(DFT_result)
imagD = np.imag(DFT_result)
# And finally, to get the amplitudes of components, we can apply the following:
cos_ampF = np.abs(2*realF)
sin_ampF = np.abs(2*imagF)
cos_ampD = np.abs(2*realD)
sin_ampD = np.abs(2*imagD)

# The actual amplitudes of our values from above
ampsF = np.sqrt(np.square(cos_ampF) + np.square(sin_ampF))
ampsD = np.sqrt(np.square(cos_ampD) + np.square(sin_ampD))

# The frequencies of our matched up amplitudes
freqs = np.linspace(0, N/sample_rate - 1, N)
# Let's just look at the first 20 points

#plot them next to each other
fig,bx = plt.subplots(1,2)

bx[0].stem(freqs[:20], ampsD[:20], label='Average amplitudes')
bx[0].set_title('DFT amplitudes')
bx[0].set_xlabel('frequency')
bx[0].set_ylabel('amplitude')

bx[1].stem(freqs[:20], ampsF[:20], label='Average amplitudes')
bx[1].set_title('FFT amplitudes')
bx[1].set_xlabel('frequency')

plt.tight_layout()
plt.savefig('dft_vs_fft')
plt.show()


#plot the error between DFT and FFT
plt.plot(freqs[:20],abs(ampsD[:20] - ampsF[:20]))
plt.title('abs error between DFT and FFT coefficient')
plt.xlabel('frequency')
plt.ylabel('abs error')
plt.tight_layout()
plt.savefig('dft_fft_errors')
plt.show()
