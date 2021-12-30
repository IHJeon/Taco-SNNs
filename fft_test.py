from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng()

T=10
fs = 1e3*T
N =  1e4*T
amp = 2 * np.sqrt(2)
#time = np.arange(N) / float(fs)
time = np.arange(0,N/fs,1/fs) 

phase = fs*0.1*np.cos(2*np.pi*time*0.25*4)

carrier = amp * np.sin(2*np.pi*fs*0.3*time*1 + phase)


noise_power = 0.01 * fs / 2
noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise



WINDOW_SIZE=int(200)
f, t, Sxx = signal.spectrogram(x, fs,\
     nfft=WINDOW_SIZE, nperseg=WINDOW_SIZE, noverlap=int(WINDOW_SIZE/2))#,\
                    #nfft=WINDOW_SIZE)
                        # noverlap=int(WINDOW_SIZE*0.8))

print('sampling rate:', int(fs), 'Nyquist Freq:', int(fs/2),'Window size:', WINDOW_SIZE, \
    'time steps:', len(time), 'max t:',max(time),\
         'phase range:', min(phase), '-',max(phase))

spmt=Sxx
print('shape spmt[0]',np.shape(spmt[0]))
print('shape spmt[1]',np.shape(spmt[1]))
print('shape spmt[2]',np.shape(spmt[2]))
print('shape spmt[3]',np.shape(spmt[3]))

print('Freq range:', np.max(spmt[1]), 'sample rate:', len(spmt[1]))
print('Time range:', np.max(spmt[2]), 'sample rate:', len(spmt[2]))

plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()