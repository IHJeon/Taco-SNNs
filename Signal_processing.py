from matplotlib import pyplot as plt
import numpy as np
import sys
from seaborn import heatmap
#from scipy.sparse import data
from scipy.fft import fft, fftfreq
import math

def rand_ind(Low=0, max_val=100):
        return np.random.randint(0,max_val)
def rand_normal():        
    return np.random.normal(scale=0.05)        

def data_generation(input_size, input_range, mode='random'):
    data = []    
    max_level_stimuli=1
    
    from sklearn.preprocessing import MinMaxScaler        
    
    t= np.linspace(0,input_range,input_range)
    #seed_val=np.random.randint(0,100)
    
    for inpt in range(input_size):        
        if mode=='random': 
            y=rand_ind()*np.sin(rand_normal()*t)+rand_ind()*np.cos(rand_normal()*t)
        elif mode=='1': 
            dt=0.001
            TIME_STEP=input_range
            t=np.arange(0, TIME_STEP*dt, dt)            
            y=np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)
            #y=np.sin(x*4*np.pi/input_range)*max_level_stimuli/2+50
            f0=50
            f1=500
            t1=TIME_STEP*dt
            t=np.arange(0,t1,dt)
            y= np.cos(2*np.pi*t*
                    (f0+(f1-f0)*np.power(t,2))/(3*t1**2)
                    )
            
            f0=50
            f1=500
            t1=TIME_STEP*dt
            t=np.arange(0,t1,dt)
            y= np.cos(2*np.pi*t*
                    (f0+(f1-f0)*np.power(t,2))/(3*t1**2)
                    )
            #y*=max_level_stimuli*0.5
            #y+=max_level_stimuli*0.5
            print('len t', len(t), 'len y', len(y), 'max y:', max(y))
        elif mode=='2': y=np.cos(0.05*t)*max_level_stimuli + max_level_stimuli/2
        #elif mode=='3': y=np.flip(x)/input_range*max_level_stimuli*2/3
        elif mode=='3': y=t/input_range*max_level_stimuli*2/3
        elif mode=='4': y=[max_level_stimuli]*int(len(t)/2)+[0]*(len(t)-int(len(t)/2))
        elif mode=='5': y=[0]*(len(t)-int(len(t)/2))+[max_level_stimuli]*int(len(t)/2)
        elif mode=='6': y=np.ones(len(t))*max_level_stimuli
        #y=MinMaxScaler(feature_range=(0,100)).fit_transform(y.reshape(-1, 1)).reshape(-1)
        #y=MinMaxScaler(feature_range=(0,max_level_stimuli)).fit_transform(y.reshape(-1, 1)).reshape(-1)
        #y=abs(y)
        #print('y max:', max(y), 'y min:', int(min(y)))
        #y=np.clip(y, 0, max_level_stimuli)        
        data.append(y)
        plt.plot(t,y)
        plt.title('Raw Stimuli input')
        plt.xlabel('Time steps')
        plt.ylabel('Signal intensity')
        plt.show()
        
    print('len y:',len(y), 'Sum y:', np.sum(y), 'AVG y', np.sum(y)/len(y))
    #print('Signal Data shape:',np.shape(data), 'Sum data:', np.sum(data), 'AVG Data', np.sum(data)/(input_size*num_input) )
    #print('y',y)
    #data=np.array(data).T
    data=np.array(data)
    return data

def stimuli_IMPACTmatrix_converter(data, MAX_VAL, OUTPUT_NUM, TIME_STEP, PRNT=True):
    #NUM_MFs=HALFNUM
    IMPACT_MATRIX=np.zeros((OUTPUT_NUM, TIME_STEP))
    coverage_ratio=int(MAX_VAL/OUTPUT_NUM)
    #data_amp_normalized=(data/coverage_ratio).astype(int)
    
    print('data shape', np.shape(data), 'IMPACT_MATRIX', np.shape(IMPACT_MATRIX),\
            'coverage_ratio',coverage_ratio)
    #print('data_amp_normalized', np.shape(data_amp_normalized))
    for ind_mf in range(OUTPUT_NUM):        
        ind_val_range=(data>=ind_mf*coverage_ratio)&(data<(ind_mf+1)*coverage_ratio)
        indices=np.where(ind_val_range, 1, 0)
        #indices=np.where(data in val_range, 1, 0)
        #print('range', ind_mf*coverage_ratio,'to', (ind_mf+1)*coverage_ratio)
        IMPACT_MATRIX[ind_mf]=indices

    
    #sys.exit()
    
    coordinate= np.argwhere(IMPACT_MATRIX).T

    i, j = coordinate
    if PRNT:
        print('shape impact mat.:', np.shape(IMPACT_MATRIX))

        #print(np.flip(IMPACT_MATRIX, axis=0))
        #print('i', i)
        #print('j', j)
        plt.scatter(j, i, marker='|', label='STIMULI dist')    
        plt.title('PRIMARY POINTs IMPACT MATRIX')
        plt.axhline(y=OUTPUT_NUM, color='c', linestyle=':', label='Max')
        plt.axhline(y=0, color='c', linestyle=':', label='Min')
        plt.show()
    return IMPACT_MATRIX, coordinate

def Neighboring_IMPACT_calculator(data, NUM_MFs, TIME_STEP, \
                                IMPACT_MATRIX, coordinate, PRNT=False):
    HALF_NUM=int(NUM_MFs/2)
    initial_amplitude=50
    IMPACT_MATRIX=IMPACT_MATRIX*initial_amplitude    
    amplitude_weight=10 #same weight to every primary points
    CUTOFF_DIST_SQ=int(TIME_STEP/10)
    for coord in coordinate.T:
        for j in [j for j in range(HALF_NUM) if (j-coord[0])**2<CUTOFF_DIST_SQ and j!=coord[0]]:            
            for i in [i for i in range(TIME_STEP) if (i-coord[1])**2<CUTOFF_DIST_SQ and i!=coord[1]]:
                distance=(i-coord[1])**2+(j-coord[0])**2
                if distance<CUTOFF_DIST_SQ:
                    IMPACT_MATRIX[j, i]+=np.exp(-distance)*amplitude_weight

    IMPACT_MATRIX=np.around(IMPACT_MATRIX, 0)
    if PRNT:
        coordinate= np.argwhere(IMPACT_MATRIX).T
        i, j = coordinate
        plt.scatter(j, i, marker='|', label='STIMULI dist')    
        plt.title('ALL effect IMPACT MATRIX')
        plt.show()
        print(np.flip(IMPACT_MATRIX, axis=0))    
    
    #print(np.shape(IMPACT_MATRIX))
    #sys.exit()
    return np.flip(IMPACT_MATRIX, axis=0)
'''
def FREQUENCY_COUNTER(IMPACT_MATRIX, INTEGER=True, PRNT=False):
    if not INTEGER: raise Exception("The data for Frequency to count is not INTEGER")
    #IMPACT_MATRIX=np.flip(IMPACT_MATRIX, axis=0)    
    #print('MATRIX values:\n', IMPACT_MATRIX)
    #MAXNUM=np.amax(data)        
    frequency=np.sum(IMPACT_MATRIX, axis=1)
    if PRNT:
        plt.bar(np.arange(len(frequency)),frequency)
        #print('shape.T sum:', np.shape(data.T))

        #bin_range = int((max(data)) - min(data))+1
        #x, bins, patch = plt.hist(data, bins=bin_range)
        #hist, edges = np.histogram(data, bins=bin_range)
        plt.title('Frequency Count distribution of IMPACT MATRIX')
        plt.xlabel('Index')
        plt.ylabel('Counts')
        plt.show()

    #UNIQUE EXample
    #a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
    #unique, counts = numpy.unique(a, return_counts=True)
    #dict(zip(unique, counts))
    return frequency

def MOVING_FREQUENCY_MATRIX(IMPACT_MATRIX, bin=10, INTEGER=True, PRNT=True):
    if not INTEGER: raise Exception("The data for Frequency to count is not INTEGER")
    #IMPACT_MATRIX=np.flip(IMPACT_MATRIX, axis=0)    
    #print('MATRIX values:\n', IMPACT_MATRIX)
    #MAXNUM=np.amax(data)
    numind=np.shape(IMPACT_MATRIX)[0]
    numtime=np.shape(IMPACT_MATRIX)[1]
    MOVING_FREQ_MAT=np.zeros((numind, numtime))
    for ind in range(numind):
        for t in range(numtime):
            MOVING_FREQ_MAT[ind, t]=np.sum(IMPACT_MATRIX[ind, t:t+bin])
    #print('MATRIX values:\n', IMPACT_MATRIX)
    #print('MOVING_FREQ_MAT\n', MOVING_FREQ_MAT)
    if PRNT:        
        #print('shape.T sum:', np.shape(data.T))
        ax = heatmap(MOVING_FREQ_MAT,  cmap="YlGnBu")
        #ax.invert_yaxis()
        #bin_range = int((max(data)) - min(data))+1
        #x, bins, patch = plt.hist(data, bins=bin_range)
        #hist, edges = np.histogram(data, bins=bin_range)
        ax.invert_yaxis()
        plt.title('Moving Frequency count distribution of IMPACT MATRIX')
        plt.xlabel('Time')
        plt.ylabel('Index')
        plt.show()

    #UNIQUE EXample
    #a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
    #unique, counts = numpy.unique(a, return_counts=True)
    #dict(zip(unique, counts))
    return MOVING_FREQ_MAT
'''


def spectrogram_EX(x=[], dt=0.001):
    if len(x)==0:        
        f0=50
        f1=500
        t1=10
        t=np.arange(0,t1,dt)
        x= np.cos(2*np.pi*t*
                (f0+(f1-f0)*np.power(t,2))/(3*t1**2)
                )
    else:
        t=np.arange(0, len(x)*dt, dt)
        
    #x=t*100
    #dt=0.001
    #TIME_STEP=int(1e3)
    #t=np.arange(0, TIME_STEP*dt, dt)
    #x=np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)
    print('specgram param', 'len x:', len(x), 'len t:', len(t), 'dt:', dt)
    fig, axs = plt.subplots(3,1)
    plt.sca(axs[0])
    plt.title("Signals")
    plt.plot(t,x)


    plt.sca(axs[1])
    plt.title("Spectrogram")
    #plt.specgram(x, NFFT=128, Fs=1/dt, noverlap=120)

    num_MFs=100
    coverage_ratio=2
    Num_frames=1000
    nfft=2*(num_MFs*coverage_ratio)-2
    n_overlap=nfft-((len(t)-nfft)/(Num_frames-1))
    spectrogram, freqs, spec_t, im=\
        plt.specgram(x, NFFT=nfft, Fs=1/dt, noverlap=n_overlap)
    

    import math
    slide_size=math.ceil(nfft-n_overlap)    
    n_slides=math.floor((len(t)-nfft)/slide_size)+1
    n_freq=math.floor(nfft/2)+1

    print('nfft:', nfft, 'n_overlap:', n_overlap)
    print('slide_size:', slide_size,'n_slides:', n_slides, 'n_freq', n_freq)    
    print('specgram returns:', np.shape(spectrogram), np.shape(freqs), \
        np.shape(spec_t), np.shape(im))
    
    plt.sca(axs[2])
    ax = heatmap(spectrogram,  cmap="YlGnBu")
    ax.invert_yaxis()
    plt.title('Spectrogram, Heatmap')
    plt.xlabel('# frames.')
    plt.ylabel('# frequency steps')
    plt.show()            
    #import sounddevice as sd
    #sd.play(1*x, 1/dt)
    #sd.wait()


def Power_spectrum_EX():
    dt=0.001
    t=np.arange(0, 1, dt)
    f=np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)
    noise = 2.5*np.random.randn(len(t))

    n=len(t)
    fhat=np.fft.fft(f,n)
    PSD=fhat*np.conj(fhat)/n #power spectrum of each freq.
    #Power Spectral Density; Power specturm, enenrgy SD.
    #freq = (1/(dt*n))*np.arange(n)
    freq = np.fft.fftfreq(n)/dt
    L = np.arange(1, np.floor(n/2), dtype='int')

    #PSD=np.real(PSD)
    print(np.where(PSD[L]>1))
    print(len(PSD[L]))
    #sys.exit()

    fig, axs = plt.subplots(2,1)
    plt.sca(axs[0])
    plt.plot(t,f+noise, label='Noisy')
    plt.plot(t,f, label='Clean')
    plt.xlim(t[0], t[-1])
    plt.legend()
    plt.title("Signals")

    plt.sca(axs[1])
    #plt.plot(freq[L], abs(fhat[L]), label='Noisy abs')
    plt.plot(freq[L],PSD[L], label='Noisy PSD')
    #plt.plot(freq[L],fhat[L],  label='Noisy fhat')

    plt.xlim(freq[L[0]],freq[L[-1]])
    plt.legend()
    plt.title("power Spectrum")

    plt.show()



def Power_spectrum_at_t(f, TIME_UNIT=1e-03, PRINT=False):
    
    if not len(f)==0:        
        dt=TIME_UNIT
        t=np.arange(0, len(f)*dt, dt)        
    else:
        print('showing sample PSD')
        dt=TIME_UNIT
        t=np.arange(0, 1, dt)
        f=np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)
    
    n=len(t)
    fhat=np.fft.fft(f,n)
    PSD=fhat*np.conj(fhat)/n #power spectrum of each freq.
    PSD=np.real(PSD)
    #Power Spectral Density; Power specturm, enenrgy SD.
    #freq = (1/(dt*n))*np.arange(n)
    freq = np.fft.fftfreq(n)/dt
    L = np.arange(0, np.floor(n/2), dtype='int')

    #print(np.where(PSD[L]>1))
    
    #print(freq[np.where(PSD[L]>1)],  PSD[L][np.where(PSD[L]>1)])
    
    if PRINT:
        print('PSD len f:', len(f),  'len t:', len(t), 'dt:',dt)
        print('len L:',len(L),'len(PSD[L]):', len(PSD[L]))
        #print('f val. sample', f[:5])
        #print('fhat val. sample:', fhat[:5])
        #print('PSD  val. sample:', PSD[:5])
        #sys.exit()

        fig, axs = plt.subplots(2,1)
        plt.sca(axs[0])    
        plt.plot(t,f, label='Signals at t')
        plt.xlim(t[0], t[-1])
        plt.legend()
        plt.title("Signals at t")

        plt.sca(axs[1])
        #plt.plot(freq[L], abs(fhat[L]), label='Noisy abs')
        
        if len(freq[L])<100:
            plt.plot(freq[L],PSD[L], label='PSD', marker='o')
        else: plt.plot(freq[L],PSD[L], label='PSD')
        
        #plt.plot(freq[L],fhat[L],  label='Noisy fhat')

        plt.xlim(freq[L[0]],freq[L[-1]])
        plt.legend()
        plt.title("power Spectrum")

        plt.show()

    Freq_components = freq[np.where(PSD[L]>1)]
    Freq_amplitude = PSD[L][np.where(PSD[L]>1)]        
    return Freq_components,  Freq_amplitude
    
    
def FREQ_MAP(f, TIME_STEP, NUMCELL, SR=1/1e-03, PRINT=True):
    freq_map=np.zeros((TIME_STEP, NUMCELL))   #Freq range = (0 ~ 1/dt *0.5); max=500
    #time_bin=int(len(f)/10)
    #Time_sampling_rate=SR
    dt=1/SR
    coverage_ratio=int(SR*0.5/NUMCELL)
    print('SR:',SR, 'FREQ_MAP input shape:', np.shape(f), 'freq map size:',np.shape(freq_map) ,\
            'frequency coverage ratio:',coverage_ratio)
    #step=int(TIME_STEP/(SR/10))
    #step=300
    step=int(SR*dt)
    print('Time loop:', len(range(0, TIME_STEP, step)), 'step=', step)
    #for i in range(0, TIME_STEP, int(TIME_STEP*dt)):
    #window_size=int(SR/10)
    window_size=256
    print('window_size', window_size)
    for i in range(0, len(f), step):
        #comp, amp = Power_spectrum_at_t(f[i:i+int(SR/2)], dt)        
        comp, amp = Power_spectrum_at_t(f[i:i+window_size], dt, PRINT=False)
        amp=amp.astype(int)
        #if i==TIME_STEP-101:
        if i==0:
            Power_spectrum_at_t(f[i:i+window_size], dt, PRINT=True)
            #print('comp, amp shape:', np.shape(comp), np.shape(amp))
            print('comp, amp value:', comp, amp)
        #print('comp, amp value:', comp, '\n', amp)
        if not sum(amp)==0:
            #ind=(comp/coverage_ratio).astype(int) #Freq bin per cell = coverage_ratio
            cell_input=np.zeros(NUMCELL)
            #print('freq amp',amp)
            for freq_ind in [f_ind for f_ind in range(len(comp)) if amp[f_ind]!=0]:
                cell_input[int(comp[freq_ind]/coverage_ratio)]+=amp[freq_ind]
            #print('cell_input', cell_input)
            
            freq_map[i]=cell_input            

    freq_map=freq_map.T
    #print(np.where(freq_map>0))
    if PRINT:
        ax = heatmap(freq_map,  cmap="YlGnBu")
        ax.invert_yaxis()
        plt.title('Freq Matrix, Heatmap')
        plt.xlabel('Time step')
        plt.ylabel('Freq input to Cell index')
        plt.show()
    return freq_map



dt=0.001
TIME_STEP=int(1e3)
t=np.arange(0, TIME_STEP*dt, dt)
f=np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)

#dt=0.001
#f0=50
#f1=250
#t=np.arange(0,TIME_STEP*dt,dt)
#f= np.cos(2*np.pi*t*
#            (f0+(f1-f0)*np.power(t,2))/(3*(TIME_STEP*dt)**2)
#            )

#f=np.ones(len(t))
#Power_spectrum_at_t(f[:100], PRINT=True)

Num_cell=100

#FREQUENCY=FREQ_MAP(f, TIME_STEP, Num_cell)
'''
freq_map=np.zeros((TIME_STEP, Num_cell))   #Freq range = (0 ~ 1/dt *0.5); max=500

for i in range(TIME_STEP):    
    comp, amp = Power_spectrum_at_t(f[i:i+100])
    if len(comp)>0:
        ind=(comp/5).astype(int)          #Freq bin per cell = 5
        freq_map[i][ind]+=amp
    

#print(np.where(freq_map>0))

ax = heatmap(freq_map.T,  cmap="YlGnBu")
ax.invert_yaxis()
plt.title('Freq Matrix, Heatmap')
plt.xlabel('Time step')
plt.ylabel('Freq input to Cell index')
plt.show()'''

def return_specgram(signal_input, Num_MFs, signal_SR=1000, Num_frames=1000, PRINT=True):
    x=signal_input
    dt=1/signal_SR
    t=np.arange(0, len(x)*dt, dt)


    #coverage_ratio=(signal_SR/2)/Num_MFs
    coverage_ratio=1
    
    #nfft=int(2*(Num_MFs*coverage_ratio)-2)
    nfft=int(2*(Num_MFs*coverage_ratio)-2)
    n_overlap=nfft-((len(t)-nfft)/(Num_frames-1))
        
    
    fig, axs = plt.subplots(3,1)
    plt.sca(axs[0])
    plt.title("Signals")
    plt.plot(t,x)


    plt.sca(axs[1])
    plt.title("Spectrogram")
    spectrogram, freqs, spec_t, im=\
        plt.specgram(x, NFFT=nfft, Fs=signal_SR, noverlap=n_overlap)
    
    if PRINT: 
        print('specgram param', 'len x:', len(x), 'len t:', len(t), 'dt:', dt)
        slide_size=math.ceil(nfft-n_overlap)    
        n_slides=math.floor((len(t)-nfft)/slide_size)+1
        n_freq=math.floor(nfft/2)+1

        print('nfft:', nfft, 'n_overlap:', n_overlap)
        print('slide_size:', slide_size,'n_slides:', n_slides, 'n_freq', n_freq)    
        print('specgram returns:', np.shape(spectrogram), np.shape(freqs), \
        np.shape(spec_t), np.shape(im))

        plt.sca(axs[2])
        plt.title("Spectrogram")
        ax = heatmap(spectrogram,  cmap="YlGnBu")
        ax.invert_yaxis()
        plt.title('Spectrogram, Heatmap')
        plt.xlabel('# frames.')
        plt.ylabel('# frequency steps')        
        
        plt.show()

    if np.shape(spectrogram)!=(Num_MFs, Num_frames):
        shape=np.shape(spectrogram)
        zero_pad=np.zeros((shape[0], Num_frames-shape[1]))
        spectrogram=np.concatenate((spectrogram, zero_pad), axis=1)
        
        #fig, axs = plt.subplots(1,1)
        #
        
        ax = heatmap(spectrogram,  cmap="YlGnBu")
        ax.invert_yaxis()
        plt.title("Spectrogram, zero-padded")
        plt.xlabel('# frames.')
        plt.ylabel('# frequency steps')        
        
        plt.show()
    return spectrogram
