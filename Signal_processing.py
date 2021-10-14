from matplotlib import pyplot as plt
import numpy as np
import sys
from seaborn import heatmap
#from scipy.sparse import data
from scipy.fft import fft, fftfreq

def rand_ind(Low=0, max_val=100):
        return np.random.randint(0,max_val)
def rand_normal():        
    return np.random.normal(scale=0.05)        

def data_generation(input_size, input_range, mode='random'):
    data = []    
    max_level_stimuli=100
    
    from sklearn.preprocessing import MinMaxScaler        
    
    x= np.linspace(0,input_range,input_range)    
    #seed_val=np.random.randint(0,100)
    
    for inpt in range(input_size):        
        if mode=='random': 
            y=rand_ind()*np.sin(rand_normal()*x)+rand_ind()*np.cos(rand_normal()*x)
        elif mode=='1': 
            dt=0.001
            TIME_STEP=input_range
            t=np.arange(0, TIME_STEP*dt, dt)
            y=np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)
            #y=np.sin(x*4*np.pi/input_range)*max_level_stimuli/2+50
            dt=0.001
            f0=50
            f1=250
            t1=1
            t=np.arange(0,t1,dt)
            y= np.cos(2*np.pi*t*
                    (f0+(f1-f0)*np.power(t,2))/(3*t1**2)
                    )
        elif mode=='2': y=np.cos(0.05*x)*max_level_stimuli
        #elif mode=='3': y=np.flip(x)/input_range*max_level_stimuli*2/3
        elif mode=='3': y=x/input_range*max_level_stimuli*2/3
        elif mode=='4': y=[max_level_stimuli]*int(len(x)/2)+[0]*(len(x)-int(len(x)/2))
        elif mode=='5': y=[0]*(len(x)-int(len(x)/2))+[max_level_stimuli]*int(len(x)/2)
        elif mode=='6': y=len(x)*[max_level_stimuli*0.1]
        y=MinMaxScaler(feature_range=(0,100)).fit_transform(y.reshape(-1, 1)).reshape(-1)
        #y=MinMaxScaler(feature_range=(0,max_level_stimuli)).fit_transform(y.reshape(-1, 1)).reshape(-1)                           
        #y=abs(y)
        y=np.clip(y, 0, max_level_stimuli)
        data.append(y)
        #plt.plot(x,y)
        #plt.xlabel('Time steps')
        #plt.ylabel('Signal intensity')
        #plt.show()
    #print('len y:',len(y), 'Sum y:', np.sum(y), 'AVG y', np.sum(y)/len(y))
    #print('Signal Data shape:',np.shape(data), 'Sum data:', np.sum(data), 'AVG Data', np.sum(data)/(input_size*num_input) )
    #print('y',y)
    data=np.array(data).T
    return data

def stimuli_IMPACTmatrix_converter(data, NUM_MFs, TIME_STEP, PRNT=True):
    #HALF_NUM=int(NUM_MFs/2)
    HALF_NUM=int(NUM_MFs)
    IMPACT_MATRIX=np.zeros((HALF_NUM, TIME_STEP))
    
    for ind_mf in range(HALF_NUM):
        IMPACT_MATRIX[ind_mf]=np.where(data==ind_mf,1,0)
    
    
    coordinate= np.argwhere(IMPACT_MATRIX).T

    i, j = coordinate
    if PRNT:
        print('shape impact mat.:', np.shape(IMPACT_MATRIX))

        print(np.flip(IMPACT_MATRIX, axis=0))
        #print('i', i)
        #print('j', j)
        plt.scatter(j, i, marker='|', label='STIMULI dist')    
        plt.title('PRIMARY POINTs IMPACT MATRIX')
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


def spectrogram_EX():
    dt=0.001
    f0=50
    f1=250
    t1=1
    t=np.arange(0,t1,dt)
    x= np.cos(2*np.pi*t*
            (f0+(f1-f0)*np.power(t,2))/(3*t1**2)
            )
    #dt=0.001
    #TIME_STEP=int(1e3)
    #t=np.arange(0, TIME_STEP*dt, dt)
    #x=np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)

    fig, axs = plt.subplots(2,1)
    plt.sca(axs[0])
    plt.title("Signals")
    plt.plot(t,x)


    plt.sca(axs[1])
    plt.title("Spectrogram")
    plt.specgram(x, NFFT=128, Fs=1/dt, noverlap=120)

    plt.show()

    import sounddevice as sd
    sd.play(1*x, 1/dt)
    sd.wait()

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



def Power_spectrum_at_t(f, TIME_UNIT=1e-3, PRINT=False):
    
    if not len(f)==0:        
        dt=TIME_UNIT
        t=np.arange(0, len(f)*TIME_UNIT, dt)        
    else:
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
    #print(len(PSD[L]))
    #print(freq[np.where(PSD[L]>1)],  PSD[L][np.where(PSD[L]>1)])
    Freq_components = freq[np.where(PSD[L]>1)]
    Freq_amplitude = PSD[L][np.where(PSD[L]>1)]
    

    if PRINT:
        fig, axs = plt.subplots(2,1)
        plt.sca(axs[0])    
        plt.plot(t,f, label='Signal')
        plt.xlim(t[0], t[-1])
        plt.legend()
        plt.title("Signals")

        plt.sca(axs[1])
        #plt.plot(freq[L], abs(fhat[L]), label='Noisy abs')
        plt.plot(freq[L],PSD[L], label='PSD')
        #plt.plot(freq[L],fhat[L],  label='Noisy fhat')

        plt.xlim(freq[L[0]],freq[L[-1]])
        plt.legend()
        plt.title("power Spectrum")

        plt.show()    
    return Freq_components,  Freq_amplitude
    
    
def FREQ_MAP(f, TIME_STEP, NUMCELL, PRINT=True):
    freq_map=np.zeros((TIME_STEP, NUMCELL))   #Freq range = (0 ~ 1/dt *0.5); max=500
    print('freq map shape:', np.shape(freq_map))
    print('freq map shape:', np.shape(freq_map[0]))
    #time_bin=int(len(f)/10)
    coverage_ratio=int(500/NUMCELL) 
    for i in range(TIME_STEP):    
        comp, amp = Power_spectrum_at_t(f[i:i+100])
        #print('comp, amp shape:', np.shape(comp), np.shape(amp))
        if not len(comp)==0:
            ind=(comp/coverage_ratio).astype(int) #Freq bin per cell = coverage_ratio
            cell_input=np.zeros(NUMCELL)
            for freq_ind, freq_comp in enumerate(comp):
                cell_input[int(freq_comp/coverage_ratio)]+=amp[freq_ind]               

            freq_map[i]+=cell_input

    #print(np.where(freq_map>0))
    if PRINT:
        ax = heatmap(freq_map.T,  cmap="YlGnBu")
        ax.invert_yaxis()
        plt.title('Freq Matrix, Heatmap')
        plt.xlabel('Time step')
        plt.ylabel('Freq input to Cell index')
        plt.show()
    return freq_map.T



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

    
