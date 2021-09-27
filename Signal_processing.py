from matplotlib import pyplot as plt
import numpy as np
import sys
from seaborn import heatmap
from scipy.sparse import data

def stimuli_IMPACTmatrix_converter(data, NUM_MFs, TIME_STEP, PRNT=True):
    HALF_NUM=int(NUM_MFs/2)
    IMPACT_MATRIX=np.zeros((HALF_NUM, TIME_STEP))
    
    for ind_mf in range(HALF_NUM):
        IMPACT_MATRIX[ind_mf]=np.where(data==ind_mf,1,0)
    
    
    coordinate= np.argwhere(IMPACT_MATRIX).T

    i, j = coordinate
    if PRNT:
        print('shape impact mat.:', np.shape(IMPACT_MATRIX))

        print(np.flip(IMPACT_MATRIX, axis=0))
        print('i', i)
        print('j', j)
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
    print('MATRIX values:\n', IMPACT_MATRIX)
    print('MOVING_FREQ_MAT\n', MOVING_FREQ_MAT)
    if PRNT:        
        #print('shape.T sum:', np.shape(data.T))
        ax = heatmap(MOVING_FREQ_MAT,  cmap="YlGnBu")
        #ax.invert_yaxis()
        #bin_range = int((max(data)) - min(data))+1
        #x, bins, patch = plt.hist(data, bins=bin_range)
        #hist, edges = np.histogram(data, bins=bin_range)
        plt.title('Moving Frequency count distribution of IMPACT MATRIX')
        plt.xlabel('Time')
        plt.ylabel('Index')
        plt.show()

    #UNIQUE EXample
    #a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
    #unique, counts = numpy.unique(a, return_counts=True)
    #dict(zip(unique, counts))
    return MOVING_FREQ_MAT