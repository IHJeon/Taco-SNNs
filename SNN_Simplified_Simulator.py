#from Spiking_Single_cells_multiple_dend import TIME_RANGE
from scipy.sparse import data
from SNN_Simplified_neurons import *
from Simplified_spiking_cells import Spiking_cells, simulator, plot_cell_dynamics
from seaborn import heatmap
from Signal_processing import *
NUM_MFs=100
NUM_GCs=NUM_MFs*3
K=4
TIME_STEP=int(1e3) #10 sec, 10,000 ms
DEGREE_MODULARITY=100
NUM_DOMAIN=2
#data = data_generation(input_size=NUM_MFs, input_range=TIME_STEP, mode='2').T
#data1 = data_generation(input_size=int(NUM_MFs/3), input_range=TIME_STEP,       mode='1').T
#data2 = data_generation(input_size=int(NUM_MFs/3), input_range=TIME_STEP,       mode='2').T
#data3 = data_generation(input_size=NUM_MFs-2*len(data1), input_range=TIME_STEP, mode='3').T
#data=np.vstack((data1,data2, data3))
#print('Data lenth-  1:', len(data1),'2:', len(data2), '3:', len(data3))
data=[]
data.append(data_generation(input_size=1, input_range=TIME_STEP, mode='3').T)
data.append(data_generation(input_size=1, input_range=TIME_STEP, mode='2').T)
#data.append(data_generation(input_size=1, input_range=TIME_STEP, mode='3').T)
data=(np.array(data)*NUM_MFs/2/100).astype(int).reshape((NUM_DOMAIN,-1))
#print('data shape:', np.shape(data), 'lenth', len(data))
#print(data[0])

#sys.exit()

#fig, axs = plt.subplots(NUM_MFs)#, gridspec_kw={'height_ratios': [1]*(NUM_MFs)})
fig, axs = plt.subplots(len(data))#, gridspec_kw={'height_ratios': [1]*(NUM_MFs)})
#for ind_mf in range(NUM_MFs):
for ind_mf in range(len(data)):
    #INDEX=ind_mf*int(NUM_MFs/3)
    INDEX=ind_mf
    print('INDEX', INDEX)
    axs[ind_mf].plot(np.arange(TIME_STEP), data[INDEX])
    axs[ind_mf].axhline(y=NUM_MFs/2, color='c', linestyle=':', label='Min')
    axs[ind_mf].axhline(y=0, color='c', linestyle=':', label='Max')
axs[0].set_title('Stimuli input')
plt.show()

IMPACT_MATRIX, COORDINATE=stimuli_IMPACTmatrix_converter(data[0], NUM_MFs, TIME_STEP)
#FREQUENCY = FREQUENCY_COUNTER(IMPACT_MATRIX)
FREQUENCY = MOVING_FREQUENCY_MATRIX(IMPACT_MATRIX)

firing_MFs=[]
for ind, mf in enumerate(range(NUM_MFs)):
    firing_MFs.append(Rate_coded_Firing_cells(index=ind, activity_time_bin=1, time_unit=0.001, \
        spontaneous_freq= rand_ind(Low=1, max_val=10)))  #spontaneous_freq must be greater than 1 for poisson calculation
#firing_MFs2=[]
#for ind, mf in enumerate(range(NUM_MFs)):
#    firing_MFs2.append(Rate_coded_Firing_cells(index=ind, activity_time_bin=1, time_unit=0.001, \
#        spontaneous_freq= rand_ind(Low=1, max_val=10)))  #spontaneous_freq must be greater than 1 for poisson calculation

'''
MF_spike_trains=[]
for ind, mf in enumerate(firing_MFs):
    single_spike_train = mf.stimulation(data[ind])
    #single_spike_train = mf.wide_stimulation(data) # by taking into account of collateral effect
    MF_spike_trains.append(single_spike_train)'''


MF_spike_trains=[]

HALFNUM=int(NUM_MFs/2)
print(HALFNUM)
for ind in range(HALFNUM):
    single_spike_train = firing_MFs[ind].stimulation(IMPACT_MATRIX[ind])
    #single_spike_train = mf.wide_stimulation(data) # by taking into account of collateral effect
    MF_spike_trains.append(single_spike_train)
for ind in range(HALFNUM):
    single_spike_train = firing_MFs[ind+HALFNUM].stimulation(FREQUENCY[ind])
    #single_spike_train = mf.wide_stimulation(data) # by taking into account of collateral effect
    MF_spike_trains.append(single_spike_train)

MF_spike_trains=np.flip(MF_spike_trains, 0)

'''Phase 1'''

print('-----------STIMULI + MF FIRING DATA-------------')
print('Input signal shape:', np.shape(data))
#print('Input signal shape[0]:', np.shape(data[0]))
print('MF Output shape:', np.shape(MF_spike_trains))
#print('GC Output shape:', np.shape(data))




'''
fig, axs = plt.subplots(NUM_MFs)#, gridspec_kw={'height_ratios': [1]*(NUM_MFs)})
for ind_mf in range(NUM_MFs):
    axs[ind_mf].plot(np.arange(TIME_STEP), data[ind_mf])
    axs[ind_mf].axhline(y=100, color='c', linestyle=':', label='Min')
    axs[ind_mf].axhline(y=0, color='c', linestyle=':', label='Max')
axs[0].set_title('Stimuli input')
plt.show()
'''


x,y = np.argwhere(MF_spike_trains == 1).T
plt.scatter(y, x, marker='|', label='SPIKE arrivals')
plt.title('MF firing records')
plt.show()
print('---------------------------------------------')
#sys.exit()
GCs=[]
for i in range(NUM_GCs):
    #SPK_GC=Spiking_cells(num_dend=K)
    GCs.append(Spiking_cells(num_dend=K))

#print('GC spec', len(GCs[0].p_EXT_U), len(GCs[0].p_EXT_R), len(GCs[0].STDP_weight)\
#            ,GCs[0].num_dend, GCs[0].membrane_potential, GCs[0].MEM_POT_NOSTP)


my_SNN=SNN_connectivity(data_shape=NUM_MFs, num_node_per_layer=NUM_GCs, K=K, \
                        modularity=DEGREE_MODULARITY)
feed_forward_indexing = my_SNN.feed_forward_indexing(MF_spike_trains.T)
my_SNN.show_edge_matrix()

print(np.shape(feed_forward_indexing))
#print('feed_forward_indexing', feed_forward_indexing)


GC_output_records=[]

RECORD_WEIGHT_CHANGE=False
for f_ind, ff in enumerate(feed_forward_indexing):
    Connected_Spike_trains=[]
    for f in ff:
        Connected_Spike_trains.append(MF_spike_trains[f])            
    Connected_Spike_trains=np.array(Connected_Spike_trains).reshape(K, TIME_STEP)

    #_, e, i, m, _, _,  Spike_record = simulator(NUM_MFs, Connected_Spike_trains, TIME_STEP, GCs[f_ind])
    _, e, i, m, _,  Spike_record = \
        simulator(NUM_MFs, Connected_Spike_trains, TIME_STEP, GCs[f_ind], RECORD_WEIGHT_CHANGE)
    #_, e, m, _,  Spike_record = simulator(NUM_MFs, Connected_Spike_trains, TIME_STEP, GCs[f_ind])
    if f_ind==0 and RECORD_WEIGHT_CHANGE: 
        plot_cell_dynamics(K, TIME_STEP, e, i, m )
        RECORD_WEIGHT_CHANGE=False
    #if np.sum(Spike_record)>0: print('Num outspike', np.sum(Spike_record))
    GC_output_records.append(Spike_record)
    

GC_output_records=np.array(GC_output_records)
'''Phase 2'''
print('-----------Feed forwarding connectivity/ Output-------------')
print('feed forward indexing shape:', np.shape(feed_forward_indexing))
print('GC firing records shape:', np.shape(GC_output_records))

x,y = np.argwhere(GC_output_records == 1).T
plt.scatter(y, x, marker='|', label='SPIKE arrivals')
plt.title('GC firing records')
plt.show()
print('---------------------------------------------')



WEIGHT_MATRIX=np.zeros((NUM_GCs, K))
for ind, gc in enumerate(GCs):
    #print('ind:', ind, 'val:', np.around(gc.STDP_weight,2))
    WEIGHT_MATRIX[ind]=gc.STDP_weight

avg=np.mean(WEIGHT_MATRIX, axis=0)
print('AVG each columne:', np.shape(avg), np.around(avg, 3))
ax = heatmap(WEIGHT_MATRIX,  cmap="YlGnBu")
ax.invert_yaxis()
plt.title('STDP Weight Matrix, Heatmap')
plt.xlabel('K, num dend.')
plt.ylabel('Num GCs')
plt.show()
