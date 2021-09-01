from SNN_Simplified_neurons import *
from Simplified_spiking_cells import Spiking_cells, simulator, plot_cell_dynamics
from seaborn import heatmap

NUM_MFs=100
NUM_GCs=NUM_MFs*3
K=10
TIME_STEP=int(1e3) #10 sec, 10,000 ms

data = data_generation(input_size=NUM_MFs, input_range=TIME_STEP).T


firing_MFs=[]
for ind, mf in enumerate(range(NUM_MFs)):
    firing_MFs.append(Rate_coded_Firing_cells(index=ind, activity_time_bin=1, time_unit=0.001, \
        spontaneous_freq= rand_ind(Low=1, max_val=10)))  #spontaneous_freq must be greater than 1 for poisson calculation


MF_spike_trains=[]
for ind, mf in enumerate(firing_MFs):
    single_spike_train = mf.stimulation(data[ind])    
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
plt.show()'''


x,y = np.argwhere(MF_spike_trains == 1).T
plt.scatter(y, x, marker='|', label='SPIKE arrivals')
plt.title('MF firing records')
plt.show()
print('---------------------------------------------')

GCs=[]
for i in range(NUM_GCs):
    SPK_GC=Spiking_cells(num_dend=K)
    GCs.append(SPK_GC)

my_SNN=SNN_connectivity(data_shape=NUM_MFs, num_node_per_layer=NUM_GCs, K=K)
feed_forward_indexing = my_SNN.feed_forward_indexing(MF_spike_trains.T)


GC_output_records=[]

RECORD_WEIGHT_CHANGE=True
for f_ind, ff in enumerate(feed_forward_indexing):
    Spike_trains=[]
    for f in ff:
        Spike_trains.append(MF_spike_trains[f])            
    spike_pttn_per_bin=np.array(Spike_trains).reshape(K, TIME_STEP)    
    #_, e, i, m, _, _, Num_OUTPUT_SPIKE, Spike_record = simulator(NUM_MFs, spike_pttn_per_bin, TIME_STEP, GCs[f_ind])
    _, e, i, m, _, Num_OUTPUT_SPIKE, Spike_record = \
        simulator(NUM_MFs, spike_pttn_per_bin, TIME_STEP, GCs[f_ind], RECORD_WEIGHT_CHANGE)
    #_, e, m, _, Num_OUTPUT_SPIKE, Spike_record = simulator(NUM_MFs, spike_pttn_per_bin, TIME_STEP, GCs[f_ind])
    if f_ind==0: 
        plot_cell_dynamics(K, TIME_STEP, e, i, m )
        RECORD_WEIGHT_CHANGE=False
    #if f_ind==0: plot_cell_dynamics(K, TIME_STEP, e, m )
    #if Num_OUTPUT_SPIKE>0: print('Num outspike', Num_OUTPUT_SPIKE)
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



WEIGHT_MATRIX=np.ones((NUM_GCs, K))
for ind, gc in enumerate(GCs):
    WEIGHT_MATRIX[ind]=gc.Synaptic_strenth

ax = heatmap(WEIGHT_MATRIX,  cmap="YlGnBu")
plt.title('Weight Matrix, Heatmap')
plt.xlabel('K, num dend.')
plt.ylabel('Num GCs')
plt.show()