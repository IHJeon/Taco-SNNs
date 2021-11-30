from Signal_processing import TIME_STEP
from degree_dist import data_load
import numpy as np
from matplotlib import pyplot as plt
import sys
MF_spike_trains=data_load('MF_spike_trains')

print('MF spikes shape:', np.shape(MF_spike_trains))

'''
x,y = np.argwhere(MF_spike_trains == 1).T
plt.scatter(y, x, marker='|', label='SPIKE arrivals')
plt.title('MF firing records')
#plt.gca().invert_yaxis()
plt.show()'''


#from Simplified_spiking_cells import Spiking_cells, simulator, plot_cell_dynamics
#/Changed to


NUM_MFs=np.shape(MF_spike_trains)[0]
NUM_GCs=NUM_MFs*2
K=4
DEGREE_MODULARITY=0
NUM_Rewiring=1
Rewiring_RATE=0.2
NUM_DOMAIN=2
DEGREE_IMPACT=88
#TIME_STEP=int(1e3) #10 sec = 10,000 ms
HALFNUM=int(NUM_MFs/2)
SR=1/1e-03


#MF_spike_counts=np.sum(MF_spike_trains, axis=1)
MF_freq_counts=np.sum(MF_spike_trains, axis=1)

#from SNN_Simplified_neurons import *  /Changed to
#from Network_info_processing import *
from Network_info_processing import SNN_connectivity
NETWORK_DRAW=False
my_SNN=SNN_connectivity(data_shape=NUM_MFs, num_node_per_layer=NUM_GCs, K=K, \
                        modularity=DEGREE_MODULARITY, NUM_modules=NUM_DOMAIN,\
                        num_rewire=NUM_Rewiring, rewire_rt=Rewiring_RATE, \
                        NETWORK_DRAW=NETWORK_DRAW, Spike_counts=MF_freq_counts)
feed_forward_indexing = my_SNN.feed_forward_indexing(MF_spike_trains.T)
#print(my_SNN.ed_list[:10])
#print('ED LIST shape:', np.shape(my_SNN.ed_list))
#sys.exit()
#my_SNN.show_edge_matrix()
#my_SNN.show_GCGC_edge_matrix()
print('feed_forward_indexing shape', np.shape(feed_forward_indexing))

TIME_STEP=np.shape(MF_spike_trains)[1]
WINDOW_INTERVAL=40
#Create GCs
from GC_processing import Spiking_cells, plot_cell_dynamics
GCs=[]
for i in range(NUM_GCs):
    #SPK_GC=Spiking_cells(num_dend=K)d
    GCs.append(Spiking_cells(num_dend=K, Recording_time=TIME_STEP))

Golgi_Cell=Spiking_cells(num_dend=NUM_GCs, Recording_time=TIME_STEP, TYPE_GOLGI=True)
GC_output_records=np.zeros((NUM_GCs, TIME_STEP))

for t in range(TIME_STEP):    
    GC_INHIBITION=Golgi_Cell.Spike_record[t-1] #-1 when 0, value is still 0
    for gc_ind, connected_mfs in enumerate(feed_forward_indexing):
        Connected_Spikes=[]
        for mf_ind in connected_mfs:            
            if t<WINDOW_INTERVAL:
                Connected_Spikes.append(MF_spike_trains[mf_ind][:t+1+WINDOW_INTERVAL])
                current_time_in_bin=t
            else:
                Connected_Spikes.append(MF_spike_trains[mf_ind][t-WINDOW_INTERVAL:t+1+WINDOW_INTERVAL])
                current_time_in_bin=WINDOW_INTERVAL
                
        #Connected_Spike_trains=np.array(Connected_Spike_trains).reshape(K, TIME_STEP)
        Connected_Spikes=np.array(Connected_Spikes).reshape(K, -1)        
        #print('Connected_Spikes to gc', gc_ind,\
        #    'from mfs', connected_mfs,'\n', Connected_Spikes)        
        GCs[gc_ind].simulator(Connected_Spikes, t, current_time_in_bin\
            , INH_Feedback=GC_INHIBITION)
        #print('np.shape(Spike_record)',np.shape(Spike_record))
    
    for gc_ind in range(NUM_GCs):
        GC_output_records[gc_ind][t]=GCs[gc_ind].Spike_record[t]
    
    if t<WINDOW_INTERVAL:
        Spikes_to_GoC=GC_output_records[:, :t+1+WINDOW_INTERVAL]
    else:
        Spikes_to_GoC=GC_output_records[:, t-WINDOW_INTERVAL:t+1+WINDOW_INTERVAL]
    Spikes_to_GoC=Spikes_to_GoC.reshape(NUM_GCs, -1)    
    Golgi_Cell.simulator(Spikes_to_GoC, t, current_time_in_bin)
        

#sys.exit()

        

print('GC firing records shape:', np.shape(GC_output_records))

x,y = np.argwhere(GC_output_records == 1).T
plt.scatter(y, x, marker='|', label='SPIKE arrivals')
plt.title('GC firing records')
plt.show()
print('---------------------------------------------')


