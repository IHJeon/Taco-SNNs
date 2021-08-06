#from Uniform_binary_net import *
from RateCodingModel import *
import numpy as np
import sys
from matplotlib import pyplot as plt
import seaborn as sns
#np.core.arrayprint._line_width = 1000
np.set_printoptions(linewidth=350)

data = [[0, 1, 0],
        [1, 1, 0]]
data=np.array(data)

#my_ubn=UBN(data_shape=data[1:].shape, num_layer= num_layer, num_node_per_layer=3)

#network_output = my_ubn.feed_forward_dense(data)

data_name='pref_4_MFs_edges'
#data_name='comp_120parentsMF_eq:0'

node_GC, node_MF, ed_list= input_processing(data_name) #10 or 30)
NUM_MFs=len(node_MF)
NUM_GCs=len(node_GC)

#NUM_MFs=8
#num_GC=NUM_MFs*3

NUM_DATA_POINTs=1000

#firing_rate=0.142
stimuli_signal_rate=0.1
mf_group='none_specified' # mf_group = 'e', or 'm' or 'l' or 'none_specified'
NUM_MFs=200
GC_expansion_ratio=1  #M/N
synaptic_degree=4 #K
#NUM_MODULES=3
#NETWORK_TYPE='random_separate'
NETWORK_TYPE='modular'
#NETWORK_TYPE='random'
#INPUT_TYPE='random'
INPUT_TYPE='Sinosoidal'
CONTINUITY_INPUT=True
#NUM_GCs=2

#print('Input data type:',type(data),'\n', data )

if INPUT_TYPE=='Sinosoidal':
    data = data_generation(input_size=NUM_MFs, num_input=NUM_DATA_POINTs, gen_mode='Sinosoidal', \
        firing_rate=stimuli_signal_rate, continuity=CONTINUITY_INPUT)
    print('data shape', np.shape(data))
    #sys.exit()

elif INPUT_TYPE=='random':
    data = data_generation(input_size=NUM_MFs, num_input=NUM_DATA_POINTs, gen_mode='rate_fixed', \
    #data = data_generation(input_size=NUM_MFs, num_input=NUM_DATA_POINTs, gen_mode='random', \
    ##data = data_generation(input_size=NUM_MFs, num_input=NUM_DATA_POINTs, gen_mode=['101'], \
    ##data = data_generation(input_size=NUM_MFs, num_input=NUM_DATA_POINTs, gen_mode=['01','001', '0001'], \
    #            #firing_rate=firing_rate)    # mf_group = 'e', or 'm' or 'l'
                firing_rate=stimuli_signal_rate, continuity=CONTINUITY_INPUT)   

elif INPUT_TYPE=='random_separate':
        data2 = data_generation(input_size=NUM_MFs, num_input=NUM_DATA_POINTs, gen_mode='rate_fixed', \
            firing_rate=stimuli_signal_rate)
        data=np.concatenate([data, data2], axis=1)

        NUM_MFs*=2

elif INPUT_TYPE=='modular':
    num_modules=3
    module_sizes=[int(NUM_MFs/num_modules)]*(num_modules-1)+[NUM_MFs-int(NUM_MFs/num_modules)*(num_modules-1)]
    modular_points=[int(NUM_DATA_POINTs/num_modules)]*(num_modules-1)+[NUM_DATA_POINTs-int(NUM_DATA_POINTs/num_modules)*(num_modules-1)]

    print('module_sizes', module_sizes, 'modular_points', modular_points)
    #data=[]
    zero_padding_map=np.zeros((NUM_DATA_POINTs, NUM_MFs))
    for ind, data in enumerate(range(num_modules)):
        modular_data = data_generation(input_size=module_sizes[ind], num_input=modular_points[ind], gen_mode='rate_fixed', \
            firing_rate=stimuli_signal_rate, continuity=CONTINUITY_INPUT)       

        coord_x, coord_y= np.sum(modular_points[:ind]).astype(int), np.sum(module_sizes[:ind]).astype(int)
        
        #print(np.shape(zero_padding_map[coord_x:, coord_y:]))
        #print('To Coordinate: [', coord_x, coord_y, ']', ind,'-th modular_data shape', np.shape(modular_data))
        zero_padding_map[coord_x:coord_x+modular_data.shape[0], coord_y:coord_y+modular_data.shape[1]]+=modular_data

        #data.append(modular_data)
    data=zero_padding_map

    #data=np.concatenate(data, axis=1)

#print('data shape:', np.shape(data))
#print('Stimuli data: \n', data.T)
#sys.exit()


'''data plotting'''

if CONTINUITY_INPUT:
    #plt.imshow(data, cmap='hot', interpolation='nearest')
    #ax = sns.heatmap(data.T, linewidth=0.5)
    #ax = sns.heatmap(data.T,  cmap="YlGnBu")
    #plt.xlabel('Time steps')
    #plt.yticks('Stimuli')
    #
    #plt.show()
    raster_plot(data, [0], heatmap=True) 
else: raster_plot(data, [0])  #[0] -> Flag data

#sys.exit()

firing_MFs=[]
for ind, mf in enumerate(range(NUM_MFs)):
    firing_MFs.append(Firing_cells(index=ind, activity_time_bin=1, time_unit=0.001, continuity=CONTINUITY_INPUT))

#time_bin=10
#data_trans=data.T
#for ind in range(NUM_MFs):
#    data=stimuli_impact_wave(data_trans[ind])
#sys.exit()

MF_spike_trains=[]
data_trans=data.T
for ind, mf in enumerate(firing_MFs):
#for mf in firing_MFs:
    #print('data_trans[ind]:',len(data_trans[ind]), 'len data', len(data_trans), 'Num MF:', len(firing_MFs))
    #sys.exit()
    single_spike_train = mf.stimulation(data_trans[ind])    
    #single_spike_train = mf.wide_stimulation(data) # by taking into account of collateral effect
    MF_spike_trains.append(single_spike_train)
    #print('data', np.shape(data), 'single spike', np.shape(single_spike_train))
    #sys.exit()
    #arr = np.array(single_spike_train).T
    #print('single_spike_train shape', np.shape(arr),'len', len(arr))        
    #print('mf', ind, 'data:', data_trans[ind],' single_spike_train', np.shape(single_spike_train)) #np.shape(data_trans[ind]))

MF_spike_trains=np.flip(MF_spike_trains, 0)


#MF_spike_trains_plot=np.array(MF_spike_trains).T
MF_spike_trains=np.array(MF_spike_trains).T
#MF_spike_trains=np.array(MF_spike_trains)

#print('Shape MF trains', np.shape(MF_spike_trains), '\n', MF_spike_trains)


raster_plot(MF_spike_trains, [1])

sum_list=[]
print("Shape spike trains:", np.shape(MF_spike_trains))
for mf in MF_spike_trains.T:
    sum_list.append(np.sum(mf))
print('MF Spikes Total SUM freq. dist.:', sum_list, \
    'Mean:', np.sum(MF_spike_trains)/NUM_MFs, 'Standard deviation:', np.std(sum_list) )

#sys.exit()


'''--------------- plot firing intensities -----------------'''

collective_firing_intensities=[]
for ind, spike_train in enumerate(MF_spike_trains.T):
    individual_firing_intensities=[]
    #print('firing_MFs[ind].activity_time_bin', firing_MFs[ind].activity_time_bin)
    #sys.exit()
    for time, stm in enumerate(spike_train):
        #at_scope=spike_train[time:time+firing_MFs[ind].activity_time_bin]
        at_scope=spike_train[time:time+100]
        #print('at time:',time, 'len scope', len(at_scope))
        rate_at_time = np.sum(at_scope)
        individual_firing_intensities.append(rate_at_time)
    collective_firing_intensities.append(individual_firing_intensities)


collective_firing_intensities=np.array(collective_firing_intensities)
collective_firing_intensities=np.flip(collective_firing_intensities, 0)

#print('shape intensities', np.shape(collective_firing_intensities), 'shape MF_spike_trains.T', np.shape(MF_spike_trains.T))
ax = sns.heatmap(collective_firing_intensities,  cmap="YlGnBu")
plt.title('collective_firing_intensities, Heatmap')
plt.xlabel('Time steps')
plt.ylabel('Firing intensity')

plt.show()

time_sum_of_firing_intensities=[]
for at_time in collective_firing_intensities.T:
    time_sum=np.sum(at_time)
    time_sum_of_firing_intensities.append(time_sum)
time_sum_of_firing_intensities=np.array(time_sum_of_firing_intensities)

heatmap_plot_time_sum_of_firing_intensities=np.array(time_sum_of_firing_intensities).reshape((1, len(time_sum_of_firing_intensities)))
print('shape time_sum_of_firing_intensities', np.shape(time_sum_of_firing_intensities))
ax = sns.heatmap(heatmap_plot_time_sum_of_firing_intensities/NUM_MFs,  cmap="YlGnBu")
plt.title('time_sum_of_firing_intensities/NUM_MFs, Heatmap')
plt.xlabel('Time steps')
plt.ylabel('Time Sum of intensity')
plt.show()


ax = plt.plot(time_sum_of_firing_intensities/NUM_MFs)
plt.title('time_sum_of_firing_intensities/NUM_MFs, Histogram')
plt.xlabel('Time steps')
plt.ylabel('Time Sum of intensity')
plt.show()

'''--------------------------------'''
sys.exit()

#my_ubn=UBN(data_shape=NUM_MFs, num_layer= 1, num_node_per_layer=num_GC, network=data_name, net_drawing=False)

#my_ubn=UBN(data_shape=NUM_MFs, num_layer= 1, num_node_per_layer=num_GC, network='random', net_drawing=True)

my_ubn=UBN(data_shape=NUM_MFs, num_layer= 1, num_node_per_layer=NUM_MFs*GC_expansion_ratio, network=NETWORK_TYPE,\
#my_ubn=UBN(data_shape=NUM_MFs, num_layer= 1, num_node_per_layer=NUM_GCs, network='random',\
         net_drawing=False, rewire_rt=0.2, num_rewire=2, K=synaptic_degree, num_modules=5)

edge_matrix_visualization(NUM_MFs, NUM_MFs*GC_expansion_ratio, my_ubn.ed_list)

REORDER=True
GRAPH=True

edge_matrix_visualization(NUM_MFs, NUM_MFs*GC_expansion_ratio, my_ubn.ed_list, GRAPH=GRAPH)
edge_matrix_visualization(NUM_MFs, NUM_MFs*GC_expansion_ratio, my_ubn.ed_list, GRAPH=GRAPH, REORDER=REORDER)
edge_matrix_visualization(NUM_MFs, NUM_MFs*GC_expansion_ratio, my_ubn.ed_list, GRAPH=GRAPH, REORDER=REORDER, symmetric=True)
sys.exit()
num_synapse=len(my_ubn.ed_list)
synapse_weight_list=[]
for ind, syn in enumerate(range(num_synapse)):
        synapse_weight_list.append(np.random.uniform(low=0,high=1))
#synapse_list=[]
#for ind, syn in enumerate(num_synapse):
#        synapse_list.append(synapse(ind))

#network_output = my_ubn.feed_forward_gcmf(data, threshold=2, inhibition=False, print_process=False)
import timeit
start = timeit.timeit()

network_output = my_ubn.feed_forward_feq_code(MF_spike_trains, weight_list=synapse_weight_list,\
                         threshold=3, inhibition=True, print_process=False)
end = timeit.timeit()
print('time elapsed for feed forwarding', end - start)

network_output_no_inhib = my_ubn.feed_forward_feq_code(MF_spike_trains, weight_list=synapse_weight_list,\
                         threshold=3, inhibition=False, print_process=False)
#print('network_output:', np.shape(network_output))

#firing_GCs=[]
#for ind, gc in enumerate(range(NUM_GCs)):
#        firing_GCs.append(Firing_cells(index=ind))
#
#
#GC_spike_trains=[]

#for ind, gc in enumerate(firing_GCs):        
#        single_spike_train = gc.stimulation(MF_spike_trains[ind])
#        GC_spike_trains.append(single_spike_train)



#raster_plot(data, [0])
#print('MF output plotting...')
#raster_plot(MF_spike_trains, [1])
print('GC output plotting...')
raster_plot(network_output, [], partition=2, print_ratio=True, gc_color_map=my_ubn.gc_color_map)
raster_plot(network_output_no_inhib, [], partition=2, print_ratio=True, gc_color_map=my_ubn.gc_color_map)
#sys.exit()
print('GC output plotting... color sorted')

GC_color_map_np=np.array(my_ubn.gc_color_map)

color_sort_indices=[np.where(GC_color_map_np=='red')[0]
                   ,np.where(GC_color_map_np==greeny_red)[0]
                   ,np.where(GC_color_map_np==mix_redgreen)[0]
                   ,np.where(GC_color_map_np==redish_green)[0]
                   ,np.where(GC_color_map_np=='green')[0] ]

print('ind color', len(color_sort_indices[0])\
                 , len(color_sort_indices[1])\
                 , len(color_sort_indices[2])\
                 , len(color_sort_indices[3])\
                 , len(color_sort_indices[4]))
sys.exit()
sorted_color_map=np.ndarray.tolist(np.take(GC_color_map_np, color_sort_indices[0]))\
                +np.ndarray.tolist(np.take(GC_color_map_np, color_sort_indices[1]))\
                +np.ndarray.tolist(np.take(GC_color_map_np, color_sort_indices[2]))\
                +np.ndarray.tolist(np.take(GC_color_map_np, color_sort_indices[3]))\
                +np.ndarray.tolist(np.take(GC_color_map_np, color_sort_indices[4]))

sorted_data=np.concatenate( (
             np.take(network_output, color_sort_indices[0], axis=1)\
            ,np.take(network_output, color_sort_indices[1], axis=1)\
            ,np.take(network_output, color_sort_indices[2], axis=1)\
            ,np.take(network_output, color_sort_indices[3], axis=1)\
            ,np.take(network_output, color_sort_indices[4], axis=1)  ), axis=1 )

#for ind, elem in enumerate(color_sort_indices):
#    print('color sort indices', ind, '-th lenght:', len(elem))
#print('my_ubn.gc_color_map', np.shape(GC_color_map_np), 'sorted_color_map', np.shape(sorted_color_map))
#print('network_output', np.shape(network_output), 'sorted_data', np.shape(sorted_data))


raster_plot(sorted_data, [], partition=2, print_ratio=True, gc_color_map=sorted_color_map)


