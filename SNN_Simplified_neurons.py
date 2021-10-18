#from Spiking_Single_cells_multiple_dend import TIME_RANGE
#from RateCode_simulator import NUM_MFs
from Simplified_spiking_cells import *
#from RateCodingModel import *
from matplotlib import pyplot as plt

def softmax(valueset): 
    #return np.exp(valueset)/np.exp(valueset).sum()
    return valueset/valueset.sum() # no Exponential

def Pruning(WEIGHT_MATRIX, NUM_PRUN=1):    
    Row=np.shape(WEIGHT_MATRIX)[0]
    Col=np.shape(WEIGHT_MATRIX)[1]-NUM_PRUN
    Pruned_MAT=np.zeros((Row, Col))
    for r in Row:
        mat=WEIGHT_MATRIX[r]
        ind_prun=np.where(mat==np.amin(mat))
        pruned=np.delete(mat, mat[ind_prun[0]])
        Pruned_MAT[r]=pruned
    
    return Pruned_MAT




def random_networks(Num_GCs, Num_MFs, previous_MF_index=0, degree_connectivity=4, degree_of_modularity=0):    
    expansion_rate=Num_GCs/Num_MFs
    node_GC=[]
    node_MF=[]
    previous_GC_index=int(previous_MF_index*expansion_rate)
    for gc in range(Num_GCs):        
        node_GC.append('G%s'%(gc+previous_GC_index))
    for mf in range(Num_MFs):        
        node_MF.append('M%s'%(mf+previous_MF_index))
    
    ed_list=[]
    for i in range(Num_GCs):
        if degree_of_modularity>0: #Syn. partner selection for modularity, preferential to close indices            
            prob_basis_val=np.exp(-1*(abs((i/expansion_rate-np.arange(Num_MFs))*degree_of_modularity/100)))            
            modular_prob_assign = softmax(prob_basis_val)            
            synapse_partner=np.random.choice(Num_MFs, size=degree_connectivity, replace=False, p=modular_prob_assign)
            #print('Prob', np.around(modular_prob_assign,2),'Choice', synapse_partner)
            #sys.exit()
        else: synapse_partner=np.random.choice(Num_MFs, size=degree_connectivity, replace=False)

        syns=sorted(synapse_partner)
        #syns=synapse_partner
        for j in syns:            
            ed_list.append(['M%s'%(j+previous_MF_index), 'G%s'%(i+previous_GC_index)])

    print('Network stats: # GCs', len(node_GC), '# MFs', len(node_MF),'# edges(synapses)', len(ed_list))
    #sys.exit()
    
    return node_GC, node_MF, ed_list

class SNN_connectivity: # Spiking Neural networks
    def __init__(self, data_shape, num_node_per_layer, net_drawing=False, K=4, modularity=0):
        self.input_shape=data_shape
        self.num_node_per_layer=num_node_per_layer
        self.node_GC, self.node_MF, self.ed_list = random_networks(self.num_node_per_layer,\
                    self.input_shape, degree_connectivity=K, degree_of_modularity=modularity)


    def feed_forward_indexing(self, data):
        FEEDFORWARD_INDICES=[]
        for gc in self.node_GC:
            single_node_output=0
            GC_INPUT_INDICES=[]
            for edge_ind, edge in [(eg_ind, eg) for eg_ind, eg in enumerate(self.ed_list) if eg[1]==gc]:
                target_MF=edge[0]
                MF_INDEX=self.node_MF.index(target_MF)
                GC_INPUT_INDICES.append(MF_INDEX)            
            FEEDFORWARD_INDICES.append(GC_INPUT_INDICES)
        return FEEDFORWARD_INDICES

    def show_edge_matrix(self, GRAPH=False):
        nd_GC, nd_MF = len(self.node_GC), len(self.node_MF)
        pallet = np.zeros((nd_GC, nd_MF))
        #print('pallet shape', np.shape(pallet))
        #print('pallet init val\n', pallet)        

        for ed in self.ed_list:
            coord_y, coord_x = int(ed[0][1:]), int(ed[1][1:])
            pallet[coord_x, coord_y]+=1

        if GRAPH:
            pallet2 = np.zeros((nd_MF, nd_GC))
            for ed in self.ed_list:            
                coord_x, coord_y = int(ed[0][1:]), int(ed[1][1:])
                pallet2[coord_x, coord_y]+=1
    
            inter_GC_synapses=np.zeros((nd_GC,nd_GC))
            inter_MF_synapses=np.zeros((nd_MF,nd_MF))
    
            print('pallet1 shape', np.shape(pallet))
            print('pallet2 shape', np.shape(pallet2))
    
            pallet2 = np.concatenate((inter_GC_synapses, pallet2), axis=0)
            pallet = np.concatenate((pallet, inter_MF_synapses), axis=0)
            pallet = np.concatenate((pallet2, pallet), axis=1)

        fig, ax = plt.subplots()

        if GRAPH: plt.title('Graph Weight Connectivity matrix')
        else: 
            plt.title('Weight Connectivity matrix')
            plt.xlabel('MFs')
            plt.ylabel('GCs')
        ax.matshow(pallet, cmap=plt.cm.Blues)
        ax.invert_yaxis()
        plt.show()
        print("Total Num Weight", np.sum(pallet))




def poisson_spike_generator(lamda, time_bin=100, time_unit=0.001):        # poisson spike per time bin    
    beta=1/lamda    
    spikes_record=[]
    while len(spikes_record)!=time_bin:        
        delta_t = np.random.exponential(beta)/time_unit
        incoming_spike = [0]*int(delta_t)+[1]
        spikes_record+=incoming_spike
        if len(spikes_record)>time_bin: spikes_record=spikes_record[:time_bin]
    return spikes_record



class Rate_coded_Firing_cells: # a cell
    def __init__(self, index, activity_time_bin=100, time_unit=0.001, \
                        spontaneous_freq=1): #time_range: second 
        self.index=index
        self.spontaneous_firing_rate=spontaneous_freq        
        self.max_firing_rate=100+spontaneous_freq #+1 for poisson intensity to be non-zero
        self.current_firing_rate=self.spontaneous_firing_rate
        
        self.activity_time_bin=activity_time_bin        
        self.time_unit=time_unit

    def stimulation(self, stimuli):
        spike_patterns=[]
        
        for stml in stimuli:            
            normalized_unit= (self.max_firing_rate-self.spontaneous_firing_rate)/100
            varying_intensities=int(normalized_unit*stml) if stml>1 else normalized_unit
            firing_intensity=varying_intensities+self.spontaneous_firing_rate
            if firing_intensity<1:
                raise Exception('firing_intensity<1', '\nfiring_intensity', firing_intensity\
                ,'=normalized unit:', normalized_unit, '* stml:', stml)
            self.current_firing_rate = np.clip(firing_intensity, self.spontaneous_firing_rate, self.max_firing_rate)


            spike_pttn_per_bin=poisson_spike_generator(self.current_firing_rate, self.activity_time_bin, self.time_unit)
            spike_patterns+=spike_pttn_per_bin

        return spike_patterns

    def cortical_map_stimulation(self, stimuli):
        spike_patterns=[]
        
        for stml in stimuli:            
            normalized_unit= (self.max_firing_rate-self.spontaneous_firing_rate)/100
            varying_intensities=int(normalized_unit*stml) if stml>1 else normalized_unit
            firing_intensity=varying_intensities+self.spontaneous_firing_rate
            if firing_intensity<1:
                raise Exception('firing_intensity<1', '\nfiring_intensity', firing_intensity\
                ,'=normalized unit:', normalized_unit, '* stml:', stml)
            self.current_firing_rate = np.clip(firing_intensity, self.spontaneous_firing_rate, self.max_firing_rate)


            spike_pttn_per_bin=poisson_spike_generator(self.current_firing_rate, self.activity_time_bin, self.time_unit)
            spike_patterns+=spike_pttn_per_bin

        return spike_patterns
