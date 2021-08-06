from Simplified_spiking_cells import *
#from RateCodingModel import *
from matplotlib import pyplot as plt


def random_networks(Num_GCs, Num_MFs, previous_MF_index=0, degree_connectivity=4):    
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
        synapse_partner=np.random.choice(Num_MFs, size=degree_connectivity, replace=False)

        syns=sorted(synapse_partner)
        for j in syns:            
            ed_list.append(['M%s'%(j+previous_MF_index), 'G%s'%(i+previous_GC_index)])

    print('Network stats: # GCs', len(node_GC), '# MFs', len(node_MF),'# edges(synapses)', len(ed_list))
    
    return node_GC, node_MF, ed_list
def rand_ind(Low=0, max_val=100):
        return np.random.randint(0,max_val)
def rand_normal():        
    return np.random.normal(scale=0.05)        

def data_generation(input_size, input_range):
    data = []    
    max_level_stimuli=100
    
    from sklearn.preprocessing import MinMaxScaler        
    
    x= np.linspace(0,input_range,input_range)    
    #seed_val=np.random.randint(0,100)
    
    for inpt in range(input_size):        
        y=rand_ind()*np.sin(rand_normal()*x)+rand_ind()*np.cos(rand_normal()*x)
        #y=MinMaxScaler(feature_range=(0,max_level_stimuli)).fit_transform(y.reshape(-1, 1)).reshape(-1)                           
        y=abs(y)
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




class SNN_connectivity: # Spiking Neural networks
    def __init__(self, data_shape, num_node_per_layer, net_drawing=False, K=4):
        self.input_shape=data_shape
        self.num_node_per_layer=num_node_per_layer
        self.node_GC, self.node_MF, self.ed_list = random_networks(self.num_node_per_layer,\
                    self.input_shape, degree_connectivity=K)


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
