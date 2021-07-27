import numpy as np
from degree_dist import *
from matplotlib import pyplot as plt

def step_function(x, threshold=1):
    if x>=threshold: return 1
    else: return 0

def ReLu(x, threshold=1):
    if x>=threshold: return x
    else: return 0

def data_padding(data, input_size, gen_size, mf_group):
    if  mf_group=='e': data=np.append(data, np.zeros(input_size-gen_size))
    elif mf_group=='m' :
        data=np.append(np.zeros(gen_size), data)
        data=np.append(data, np.zeros(input_size-2*gen_size))
    elif mf_group=='l': data=np.append(np.zeros(input_size-gen_size), data)
    return data

def poisson_spike_generator(lamda, time_bin=100, time_unit=0.001):        # poisson spike per time bin
    beta=1/lamda
    #print('lambda', lamda, 'beta:', beta)
    #time_step=0
    spikes_record=[]
    while len(spikes_record)!=time_bin:
        delta_t = np.random.exponential(beta)/time_unit
        incoming_spike = [0]*int(delta_t)+[1]        
        spikes_record+=incoming_spike
        #print('delta t', delta_t, 'len new spike', len(incoming_spike))        
        if len(spikes_record)>time_bin: spikes_record=spikes_record[:time_bin]
    
    #print('total spike', np.sum(spikes_record))
    #sys.exit()
    return spikes_record
    
    
    
    

def data_generation(input_size, num_input, gen_mode,  firing_rate=0.1, mf_group='none_specified'): # m or e or l
    data = []
    if not mf_group=='none_specified': gen_size=int(input_size/3)
    else: gen_size=input_size
    if gen_mode=='random':
        for inpt in range(num_input):
            datum=np.random.randint(2, size= gen_size)
            if not mf_group=='none_specified': datum = data_padding(datum, input_size, gen_size, mf_group)
            data.append(datum)
            
    elif gen_mode=='rate_fixed':
        for inpt in range(num_input):
            datum=np.random.choice([0, 1], size= gen_size, p= [1-firing_rate, firing_rate])
            if not mf_group=='none_specified': datum = data_padding(datum, input_size, gen_size, mf_group)
            data.append(datum)
    elif gen_mode=='prob_of_rate_fixed':
        print('some prob of fixed rate')
    else:
        if len(gen_mode)==1:
            for i, inpt in enumerate(range(num_input)):
                #print(gen_mode[0][0])
                datum=[int(gen_mode[0][i%len(gen_mode[0])])]*input_size
                print(datum)
                #sys.exit()
                #for i, gen in enumerate(range(gen_size)):
                #    datum.append(int(gen_mode[i%len(gen_mode)]))
                data.append(datum)
        elif len(gen_mode)>1:
            for i, inpt in enumerate(range(num_input)):
                datum=[]
                while len(datum)!=input_size:
                    for k in gen_mode:
                        if len(datum)!=input_size:
                            datum.append(int(k[i%len(k)]))
                data.append(datum)
                
                #for d in range(len(gen_mode[0])):
                #datum=[int(gen_mode[i%len(gen_mode[1])][i%len(gen_mode)])]*input_size

            


                #for j, code in enumerate(range(len(gen_mode))):

    data=np.array(data)
    return data

def timing_tipping_point_ind(color_map, colors, partition):    

    timing_ind=[]
    #middle_flag=False
    #Late_flag=False
    num_flag=partition-1
    flags=[False]*num_flag    

    
    for ind_f, flag in enumerate(flags):
        for ind, c in enumerate(color_map):        
            #if not flag and np.all(c!=color_map[ind_f+1]):
            if not flag and np.all(c not in ['blue', 'powderblue', 'red', 'r']):
                #print('flag ind', ind)
                flag=True
                timing_ind.append(ind)
            
    timing_ind.append(len(color_map))  # 4.14 afternoon editted
        #if   not flags[0] and np.all(c!=color_map[1]):
        #    middle_flag=True
        #    timing_ind.append(ind)
        #elif not flags[1] and np.all(c==colors[2]):
        #    Late_flag=True
        #    timing_ind.append(ind)
    #print('timing_ind',timing_ind)
    return timing_ind
def checkIfDuplicates_3(listOfElems):#''' Check if given list contains any duplicates '''        
            for elem in listOfElems:
                if listOfElems.count(elem) > 1:
                    return True, elem
            return False, []

def rewiring(edl1, edl2, node_GC1, node_GC2, rewire_rate, MF1, MF2, ed_color_map=[], gc_color_map=[], mf_color_map=[]):
    Num_swap = int(len(edl1)*rewire_rate)
    #print(MF1)
    #if MF1[0] in MF1:
    #    print(MF1[0])
    
    duplicates, elem = checkIfDuplicates_3(edl1+edl2) #duplication prior check
    if duplicates: raise Exception('duplictes from the beginning:', duplicates, elem)

    for ind in range(Num_swap): # Swapping
        ind_swap1 = np.random.choice(len(edl1))
        ind_swap2 = np.random.choice(len(edl2))        
                
        #duplication preswap check && maximum(2) swapping constraint
        cnt=0
        while [edl2[ind_swap2][0], edl1[ind_swap1][1]] in edl1 or [edl1[ind_swap1][0], edl2[ind_swap2][1]] in edl2\
            or len([ed for ed in edl1 if ed[1]==edl1[ind_swap1][1] and ed[0] in MF2])==2\
            or len([ed for ed in edl2 if ed[1]==edl2[ind_swap2][1] and ed[0] in MF1])==2:
            
            cnt+=1
            if cnt==10000:
                raise Exception('Infinite loop')
            if [edl2[ind_swap2][0], edl1[ind_swap1][1]] in edl1 or \
                len([ed for ed in edl1 if ed[1]==edl1[ind_swap1][1] and ed[0] in MF2])==2: 
                ind_swap1 = np.random.choice(len(edl1))
            if [edl1[ind_swap1][0], edl2[ind_swap2][1]] in edl2 or \
                len([ed for ed in edl2 if ed[1]==edl2[ind_swap2][1] and ed[0] in MF1])==2:
                ind_swap2 = np.random.choice(len(edl2))            
                        
        edl1[ind_swap1][0], edl2[ind_swap2][0] = edl2[ind_swap2][0], edl1[ind_swap1][0]
        
        if edl1[ind_swap1][0] in MF2: ed_color_map[ind_swap1]='cyan'
        else: ed_color_map[ind_swap1]='red'
        
        if edl2[ind_swap2][0] in MF1: ed_color_map[len(edl1)+ind_swap2]='magenta'
        else: ed_color_map[len(edl1)+ind_swap2]='green'
            
        #ed_color_map[ind_swap1], ed_color_map[len(edl1)+ind_swap2] = 'cyan', 'magenta'

        GC1_ind, GC2_ind = node_GC1.index(edl1[ind_swap1][1]), node_GC2.index(edl2[ind_swap2][1])
        #gc_color_map[GC1_ind], gc_color_map[len(node_GC1)+GC2_ind] = 'sienna', 'gold'

        for ind, gc in enumerate(node_GC1):
            if len([ed for ed in edl1 if ed[1]==gc and ed[0] in MF2])==1: gc_color_map[ind]='powderblue'
            elif len([ed for ed in edl1 if ed[1]==gc and ed[0] in MF2])==2: gc_color_map[ind]='blue'
            else: gc_color_map[ind]='red'

        for ind, gc in enumerate(node_GC2):
            if len([ed for ed in edl2 if ed[1]==gc and ed[0] in MF1])==1: gc_color_map[len(node_GC1)+ind]='violet'
            elif len([ed for ed in edl2 if ed[1]==gc and ed[0] in MF1])==2: gc_color_map[len(node_GC1)+ind]='indigo'
            else: gc_color_map[len(node_GC1)+ind]='green'
        

        duplicates, elem = checkIfDuplicates_3(edl1+edl2) #duplication after check
        if duplicates: raise Exception('duplictes after swapping:', duplicates, elem)
        
        #gc_color_map[GC1_ind], gc_color_map[len(node_GC1)+GC2_ind] = 'darkviolet', 'blue'
        #ed_color_map[ind_swap1], ed_color_map[len(edl1)+ind_swap2] = 'darkviolet', 'blue'

        #draw_network(gc_color_map, mf_color_map, edl1+edl2, node_GC1+node_GC2, MF1+MF2, \
        #                edge_color_map=ed_color_map)
        #gc_color_map[GC1_ind], gc_color_map[len(node_GC1)+GC2_ind] = 'sienna', 'gold'
        #ed_color_map[ind_swap1], ed_color_map[len(edl1)+ind_swap2] = 'sienna', 'gold'
        #draw_network(gc_color_map, mf_color_map, edl1+edl2, node_GC1+node_GC2, MF1+MF2, \
        #                edge_color_map=ed_color_map)
    
    num_swapped1=len([ed for ed in edl1 if ed[0] in MF2])
    num_swapped2=len([ed for ed in edl2 if ed[0] in MF1])
    #print('Total edge:', len(edl1), 'Num swapped:', num_swapped1+num_swapped2)    
    print('Total edge:', len(edl1+edl2), 'Num swap for each group:', Num_swap, '/Num swapped1:', num_swapped1, 'Num swapped2:', num_swapped2)

    duplicates, elem = checkIfDuplicates_3(edl1+edl2) #duplication after check
    if duplicates: print('duplictes after swapping:', duplicates, elem)
    return edl1, edl2, ed_color_map, gc_color_map

class Firing_cells: # a cell
    def __init__(self, index, activity_time_bin=100, time_range=1, time_unit=0.001, normal_freq=2): #time_range: second 
        self.index=index
        self.normal_state_firing_rate=normal_freq
        self.activated_state_firing_rate=100

        self.current_firing_rate=self.normal_state_firing_rate

        self.current_time=0
        self.activity_time_bin=activity_time_bin
        self.time_range=time_range
        self.time_unit=time_unit

    def stimulation(self, stimuli):        

        spike_patterns=[]
        acitivity_step = self.time_range/self.time_unit/self.activity_time_bin

        for stml in stimuli:
            self.current_firing_rate=self.normal_state_firing_rate if not stml else self.activated_state_firing_rate
            spike_pttn_per_bin=poisson_spike_generator(self.current_firing_rate, self.activity_time_bin, self.time_unit)
            spike_patterns+=spike_pttn_per_bin

        return spike_patterns
        
        

class synapse:
    def __init__(self, ind):
        self.index=ind
        self.weight=np.random.uniform(low=0,high=1)
        


class UBN: # Uniform binary network
    def __init__(self, data_shape, num_layer, num_node_per_layer, network='random', \
        net_drawing=False, rewire_rt=0, num_rewire=0):  
        self.num_layers = num_layer  # the input layer does not count as a layer
        self.input_shape=data_shape
        self.num_node_per_layer=num_node_per_layer
        self.network_type=network
        #MF_colors=['c','y', 'm']
        MF_colors=['m','y', 'c']        
        self.mf_color_map = migration_timing(data_shape, MF_colors)    
        self.gc_color_map = migration_timing(num_node_per_layer, GC_colors)

        if network=='random': 
            self.node_GC, self.node_MF, self.ed_list = random_networks(self.num_node_per_layer, self.input_shape)
        elif network=='random_separate':
            nGC1, nMF1, edl1 = random_networks(self.num_node_per_layer, self.input_shape)            
            nGC2, nMF2, edl2 = random_networks(self.num_node_per_layer, self.input_shape, previous_MF_index=len(nMF1))            

            self.mf_color_map = migration_timing(data_shape*2, MF_colors, bipartite=True)    
            self.gc_color_map = migration_timing(num_node_per_layer*2, GC_colors, bipartite=True)            
            self.edge_color_map = ['r']*len(edl1) + ['g']*len(edl2)
            
            for i in range(num_rewire):
                print('rewire', i+1)
                #edl1, edl2 = rewiring(edl1, edl2, nGC1, nGC2, rewire_rt, nMF1, nMF2, self.edge_color_map)
                edl1, edl2, self.edge_color_map, self.gc_color_map  = rewiring(edl1, edl2, nGC1, nGC2, rewire_rt, nMF1, nMF2, \
                    self.edge_color_map, self.gc_color_map, self.mf_color_map)
                
                print('color mapt size:', len(self.edge_color_map), 'edge size:', len(edl1+edl2))
            #draw_network(self.gc_color_map, self.mf_color_map, edl1+edl2, nGC1+nGC2, nMF1+nMF2, \
            #            edge_color_map=self.edge_color_map)
            

            self.node_GC = nGC1 + nGC2
            self.node_MF = nMF1 + nMF2
            self.ed_list = edl1 + edl2
        else:
            self.node_GC, self.node_MF, self.ed_list = input_processing(network)

        if net_drawing: #neuralnet4(self.node_GC, self.node_MF, self.ed_list, \
                        # MF_subgroup='total', plot_frequency_dist=False, \
                        #     draw_net=True, mf_color_map=self.mf_color_map, gc_color_map=self.gc_color_map)
                        draw_network(self.gc_color_map, self.mf_color_map, self.ed_list, self.node_GC, self.node_MF\
                            ,edge_color_map=self.edge_color_map) 
        #self.mf_color_map = migration_timing(self.node_MF, MF_colors)    
        #self.gc_color_map = migration_timing(self.node_GC, GC_colors)
        
        if network=='random_separate': self.partition=2
        else: self.partition=2
        self.gc_timing_ind = timing_tipping_point_ind(color_map= self.gc_color_map, colors=GC_colors, partition=self.partition)
        self.mf_timing_ind = timing_tipping_point_ind(color_map= self.mf_color_map, colors=MF_colors, partition=self.partition)
        print('input shape', self.input_shape, 'num layer:', self.num_layers)

    """
    def feed_forward_dense(self, data, threshold=2):
        network_output=[]
        for datum in data:
            datum_in_processing=datum
            print('input datum', datum_in_processing)
            for layer in range(self.num_layers):
                
                
                
                layer_output=[]
                for node in range(self.num_node_per_layer):    # for each node
                    summation=sum(datum_in_processing)     #input sum
                    single_node_output=step_function(summation, threshold=threshold) # activation function
                    layer_output.append(single_node_output)     #add a single node output
                datum_in_processing=layer_output
                print('datum_in_processing', datum_in_processing)
            network_output.append(datum_in_processing)
        
        return np.array(network_output)"""
    """             
    def timing_ind(self, color_map, colors):
        timing_ind=[]
        middle_flag=False
        Late_flag=False
        for ind, c in enumerate(self.color_map):
            if   not middle_flag and c==colors[1]:
                middle_flag=True
                timing_ind.append(ind)
            elif not Late_flag and c==colors[2]:
                Late_flag=True
                timing_ind.append(ind)
        return timing_ind"""
    """
    def feed_forward_gcmf(self, data, threshold=2, inhibition=False, print_process=False):
        print('inhibition',inhibition, 'Feeding Forward')
        #if network=='random': node_GC, node_MF, ed_list = random_networks(self.num_node_per_layer, self.input_shape)
        #print('node GC', node_GC)
        #print('node MF', node_MF)
        #print('ed_list', ed_list)
        
        #neuralnet4(node_GC, node_MF, ed_list, \
        #             MF_subgroup='total', plot_frequency_dist=False, draw_net=True)

        if not self.network_type=='random_separate': data=np.reshape(data, [-1, self.input_shape] )
        else: 
            data_comp=np.reshape(data, [-1, self.input_shape] )            
            data=np.concatenate([data_comp, data_comp], axis=1)

            
        if print_process: print('data shape', data.shape)
        network_output=[]
        inhibition_level=0
        if inhibition: print('Inhibition Threshold:', len(self.node_GC)/2)
        for datum in data:
            datum_in_processing=datum
            if print_process: print('input datum', datum_in_processing)        
            
            layer_output=[]
            for gc in self.node_GC:
                single_node_output=0
                for edge in [eg for eg in self.ed_list if eg[1]==gc]:
                    #print('edge:',edge, 'MF index',node_MF.index(edge[0]), 'increment:', datum[node_MF.index(edge[0])])                    

                    MF_input = datum[self.node_MF.index(edge[0])]
                    single_node_output+=MF_input                        #input sum
                
                single_node_output=single_node_output+inhibition_level
                #print('single_node_sum', single_node_output)                    
                single_node_output=step_function(single_node_output, threshold=threshold) # activation function
                #single_node_output=ReLu(single_node_output, threshold=threshold) # activation function
                #print('single_node_output', single_node_output)      
                layer_output.append(single_node_output)
            
            
            datum_in_processing=layer_output
            #print('\u03A3in',np.sum(datum),'\u03A3Out', np.sum(datum_in_processing), "inhb", inhibition_level )
            if inhibition:
                print('\u03A3in',np.sum(datum),'\u03A3Out', np.sum(datum_in_processing), "inhb", inhibition_level )
                if np.sum(datum_in_processing)>=len(self.node_GC)/2: inhibition_level-=1
                elif np.sum(datum_in_processing)<len(self.node_GC)/2: inhibition_level=0                
            if print_process: print('datum_output', datum_in_processing)


            network_output.append(datum_in_processing)
        
        return np.array(network_output)
    """
    def feed_forward_feq_code(self, data, weight_list, threshold=2, inhibition=False, print_process=False):        
        print('inhibition',inhibition, 'Feeding Forward')
        freq_time_bin=10
        data=np.reshape(data, [-1, self.input_shape] )
        if self.network_type=='random_separate':             
            data=np.concatenate([data, data], axis=1)
        if print_process: print('feed forward data shape', data.shape)
        network_output=[]
        inhibition_level=0
        if inhibition: print('Inhibition Threshold:', len(self.node_GC)/2)
        for data_ind, datum in enumerate(data): #data_ind:time step            
            if data_ind > freq_time_bin:    freq_bin=np.s_[data_ind-freq_time_bin:data_ind]
            elif data_ind <= freq_time_bin: freq_bin=np.s_[:data_ind]
            input_in_freq_bin=data[freq_bin]

            if print_process: print(data_ind,'-th input datum', input_in_freq_bin)
            layer_output=[]
            for gc in self.node_GC:
                single_node_output=0
                for edge_ind, edge in [(eg_ind, eg) for eg_ind, eg in enumerate(self.ed_list) if eg[1]==gc]:
                    #print('edge:',edge, 'MF index',self.node_MF.index(edge[0]), 'increment:', datum[self.node_MF.index(edge[0])])                                            
                    
                    #MF_input = datum[self.node_MF.index(edge[0])]                    
                    MF_input = data[freq_bin, self.node_MF.index(edge[0])]
                    #input sum
                    #single_node_output+=np.sum(MF_input[np.where(MF_input)])*weight_list[edge_ind]
                    single_node_output+=np.sum(MF_input[np.where(MF_input)])
                    
                    #sys.exit()
                single_node_output=single_node_output+inhibition_level
                #print('single_node_sum', single_node_output)                    
                #single_node_output=step_function(single_node_output, threshold=threshold) # activation function
                single_node_output=step_function(single_node_output, threshold=threshold*freq_time_bin/5) # activation function
                #single_node_output=ReLu(single_node_output, threshold=threshold) # activation function
                #print('single_node_output', single_node_output)      
                layer_output.append(single_node_output)
            #datum_in_processing=layer_output
            #print('\u03A3in',np.sum(datum),'\u03A3Out', np.sum(datum_in_processing), "inhb", inhibition_level )
            if inhibition:
                input_bin_sum = np.sum(input_in_freq_bin)
                output_sum=np.sum(layer_output)
                #print('time step:{:<4}'.format(data_ind),'\u03A3in',input_bin_sum,'\u03A3Out', output_sum, "inhbition exerted", inhibition_level )
                if  output_sum>=len(self.node_GC)/2: inhibition_level-=1
                elif output_sum<len(self.node_GC)/2 and inhibition_level<0: inhibition_level+=1
            if print_process: print('datum_output', layer_output)
            network_output.append(layer_output)
        return np.array(network_output)    

                      
"""
def data_index_of_tipping_point(x,y, timing_ind, gc_color_map=[]):
    #index_early=np.argwhere(np.all( [y <  timing_ind[0]], axis=0))    
    #index_middl=np.argwhere(np.all( [y >= timing_ind[0], y<timing_ind[1]], axis=0))
    #index_latee=np.argwhere(np.all( [y >= timing_ind[1]], axis=0))

    indices=[]

    if len(gc_color_map)==0:
        prev_ind=0
        for ind in timing_ind:        
            indices.append(np.argwhere(np.all( [y <  ind] and [y >=  prev_ind], axis=0)))
            prev_ind=ind
            #print('ind:',ind)
        #sys.exit()
    else:
        for ind, color in enumerate(gc_color_map):
            indices.append(np.argwhere(np.all( [y ==  ind], axis=0)))



    def sqz_or_empty(index):        
        if   index.ndim==0: index=np.array([])
        elif index.ndim>1: index=np.squeeze(index, axis=1)
        return index
    
    #return sqz_or_empty(index_early), sqz_or_empty(index_middl), sqz_or_empty(index_latee)
    for ind in indices:
        ind = sqz_or_empty(ind)    
    
    return indices"""





def raster_plot(data, timing_ind, partition=3, print_ratio=False, gc_color_map=[]):
    num_data_point=np.shape(data)[0]
    num_GC=np.shape(data)[1]
    
    

    if len(timing_ind)==1:
        print('data shape:', np.shape(data))
        fig, ax = plt.subplots()
        if timing_ind==[0]:
            spike_color='k'
            plt.title('Input external stimuli')
            Label='Stimuli'
        elif timing_ind==[1]:
            spike_color='b'
            plt.title('Spike trains raster plot per MFs')
            Label='Neurons'

        
        x,y = np.argwhere(data == 1).T
        
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.scatter(x,y , marker='|', color=spike_color)
        #fig = plt.scatter(x,y , marker='|', color='black')

        plt.xlabel('Time steps')
        plt.ylabel(Label)
        xtick, ytick = range(len(data)) if len(data) <= 10 else None, range(len(data[0])) if len(data[0])<=10 else None
        plt.xticks(xtick)
        plt.yticks(ytick)
        plt.show()

        
        return 0

    
    



    print('num_GC', num_GC, 'num_data_point', num_data_point, 'timing_ind', timing_ind)
    #sys.exit()

    x,y = np.argwhere(data == 1).T    
    #print(x, 'n', y)
    #sys.exit()

    if print_ratio:
        #activated_eGC_ratio = len(index_early)/(timing_ind[0]*num_data_point)
        #activated_mGC_ratio = len(index_middl)/((timing_ind[1]-timing_ind[0])*num_data_point)
        #activated_lGC_ratio = len(index_latee)/((num_GC-timing_ind[1])*num_data_point)        
        activated_totalGC_ratio = len(x)/((num_GC)*num_data_point)

        #print('total GC num=',num_GC, 'e, m, l:', timing_ind[0], timing_ind[1]-timing_ind[0], num_GC-timing_ind[1])
        #print('total GC num=',num_GC, 'timing partition index:', timing_ind)
        #print('activated_eGC_ratio', activated_eGC_ratio)
        #print('activated_mGC_ratio', activated_mGC_ratio)
        #print('activated_lGC_ratio', activated_lGC_ratio)
        print('activated_total_GC_ratio', activated_totalGC_ratio)        

    
    
    if num_GC<40: plt.yticks(np.arange(0, num_GC, step=1)) # if num_data_point*num_GC<200: 
    else: plt.yticks(np.arange(0, num_GC, step=int(num_GC/10)))
    if num_GC*num_data_point>2000: size=3
    
    
    plt.scatter(x, y , marker='|', color=np.array(gc_color_map)[y], s=size)
    #mf_group='mf_grouptest'
    #plt.title('Spike raster plot for mf_group %s'% mf_group )
    #plt.title('Spike raster plot for No inhibition')
    #plt.title('Spike raster plot for inhibition case4')
    plt.title('GC Spike raster plot with synapse')
    plt.xlabel('Time steps')
    plt.ylabel('Neuron numbers')
      # Set label locations.
    #plt.yticks(np.arange(0, num_MFs, step=1))  # Set label locations.
    #plt.xticks(np.arange(0, num_data_point, step=1))  # Set label locations.
    plt.show()

