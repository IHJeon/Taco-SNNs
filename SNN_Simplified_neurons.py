#from Spiking_Single_cells_multiple_dend import TIME_RANGE
#from RateCode_simulator import NUM_MFs
from Simplified_spiking_cells import *
#from RateCodingModel import *
from matplotlib import pyplot as plt

def softmax(valueset): 
    #return np.exp(valueset)/np.exp(valueset).sum()
    return valueset/valueset.sum() # no Exponential

'''
def Pruning(WEIGHT_MATRIX, NUM_PRUN=1):    
    Row=np.shape(WEIGHT_MATRIX)[0]
    Col=np.shape(WEIGHT_MATRIX)[1]-NUM_PRUN
    Pruned_MAT=np.zeros((Row, Col))
    for r in Row:
        mat=WEIGHT_MATRIX[r]
        ind_prun=np.where(mat==np.amin(mat))
        pruned=np.delete(mat, mat[ind_prun[0]])
        Pruned_MAT[r]=pruned
    
    return Pruned_MAT'''




def random_networks(Num_GCs, Num_MFs, previous_MF_index=0,\
             degree_connectivity=4, degree_of_modularity=0, Selection_basis=[]):    
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
            if Selection_basis!=[]: 
                preference= softmax(Selection_basis)
                modular_prob_assign=np.multiply(modular_prob_assign, preference)
                modular_prob_assign=softmax(modular_prob_assign)
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


def migration_timing(cells, colors, Num_Modules=3):
    if type(cells)==int: num_cells=cells
    else: num_cells=len(cells)
    color_map = []    
    denominator=Num_Modules
    Num_cells_per_modules=int(num_cells/denominator)
    rmnder=num_cells-Num_Modules*Num_cells_per_modules    
    print('Num_cells_per_modules', Num_cells_per_modules, 'rmnder', rmnder)
    color_division=0  #timing=colors
    modules=[]
    for i in range(Num_Modules):
        if not i==Num_Modules-1: modules.append([colors[i]]*Num_cells_per_modules)
        else: modules.append([colors[i]]*(Num_cells_per_modules+rmnder))

    color_map=np.ndarray.tolist(np.hstack(modules))
    return color_map

class SNN_connectivity: # Spiking Neural networks
    def __init__(self, data_shape, num_node_per_layer, net_drawing=False, K=4, \
                    modularity=0, NUM_modules=1, num_rewire=0, rewire_rt=0, \
                        NETWORK_DRAW=False, Spike_counts=[]):
        self.input_shape=data_shape
        self.num_node_per_layer=num_node_per_layer
        self.MF_colors=['m','y', 'c']
        self.GC_colors = ['red', 'Black', 'green']

        if NUM_modules==1:
            self.node_GC, self.node_MF, self.ed_list = random_networks(self.num_node_per_layer,\
                        self.input_shape, degree_connectivity=K, \
                        degree_of_modularity=modularity, Selection_basis=Spike_counts)
            self.edge_color_map = ['k']*len(self.ed_list)
        elif NUM_modules==2:
            self.nGC1, self.nMF1, self.edl1 = random_networks(int(num_node_per_layer/2),\
                 int(data_shape/2), degree_connectivity=K, Selection_basis=Spike_counts)
            self.nGC2, self.nMF2, self.edl2 = random_networks(num_node_per_layer-len(self.nGC1),\
                 data_shape-len(self.nMF1), previous_MF_index=len(self.nMF1),\
                      degree_connectivity=K, Selection_basis=Spike_counts)
            self.node_GC = self.nGC1 + self.nGC2
            self.node_MF = self.nMF1 + self.nMF2
            self.ed_list = self.edl1 + self.edl2
            self.edge_color_map = ['red']*len(self.edl1) + ['green']*len(self.edl2)
        self.mf_color_map = migration_timing(len(self.node_MF), self.MF_colors, NUM_modules)
        self.gc_color_map = migration_timing(len(self.node_GC), self.GC_colors, NUM_modules)
        
        
        if NUM_modules>1 and num_rewire>0:
            for i in range(num_rewire):
                    print('rewire', i+1)
                    #edl1, edl2 = rewiring(edl1, edl2, nGC1, nGC2, rewire_rt, nMF1, nMF2, self.edge_color_map)
                    self.edl1, self.edl2, self.edge_color_map, self.gc_color_map  = \
                        rewiring(self.edl1, self.edl2, self.nGC1, self.nGC2, \
                            rewire_rt, self.nMF1, self.nMF2, \
                                self.edge_color_map, self.gc_color_map, self.mf_color_map)
            self.ed_list = self.edl1 + self.edl2
            print('color mapt size:', len(self.edge_color_map), 'edge size:', len(self.ed_list))
        if NETWORK_DRAW:
            draw_network(self.gc_color_map, self.mf_color_map, self.ed_list, \
                self.node_GC, self.node_MF, \
                        edge_color_map=self.edge_color_map)

        

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
    def show_GCGC_edge_matrix(self, GRAPH=True, Title=''):
        nd_GC, nd_MF = len(self.node_GC), len(self.node_MF)
        pallet = np.zeros((nd_GC, nd_GC))        

        for ind_mf in range(nd_MF):
            #print('target eds:', [ed for ed in self.ed_list if int(ed[0][1:])==ind_mf])
            ind_target_gcs=[int(ed[1][1:]) for ed in self.ed_list if int(ed[0][1:])==ind_mf]
            for ind_gc in ind_target_gcs:
                excluded=[int(ex) for ex in ind_target_gcs if ex!=ind_gc]
                #print('gc:',ind_gc, 'exs:', excluded)
                #sys.exit()
                for ex in excluded:
                    pallet[ind_gc, ex]=1
                #print('mf', ind_mf,'target_gcs:', target_gc)
                #int(target_ed[1][1:])
        fig, ax = plt.subplots()

        if GRAPH: plt.title('Graph Weight Connectivity matrix')
        else: 
            plt.title('Weight Connectivity matrix')
            plt.xlabel('MFs')
            plt.ylabel('GCs')
        ax.matshow(pallet, cmap=plt.cm.Blues)
        plt.title('Original')
        plt.show()
        '''
        pallet=np.flip(pallet, axis=0)
        #pallet=np.flip(pallet)
        fig, ax = plt.subplots()
        ax.matshow(pallet, cmap=plt.cm.Blues)
        plt.title('axis 0 fliped')
        #ax.invert_yaxis()
        plt.show()'''
        REORDER=True
        #symmetric=True
        symmetric=False
        if REORDER:
            fig, ax = plt.subplots() 
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import reverse_cuthill_mckee

            graph = csr_matrix(pallet)        
            permutation = reverse_cuthill_mckee(graph, symmetric_mode=symmetric)
            arr_perm=pallet[permutation]            
            #arr_perm=pallet(np.ndarray.tolist(perm))
            if not Title=='':plt.title(Title+'symmetric={}'.format(symmetric))
            else: plt.title('Reordered Graph weight matrix, symmetric={}'.format(symmetric))
            ax.matshow(arr_perm, cmap=plt.cm.Blues)
        else: ax.matshow(pallet, cmap=plt.cm.Blues)
        #ax.invert_yaxis()
        plt.show()
        print("Total Num Weight", np.sum(pallet))


    def show_edge_matrix(self, GRAPH=True, Title=''):
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
        REORDER=True
        symmetric=True
        if REORDER:
            fig, ax = plt.subplots() 
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import reverse_cuthill_mckee

            graph = csr_matrix(pallet)        
            permutation = reverse_cuthill_mckee(graph, symmetric_mode=symmetric)
            arr_perm=pallet[permutation]            
            #arr_perm=pallet(np.ndarray.tolist(perm))
            if not Title=='':plt.title(Title+'symmetric={}'.format(symmetric))
            else: plt.title('Reordered Graph weight matrix, symmetric={}'.format(symmetric))
            ax.matshow(arr_perm, cmap=plt.cm.Blues)
        else: ax.matshow(pallet, cmap=plt.cm.Blues)
        ax.invert_yaxis()
        plt.show()
        print("Total Num Weight", np.sum(pallet))

def checkIfDuplicates_3(listOfElems):#''' Check if given list contains any duplicates '''        
            for elem in listOfElems:
                if listOfElems.count(elem) > 1:
                    return True, elem
            return False, []

def rewiring(edl1, edl2, node_GC1, node_GC2, rewire_rate, MF1, MF2, ed_color_map=[], gc_color_map=[], mf_color_map=[]):
    Num_swap = int(len(edl1)*rewire_rate)
    # color mixer -> https://trycolors.com/
    greeny_red='#9F4400'  #greeny red 3:5
    redish_green ='#607100' #redish green 3:5
    #mix_redgreen='#805B00'
    mix_redgreen='k'
    
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
            if cnt==20000:
                raise Exception('Infinite loop in rewiring (exceeded 20000 counts)')
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
        
        for ind, gc in enumerate(node_GC1):
            if len([ed for ed in edl1 if ed[1]==gc and ed[0] in MF2])==1: gc_color_map[ind]=greeny_red
            #elif len([ed for ed in edl1 if ed[1]==gc and ed[0] in MF2])==2: gc_color_map[ind]='blue'
            elif len([ed for ed in edl1 if ed[1]==gc and ed[0] in MF2])==2: gc_color_map[ind]=mix_redgreen 
            else: gc_color_map[ind]='red'

        for ind, gc in enumerate(node_GC2):
            if len([ed for ed in edl2 if ed[1]==gc and ed[0] in MF1])==1: gc_color_map[len(node_GC1)+ind]=redish_green
            elif len([ed for ed in edl2 if ed[1]==gc and ed[0] in MF1])==2: gc_color_map[len(node_GC1)+ind]=mix_redgreen
            else: gc_color_map[len(node_GC1)+ind]='green'
        

        duplicates, elem = checkIfDuplicates_3(edl1+edl2) #duplication after check
        if duplicates: raise Exception('duplictes after swapping:', duplicates, elem)
    num_swapped1=len([ed for ed in edl1 if ed[0] in MF2])
    num_swapped2=len([ed for ed in edl2 if ed[0] in MF1])    
    print('Total edge:', len(edl1+edl2), 'Num swap for each group:', Num_swap, '/Num swapped1:', num_swapped1, 'Num swapped2:', num_swapped2)

    duplicates, elem = checkIfDuplicates_3(edl1+edl2) #duplication after check
    if duplicates: print('duplictes after swapping:', duplicates, elem)        
    
    #ed_color_map=np.array(ed_color_map)
    
    #unswapped_ed_inds=np.where(ed_color_map=='green' or ed_color_map=='red')
    unswapped_ed_inds=[ind for (ind, edc) in enumerate(ed_color_map)\
                       if edc=='green' or edc=='red' ] 
    #unswapped_ed_inds=[edc for (ind, edc) in enumerate(ed_color_map)]    
    unswapped_gc_inds=np.where(gc_color_map=='green' or gc_color_map=='red')
    #print('unswapped', unswapped_ed_inds)    
    #sys.exit()
    #return edl1, edl2, ed_color_map, gc_color_map, unswapped_ed_inds, unswapped_gc_inds
    return edl1, edl2, ed_color_map, gc_color_map


def draw_network(gc_color_map, mf_color_map, edges, node_GC, node_MF, edge_color_map=None):
    import networkx as nx
    
    nodes= node_GC + node_MF
    color_map=gc_color_map+mf_color_map
    edge_color_map=edge_color_map

    if len(nodes)/len(color_map)>1: 
        duplication=int(len(nodes)/len(color_map))
    else: duplication=1

    if duplication!=1:
        color_map=gc_color_map*duplication+mf_color_map*duplication

    #G = nx.DiGraph() # Define the Graph and add the nodes/edges
    #G = nx.Graph() # Define the Graph and add the nodes/edges
    G = nx.MultiGraph()
    #G=nx.complete_bipartite_graph(node_GC,node_MF)
    #G= nx.OrderedDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    expansion_rate=len(node_GC)/len(node_MF)
    pos=       dict(zip(node_MF, zip([0]*len(node_MF), np.arange(len(node_MF))*0.1*expansion_rate ) )) # leftt nodes
    #print(pos)
    pos.update(dict(zip(node_GC, zip([1]*len(node_GC), np.arange(len(node_GC))*0.1  ))))               # right  nodes

    if len(edges)>2000:
        raise Exception("Exception you set by yourself!  Too many edges to plot can't see all the edges ")

    if len(edges)!=len(G.edges):
        #missing_list = [ed for ed in edges if not tuple(ed) in list(G.edges)]
        #duplicates=set([x for x in edges if edges.count(x) > 1])
        def checkIfDuplicates_3(listOfElems):#''' Check if given list contains any duplicates '''        
            for elem in listOfElems:
                if listOfElems.count(elem) > 1:
                    return True, elem
            return False, []
        duplicates, elem = checkIfDuplicates_3(edges)
        print('duplictes:', duplicates, elem)
        missing_list = [ed for ed in edges if ed not in [[list(edg)[1],list(edg)[0]] for edg in G.edges]]
        #missing_list = np.delete(edges, list(map(lambda x: np.where([list(ed) for ed in G.edges]==x), list(G.edges))), axis=0)
        print('num edges to draw:', len(edges),'-> num edges actually drawn:', len(G.edges))
        #print('index of missing one:', np.where(edges==[]))
        raise Exception("some edges are missing!!", missing_list, 'len missing:', len(missing_list))

    print('num edges to draw:', len(edges),'-> num edges actually drawn:', len(G.edges))
    if not edge_color_map==None:        
        #nx.draw(G, pos=pos,\
        #    node_color=color_map, edge_color=edge_color_map, with_labels=True)
        nx.draw_networkx_nodes(G, pos=pos,node_color=color_map)
        nx.draw_networkx_labels(G,pos=pos)
        nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color_map\
                            , alpha=0.4)
    else:
        #nx.draw(G, pos=pos,\
        #    node_color=color_map, with_labels=True)
         #scale= 2, node_color=color_map, with_labels=True)
        nx.draw_networkx_nodes(G, pos=pos,node_color=color_map)
        nx.draw_networkx_labels(G,pos=pos)
        nx.draw_networkx_edges(G, pos=pos, alpha=0.4)
    print('Drawing the network')
    plt.show()




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
                        spontaneous_freq=2): #time_range: second 
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
