from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import os
import sys
from SNN_Simplified_neurons import *


def draw_network(gc_color_map, mf_color_map, edges, node_GC, node_MF, edge_color_map=None):
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
    #if not edge_color_map==None:
    if not len(edge_color_map)==0:
        nx.draw_networkx_nodes(G, pos=pos,node_color=color_map)
        nx.draw_networkx_labels(G,pos=pos)
        nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color_map\
                            , alpha=0.4)

        #nx.draw(G, pos=pos,\
        #    node_color=color_map, edge_color=edge_color_map, \
        #        with_labels=True,  alpha=0.5)
    else:
        nx.draw(G, pos=pos,\
            node_color=color_map, with_labels=True)
         #scale= 2, node_color=color_map, with_labels=True)
    print('Drawing the network')
    plt.show()


num_MF=range(0,100)
num_GC=range(100,300)

K=2
NUM_modules=2
num_node_per_layer=len(num_GC)
data_shape=len(num_MF)
MF_colors=['m','y', 'c']
GC_colors = ['red', 'Black', 'green']

nGC1, nMF1, edl1 = random_networks(int(num_node_per_layer/2), int(data_shape/2), degree_connectivity=K)            
nGC2, nMF2, edl2 = random_networks(num_node_per_layer-len(nGC1), data_shape-len(nMF1), \
                                                            previous_MF_index=len(nMF1), degree_connectivity=K)

node_GC = nGC1 + nGC2
node_MF = nMF1 + nMF2
ed_list = edl1 + edl2

mf_color_map = migration_timing(len(node_MF), MF_colors, NUM_modules)
gc_color_map = migration_timing(len(node_GC), GC_colors, NUM_modules)
edge_color_map = ['red']*len(edl1) + ['green']*len(edl2)

print('Num MF:', len(node_MF),'Num GC:', len(node_GC), 'Num ed:', len(ed_list))
print('Num MF c:', len(mf_color_map),'Num GC c:', len(gc_color_map), 'Num ed c:', len(edge_color_map))
#sys.exit()
draw_network(gc_color_map, mf_color_map, ed_list, \
                node_GC, node_MF, \
                        edge_color_map=edge_color_map)


num_rewire=1
rewire_rt=0.3

if num_rewire>0:
            for i in range(num_rewire):
                    print('rewire', i+1)
                    #edl1, edl2 = rewiring(edl1, edl2, nGC1, nGC2, rewire_rt, nMF1, nMF2, edge_color_map)
                    edl1, edl2, edge_color_map, gc_color_map, \
                        unswapped_ed_inds, unswapped_gc_inds  = \
                        rewiring(edl1, edl2, nGC1, nGC2, \
                            rewire_rt, nMF1, nMF2, \
                                edge_color_map, gc_color_map, mf_color_map)
            ed_list = edl1 + edl2            
            print('color map size:', len(edge_color_map), 'edge size:', len(ed_list))
draw_network(gc_color_map, mf_color_map, ed_list, \
                node_GC, node_MF, \
                        edge_color_map=edge_color_map)

'''ed_list=np.array(ed_list)
ed_unswapped=ed_list[unswapped_ed_inds]
unswapped_MFs=[]
print('len(edge_color_map)', len(edge_color_map)\
    ,'ed_unswapped', len(ed_unswapped))

for ed in ed_unswapped:
    if not ed[0] in unswapped_MFs: 
        unswapped_MFs.append(ed[0])'''

#print('len(unswapped_MFs)', len(unswapped_MFs))

#sys.exit()

#node_MF=np.array(node_MF)
sys.exit()
print('before', 'ed:', ed_list, '\nedc:', edge_color_map)
edge_color_map=np.array(edge_color_map)
ed_c_unswapped = edge_color_map[unswapped_ed_inds]
ed_c_swapped= np.delete(edge_color_map, unswapped_ed_inds)
edge_color_map=np.concatenate((ed_c_unswapped, ed_c_swapped)).tolist()

#ind_unswp_c = np.wehre(edge_color_map==ind_unswp_c)
ed_list=np.array(ed_list)
ed_unswapped = ed_list[unswapped_ed_inds]
ed_swapped= np.delete(ed_list, unswapped_ed_inds, axis=0)
#sprint(np.shape(ed_unswapped), np.shape(ed_swapped))
ed_list=np.concatenate((ed_unswapped, ed_swapped), axis=0).tolist()

print('color map size:', len(edge_color_map), 'edge size:', len(ed_list))
print('After', 'ed:', ed_list, '\nedc:', edge_color_map)
'''node_GC=np.array(node_GC)
gc_unswapped=node_GC[unswapped_gc_inds]
gc_swapped= [gc for gc in node_GC if gc not in node_GC[unswapped_gc_inds]]
node_GC=np.concatenate((gc_unswapped, gc_swapped))
node_GC=node_GC.tolist()

draw_network(gc_color_map, mf_color_map, ed_list, \
                node_GC, node_MF, \
                        edge_color_map=edge_color_map)'''
sys.exit()

#print(ed_list)

# Define the Graph and add the nodes/edges
G = nx.DiGraph()
G.add_nodes_from(nd_list1)
G.add_nodes_from(nd_list2)
G.add_edges_from(ed_list)

color_map = []

# This will store the nodes belonging to one set of 
# the bipartite graph. In this case, the nodes which
# start with "m"
one_side_nodes = []

for node in G:
    if node in nd_list1: #GCs
        color_map.append('red')
        # Add the node to the list
        one_side_nodes.append(node)
    else: #MFs
        color_map.append('green')

'''
for i in nd_list1:
    if i<len(nd_list1)*0.333:
        nx.set_node_attributes(G, values={i: 'early'}, name='mig')
    elif i<len(nd_list1)*0.666:
        nx.set_node_attributes(G, values={i: 'mid'}, name='mig')
    else:
        nx.set_node_attributes(G, values={i: 'late'}, name='mig')'''

# Now in the `pos` attribute pass in nx.bipartite_layout,
# with the list of nodes belonging to one set of the bipartite
# graph, i.e. one_side_nodes.
nx.draw(G, pos=nx.bipartite_layout(G, one_side_nodes),node_size=140,
        scale= 2, node_color=color_map, with_labels=True)

plt.show()


sys.exit()
for gc in nd_list1:
    if G.node[gc]['mig']=='early':
        print(G.node[gc])
