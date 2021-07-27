from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import scipy.stats as stats
import math
import os
import sys
from vpython import *
from function2 import * #load data
MF_colors=np.array([[192,192,192], [135,121,78], [109,135,100]])/256
GC_colors = ['red', 'Black', 'green']    

#num_synapses_range
def input_processing(data_name):
    load = data_load(data_name)
    synapse_data=load[:-1] #load[-1] : GC number
    num_GC=load[-1][1]

    ed_list=[]
    #mf_color_map=[]
    for mf in synapse_data:
        #mf_color_map.append([mf[-1].x, mf[-1].y, mf[-1].z])
        for synapse in mf[:-1]:
            edge=['M%s'%synapse[0], 'G%s'%synapse[1]]
            ed_list.append(edge)

    node_GC=[]
    node_MF=[]
    for ind_gc in range(num_GC):
        node_GC.append('G%s'%ind_gc)
    for ind_mf in range(len(synapse_data)):
        node_MF.append('M%s'%ind_mf)
    print('# GCs', len(node_GC))
    print('# MFs', len(node_MF))
    
    return node_GC, node_MF, ed_list

def normal_dist():
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

def broadness_difference(node_GC, node_MF, ed_list, only_original=False):
    #print('types', type(node_GC), type(node_MF), type(ed_list))
    node_GC2=node_GC.copy()
    node_MF2=node_MF.copy()
    ed_list2=ed_list.copy()

    node_GC3=node_GC.copy()
    node_MF3=node_MF.copy()
    ed_list3=ed_list.copy()
    #print('GC IDs', id(node_GC), 'MF IDs', id(node_MF), 'ed IDs', id(ed_list), \
    #    'new:', 'GC2 IDs', id(node_GC2), 'MF2 IDs', id(node_MF2), 'ed2 IDs', id(ed_list2))
    unshuffled=neuralnet4(node_GC2, node_MF2, ed_list2, \
            MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
    #unshuffled=unshuffled1.copy()
    shuffled=neuralnet4(node_GC3, node_MF3, ed_list3, \
                MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
    #rand_node_GC, rand_node_MF, rand_ed_list = random_networks(len(node_GC), len(node_MF))
    #shuffled=neuralnet4(rand_node_GC, rand_node_MF, rand_ed_list, \
    #        MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
    
    std_unshfld=np.std(unshuffled)
    #print('std_unshfld', std_unshfld)
    if only_original:
        #print('std_unshfld', '{:<7}'.format(round(std_unshfld,5)), 'ed_list', ed_list)
        print('std_unshfld', std_unshfld, 'shuffled', np.std(shuffled))
        return std_unshfld, 0
    '''
    else:
        print('print both')
        std_diff_list=[]
        for i in range(1):
            #print(i, '-th shuffling...')
            shuffled=neuralnet4(node_GC, node_MF, ed_list, \
                MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
            std_diff = np.std(shuffled) - std_unshfld
            std_diff_list.append(std_diff)
            #print(i, '-th shuffling... std_unshfld', std_unshfld, 'std_shuffled', np.std(shuffled), 'std_diff', std_diff)
        avg_broadness_difference=np.mean(std_diff_list)
        print('std_unshfld', std_unshfld, 'AVG Broadness dfference with shfld', avg_broadness_difference )    
        return std_unshfld, avg_broadness_difference'''
    
def range_for_plot(distribution, range):
    range_max=int(max(abs(distribution)))+1
    dist_range= np.arange(-range_max, range_max)
    return dist_range

def CDF_comparison(node_GC, node_MF, ed_list, comp_type='rand'):
    if not comp_type=='shf2': #data unshuffled net
        dist1=neuralnet4(node_GC, node_MF, ed_list, \
            MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)    

    if comp_type=='shf':
        dist2=neuralnet4(node_GC, node_MF, ed_list, \
                MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        plt.title('CDF_comparison_with_shuffling')            
    elif comp_type=='rand':
        rand_node_GC, rand_node_MF, rand_ed_list = random_networks(len(node_GC), len(node_MF))
        dist2=neuralnet4(rand_node_GC, rand_node_MF, rand_ed_list, \
                MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        plt.title('CDF_comparison_with_randomnet')             
    elif comp_type=='shf2':
        dist1=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        dist2=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        plt.title('CDF_comparison shuffled vs shuffled')
    elif comp_type=='shf3':
        
        dist2=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        dist3=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        dist4=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        plt.title('CDF_comparison data with shuffled 3')

    elif comp_type=='shf4':
        dist1=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        dist2=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        dist3=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        dist4=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)
        plt.title('CDF_comparison shuffled 4')
    elif comp_type=='rand3':
        num_MFs=len(node_MF)
        num_GC=num_MFs*3
        
        node_GC2, node_MF2, ed_list2 = random_networks(num_GC, num_MFs)
        node_GC3, node_MF3, ed_list3 = random_networks(num_GC, num_MFs)
        node_GC4, node_MF4, ed_list4 = random_networks(num_GC, num_MFs)
        dist1=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        dist2=neuralnet4(node_GC2, node_MF2, ed_list2, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        dist3=neuralnet4(node_GC3, node_MF3, ed_list3, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        dist4=neuralnet4(node_GC4, node_MF4, ed_list4, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        plt.title('CDF_comparison random net 3')

    elif comp_type=='rand4':
        num_MFs=4000
        num_GC=num_MFs*3
        node_GC, node_MF, ed_list = random_networks(num_GC, num_MFs)
        node_GC2, node_MF2, ed_list2 = random_networks(num_GC, num_MFs)
        node_GC3, node_MF3, ed_list3 = random_networks(num_GC, num_MFs)
        node_GC4, node_MF4, ed_list4 = random_networks(num_GC, num_MFs)
        dist1=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        dist2=neuralnet4(node_GC2, node_MF2, ed_list2, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        dist3=neuralnet4(node_GC3, node_MF3, ed_list3, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        dist4=neuralnet4(node_GC4, node_MF4, ed_list4, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)
        plt.title('CDF_comparison random net 4')
    elif comp_type=='comp':
        num_MFs=len(node_MF)
        num_GC=num_MFs*3
        
        node_GC2, node_MF2, ed_list2 = random_networks(num_GC, num_MFs)


        dist2=neuralnet4(node_GC2, node_MF2, ed_list2, \
                    MF_subgroup='total', shuffle=False, rt_ratio=True, print_stat=False)

        dist3=neuralnet4(node_GC, node_MF, ed_list, \
                    MF_subgroup='total', shuffle=True, rt_ratio=True, print_stat=False)

        st1=np.std(dist1)
        st2=np.std(dist2)
        st3=np.std(dist3)

        print('std data', '{:<6}'.format(np.round(st1,4)), '| dffrnc',\
            '\nstd rand', '{:<6}'.format(np.round(st2,4)), '|', np.round(st2-st1,4), \
            '\nstd shfl', '{:<6}'.format(np.round(st3,4)), '|',  np.round(st3-st1,4) )
        sys.exit()
    #plot_degree_dist(unshuffled, plot=True, CDF_return2=False)
    #plot_degree_dist(shuffled  , plot=True, CDF_return2=False)
    #print('std_dist1', np.std(dist1), 'std_dist2', np.std(dist2))

    x1, val1 = plot_degree_dist(dist1, plot=False, CDF_return2=True)
    x2, val2 = plot_degree_dist(dist2  , plot=False, CDF_return2=True)
    x3, val3 = plot_degree_dist(dist3  , plot=False, CDF_return2=True)
    #x4, val4 = plot_degree_dist(dist4  , plot=False, CDF_return2=True)
    
    
    #plt.title('CDF_comparison_with_shuffling')
    plt.plot(x1, val1)
    plt.plot(x2, val2)
    plt.plot(x3, val3)
    #plt.plot(x4, val4)
    if comp_type=='rand': plt.legend(['data','shuffled'])
    elif comp_type=='shf': plt.legend(['data','randomnet'])
    elif comp_type=='shf2': plt.legend(['shuffled1','shuffled2'])
    elif comp_type=='shf4': plt.legend(['shuffled1','shuffled2', 'shuffled3', 'shuffled4'])
    elif comp_type=='shf3': plt.legend(['data','shuffled1', 'shuffled2', 'shuffled3'])
    elif comp_type=='rand3': plt.legend(['data','randomnet1', 'randomnet2', 'randomnet3'])
    elif comp_type=='comp': plt.legend(['data','randomnet', 'shuffled'])
    
    plt.show()


def plot_degree_dist(distribution, normal_dist=False, plot=True, \
                    moments_return=False, CDF_return2=True, PDF=False):
    #x=[1,2,3,4,5]
    #y=[2,4,6,8,10]
    #print('variance:', np.var(distribution))
    
    if normal_dist==True:
        mu = 0
        #variance = 8.36514
        variance = 1
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))

    print_dist_stats(distribution)
    
    #plt.axvline(x=0)

    if CDF_return2:
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(distribution)
        if plot==True:        
            plt.title('ECDF')
            plt.plot(ecdf.x, ecdf.y)
            plt.show()
        return ecdf.x, ecdf.y
    #sys.exit()

    range_max=int(max(abs(distribution)))+1
    log_range_= np.arange(-range_max, range_max)
    #log_range_= np.arange(int(min(distribution)-2), int(max(distribution)+3))
    #if CDF_return2: return ecdf.x, ecdf.y
    if PDF and plot:
        print('plotting PDF...')
        plt.title('Log scale Degree of frequency distibution')
        #print('log min:', min(log_counts), 'log max:', max(log_counts), \
        #        'mean:', np.mean(log_counts), 'var:', np.var(log_counts), 'std:', np.std(log_counts))
        #print('log min:', np.round(min(log_counts),3), 'log max', np.round(max(log_counts),3), \
        dist_of_probability=True        
        plt.xlim(-range_max, range_max)
        plt.hist(distribution, bins=log_range_, density=dist_of_probability, align='mid', cumulative=False)
        plt.show()
        

    if moments_return==True:
        std=np.round(np.std(distribution),4)
        mean=np.round(np.mean(distribution),4)
        return mean, std
    

''' 
#sample drawing
def neuralnet():
    from networkx.drawing.nx_agraph import graphviz_layout

    G = nx.DiGraph()
    ed = [[0, 4, -1],
     [0, 5, -1],
     [1, 4, -1],
     [1, 5, -1],
     [2, 4, -1],
     [2, 5, 10],
     [4, 3, -1],
     [5, 3, 100]]

    G.add_weighted_edges_from(ed)
    pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
    nx.draw(G,with_labels=True,pos=pos, font_weight='bold')
    plt.show()
'''

#refer networkx_Test.py instead of this

def migration_timing(cells, colors, Num_Modules=3):
    color_map = []    
    denominator=Num_Modules

    if type(cells)==int: num_cells=cells
    else: num_cells=len(cells)

    if num_cells/denominator-int(num_cells/denominator) >= 0.5: rmnder=1
    else: rmnder=0

    color_division=0  #timing=colors
    modules=[]
    #if Num_Modules==2:
        #early=[colors[0]]*int(num_cells/denominator+rmnder)        
        #latee=[colors[2]]* (num_cells-len(early) )
        #color_map=early+latee
    for i in range(Num_Modules):
        if not i==Num_Modules: modules.append([colors[i]]*int(num_cells/denominator+rmnder))
        else: modules.append([colors[i]]*(num_cells-len(modules[0]*(num_cells-1))))

    #elif Num_Modules==3:
    #    early=[colors[0]]*int(num_cells/denominator+rmnder)
    #    midle=[colors[1]]*int(num_cells/denominator+rmnder)
    #    latee=[colors[2]]* (num_cells-2*len(early) )
    #    color_map=early+midle+latee
    """
    e, m, l =0, 0, 0
    for index, cell in enumerate(cells):
        if index<int(len(cells)/3): #early MF
            color=colors[0]
            if stat==True: e+=1
            #print('Early', index)
        elif index<int(len(cells)*2/3): #mid MF
            color=colors[1]
            if stat==True: m+=1
            #print('Mid', index)
        else:                               #late MF
            color=colors[2]
            if stat==True: l+=1
            #print('Late', index)
        color_map.append(color)        
    if stat==True: print('Colormap, Early: %d,  Mid: %d,  Late: %d '%(e, m, l))"""
    #print('Colormap, Early: %d,  Mid: %d,  Late: %d '%(len(early), len(midle), len(latee)))
    
    color_map=np.ndarray.tolist(np.hstack(modules))
    return color_map

def node_pruning(indices, color_map, toatal_cells):
    #remove_list=[]
    remove_indices=[]
    for index, cell in enumerate(toatal_cells):
        if not index in indices:
            #remove_list.append(cell)
            remove_indices.append(index)
    
    target_cells=np.ndarray.tolist(np.delete(toatal_cells, remove_indices))
    color_map=np.ndarray.tolist(np.delete(color_map, remove_indices, axis=0))

    return target_cells, color_map

def edge_pruning(remaining_nodes, edges):
    remove_list_edge=[]
    for edge in edges:
        if not edge[0] in remaining_nodes and edge not in remove_list_edge:
            remove_list_edge.append(edge)
    #eds=[edg for edg in edges if not edg in remove_list_edge]
    if len(remove_list_edge)==0: eds=edges
    else: eds=[edg for edg in edges if not edg in remove_list_edge]
    #print('remove_list_edge', len(remove_list_edge))
    #print('{:<10}'.format('pruning'), id(edges), id(eds))
    return eds

def print_dist_stats(distribution):
    print('{:<5}'.format('min:'), '{:>10}'.format(np.round(min(distribution),3)),\
          '{:<5}'.format('max:'), '{:>10}'.format(np.round(max(distribution),3)),\
          '{:<6}'.format('mean:'), '{:>10}'.format(np.round(np.mean(distribution),4)),\
          '{:<5}'.format('var:'), '{:>15}'.format(np.round(np.var(distribution),4)),\
          '{:<5}'.format('std:'), '{:>10}'.format(np.round(np.std(distribution),4))) 


def frequency_distribution(node_MF, node_GC, edges, gc_color_map, noise=True):
    total_syn=[]
    for mf in node_MF:
        syn_for_mf=[]
        for edge in edges:
            if edge[0]==mf:
                syn_for_mf.append(edge)
        total_syn.append(syn_for_mf)

    ratios=[]
    for syns_per_mf in total_syn:
        early=0
        late=0
        for syn in syns_per_mf:
            ind_GC=node_GC.index(syn[1])
            if gc_color_map[ind_GC]==GC_colors[0]: early+=1
            elif gc_color_map[ind_GC]==GC_colors[2]: late+=1
        #ratio= (early+0.1)/(late+0.1)
        #noise1 = abs(np.random.normal(0, 1e-01))
        #noise2 = abs(np.random.normal(0, 1e-01))
        if noise:
            noise1 = abs(np.random.normal(0, 1))
            noise2 = abs(np.random.normal(0, 1))
            ratio= (early+noise1+1)/(late+noise2+1)
            #ratio= (early+noise1)/(late+noise2)
            #noise1 = np.random.normal(0, 1e-05)
            #noise2 = np.random.normal(0, 1e-05)
            #ratio= (1+early+noise1)/(1+late+noise2)
        else:
            ratio= (early)/(1+late)
        #ratio = np.log(ratio)
        ratios.append(ratio)
    log_ratios=np.log(ratios)
    #log_ratios=ratios
    return log_ratios

def frequency_distribution_of_difference(node_MF, node_GC, edges, gc_color_map):
    total_syn=[]
    for mf in node_MF:
        syn_for_mf=[]
        for edge in edges:
            if edge[0]==mf:
                syn_for_mf.append(edge)
        total_syn.append(syn_for_mf)

    differences=[]
    for syns_per_mf in total_syn:
        early=0
        late=0
        for syn in syns_per_mf:
            ind_GC=node_GC.index(syn[1])
            if gc_color_map[ind_GC]==GC_colors[0]: early+=1
            elif gc_color_map[ind_GC]==GC_colors[2]: late+=1
        #ratio= (early+0.1)/(late+0.1)
        difference= early-late
        differences.append(difference)
    print('differences shape', np.shape(differences))
    return differences

def frecuency_MF(node_MF, node_GC, edges, gc_color_map):
    total_syn=[]
    for mf in node_MF:
        syn_for_mf=[]
        for edge in edges:
            if edge[0]==mf:
                syn_for_mf.append(edge)
        total_syn.append(syn_for_mf)

    counts=[]
    total_early=0
    total_mid=0
    total_late=0
    for syns_per_mf in total_syn:
        #early=0
        #mid=0
        #late=0
        for syn in syns_per_mf:
            ind_GC=node_GC.index(syn[1])
            if gc_color_map[ind_GC]==GC_colors[0]: total_early+=1
            elif gc_color_map[ind_GC]==GC_colors[1]: total_mid+=1
            elif gc_color_map[ind_GC]==GC_colors[2]: total_late+=1
        #print(early, late)
        #total_early+=early
        #total_mid+=mid
        #total_late+=late
        #count = [early, late]
        #counts.append(count)
    #avg_early=np.average(counts[0])
    #avg_late=np.average(counts[1])    
    #return total_early, total_late, avg_early, avg_late
    return total_early, total_mid, total_late

def frecuencyGC(node_MF, node_GC, edges, gc_color_map):
    Counts=[]
    for gc in node_GC:
        early=0
        late=0
        for edge in edges:
            if edge[1]==gc:
                ind=node_GC.index(gc)
                if gc_color_map(ind)==GC_colors[0]: early+=1
                elif gc_color_map[ind]==GC_colors[2]: late+=1  
        Counts.append([early, late])

def target_MF_indexing(node_MF, MF_subgroup):
    if MF_subgroup==None:
        print('Non MF subgroup selected, selecting 5 samples randomly')
        indices = np.random.choice(range(len(node_MF)), 5, replace=False)
    elif MF_subgroup=='early':
        indices = [index for index in range(len(node_MF))if index < int(len(node_MF)/3)]
    elif MF_subgroup=='mid':
        indices = [index for index in range(len(node_MF)) if index < int(len(node_MF)*2/3)\
            and index >=int(len(node_MF)/3)]
    elif MF_subgroup=='late':
        indices = [index for index in range(len(node_MF)) if index >= int(len(node_MF)*2/3)]
    elif MF_subgroup=='total':
        indices = [index for index in range(len(node_MF))]
    #print('# indices MFs',MF_subgroup,':', len(indices))
    return indices

def shuffling(node_GC, node_MF, edges, gc_color_map, eval_shf_func=False): #shuffle the synapses between target GC groups and MF groups 

    #print('{:<10}'.format('1 IDs'), id(edges))
    if eval_shf_func:
        print('edge before', [eg for eg in edges if gc_color_map[node_GC.index(eg[1])]==GC_colors[0]])
    total_early_gc_edges2=[]
    for mf in node_MF: # For selected MFs (sub) group
        early_gc_edges=[]
        for edge in [eg for eg in edges if eg[0]==mf]: # For whole edges with the mf            
            if gc_color_map[node_GC.index(edge[1])]==GC_colors[0]: #if synapsed with early GCs                
                early_gc_edges.append(edge)                               
        total_early_gc_edges2.append(early_gc_edges) #early GC edges list per MF
    if eval_shf_func:
        print('Original edges', np.shape(edges))
        for i in total_early_gc_edges2:
            print(i)
    #edges=np.copy(edges)
    #print('{:<10}'.format('2 IDs'), id(edges))
    edges=np.array(edges)
    #print('{:<10}'.format('3 IDs'), id(edges))
    #print('total_early_gc_edges2', total_early_gc_edges2)
    for edge_list in total_early_gc_edges2:
        if len(edge_list)>0:
            for edge in edge_list:
                edges=np.delete(edges, np.where(np.all(edges==edge, axis=1)), axis=0)
    if eval_shf_func: print('Deleted edges', np.shape(edges), edges)
        
    np.random.shuffle(total_early_gc_edges2)
    #print('total_early_gc_edges2_shuffled', total_early_gc_edges2)
    #print('{:<10}'.format('4 IDs'), id(edges))

    if eval_shf_func:
        print('shuffled')
        for i in total_early_gc_edges2:
            print(i)     
    
    for ind, edge_list in enumerate(total_early_gc_edges2):
        if len(edge_list)>0:
            for edge in edge_list:                
                edge[0]='M%s'%ind
                #print(np.shape(edges))
                #print(np.shape([edge]))
                #edges=np.append(edges, [edge], axis=0)
    if eval_shf_func:
        print('sorted')
        for i in total_early_gc_edges2:
            print(i)     
    
    rebuilt_edges=[]    
    for ind, edge_list in enumerate(total_early_gc_edges2):
        if len(edge_list)>0:
            for edge in edge_list:                
                rebuilt_edges.append(edge)
    if eval_shf_func:
        print('reshaped')
        print(rebuilt_edges)
    edges=np.concatenate((rebuilt_edges, edges), axis=0)    
    edges=np.ndarray.tolist(edges)
    
    if eval_shf_func: 
        print('Shuffled edges', np.shape(edges))    
        print('early edge final', [eg for eg in edges if gc_color_map[node_GC.index(eg[1])]==GC_colors[0]])
    #sys.exit()
    #print('{:<10}'.format('5 IDs'), id(edges))
    #sys.exit()
    return edges

def print_connectivity_stats(total_early, total_mid, total_late, MF_subgroup, node_MF):
    avg_early = np.around(total_early/len(node_MF),3)
    avg_mid = np.around(total_mid/len(node_MF),3)
    avg_late = np.around(total_late/len(node_MF), 3)    
    print('For ', MF_subgroup, 'MF subgroup,', '# node_MF', len(node_MF))
    print('# dend. from Early GC:', total_early, 'Average #Early GC per MF:', avg_early)
    print('# dend. from Mid GC:', total_mid, 'Average #Mid GC per MF:', avg_mid)
    print('# dend. from Late GC:', total_late, 'Average #Late GC per MF:', avg_late) 

def draw_network(gc_color_map, mf_color_map, edges, node_GC, node_MF, edge_color_map=None):

    inh=False
    inh2=False
    inh3=False
    if inh and not inh3:
        inhibit_node=['inhb']
        inhib_edges=[[gc,'inhb'] for gc in node_GC]
        
        
        #inhib_edges=[['inhb', gc] for gc in node_GC]

        if not inh2:
            nodes= node_GC + node_MF +inhibit_node
            color_map=gc_color_map+mf_color_map + ['k']

            edges=edges+inhib_edges
            edge_color_map=edge_color_map + ['k']*len(node_GC)
        elif inh2 :
            nodes= inhibit_node+ node_GC + node_MF
            color_map=['k']+gc_color_map+mf_color_map

            edges=inhib_edges+edges
            edge_color_map=['k']*len(node_GC)+edge_color_map

        #print('nodes\n', nodes)
        #print('edges\n', edges)
        #print('edge_color_map\n', edge_color_map)
    
    else:
        nodes= node_GC + node_MF
        color_map=gc_color_map+mf_color_map
        edge_color_map=edge_color_map
    
    if inh3:
        inh4=False
        inhibit_node=['inhb']
        inhib_edges1=[[gc,'inhb'] for gc in node_GC]
        inhib_edges2=[['inhb', gc] for gc in node_GC]

        i_nodes= inhibit_node+ node_GC
        i_color_map=['silver']+gc_color_map        

        i_edges=inhib_edges1+inhib_edges2
        i_edge_color_map=['k']*len(node_GC)+['slategrey']*len(node_GC)

        if inh4:
            inhib_edges2=[['inhb', mf] for mf in node_MF]

            i_nodes= inhibit_node+ node_GC +node_MF
            i_color_map=['silver']+gc_color_map+mf_color_map

            i_edges=inhib_edges1+inhib_edges2
            i_edge_color_map=['k']*len(node_MF)+['slategrey']*len(node_GC)
        

        H = nx.MultiDiGraph()

        H.add_nodes_from(i_nodes)
        H.add_edges_from(i_edges)
        #H.add_edges_from(inhib_edges2, color="pink")
        #H.add_edges_from(inhib_edges1, color="k")

        i_pos={'inhb': (0.9, 2)}
        i_pos.update(dict(zip(node_GC, zip([1]*len(node_GC), np.arange(len(node_GC))*0.1  ))))

        if inh4:
            expansion_rate=len(node_GC)/len(node_MF)
            i_pos.update(dict(zip(node_MF, zip([0]*len(node_MF), np.arange(len(node_MF))*0.1*expansion_rate ))) )

    


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

    if inh:
        inhb_pos={'inhb': (1.3, -0.3)}
        pos.update(inhb_pos)
        #print('inh updated pos:\n', pos)
    #from networkx.algorithms import bipartite
    #G = nx.Graph()
    #G.add_nodes_from(node_GC, bipartite=0)
    #print(G.node)
    #
    #G.add_nodes_from(node_MF, bipartite=1)
    #
    #
    #G.add_edges_from(edges)
#
    #print(G.edges)
    #nx.is_connected(G)  
    
    

    if len(edges)>2000:
        raise Exception("Exception you set by yourself!  Too many edges to plot can't see all the edges ")
    #nx.draw(G, pos=nx.bipartite_layout(G, node_MF),node_size=140,\
    #nx.draw(G, pos=pos,node_size=140,\
    #print(edges, edge_color_map)
    #print('before drawing    color mapt size:', len(edges), 'edge size:', len(edge_color_map))
    #print('edges\n', edges)
    #print('G.edges\n',[[list(edg)[1],list(edg)[0]] for edg in G.edges])

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

    

    #print('nodes', G.node)
    
    
    #print(inhib_edges, '\n inhib pos:', inhb_pos)
    #sys.exit()
    #G.add_edges_from(inhib_edges)
    
    
    
    

    if not edge_color_map==None:        
        nx.draw(G, pos=pos,\
            node_color=color_map, edge_color=edge_color_map, with_labels=True)

    else:
        nx.draw(G, pos=pos,\
            node_color=color_map, with_labels=True)
         #scale= 2, node_color=color_map, with_labels=True)

    if inh3:
        #F=nx.compose(G, H)
        #nx.draw(H)
        nx.draw(H, pos=i_pos,\
            node_color=i_color_map, edge_color=i_edge_color_map, connectionstyle='arc3, rad = 1', with_labels=True)
            #node_color=i_color_map, connectionstyle='arc3, rad = 1', with_labels=True)
        print('inh3')
    print('Drawing the network')
    plt.show()

def neuralnet4(node_GC, node_MF, edges, MF_subgroup='total', \
            plot_frequency_dist=False, draw_net=False, shuffle=False, \
                rt_ratio=False, print_stat=True, mf_color_map=None, gc_color_map=None):
    #if shuffle: print('--unshfld-------------')
    node_GC_cp=node_GC.copy()
    node_MF_cp=node_MF.copy()
    edges_cp  =  edges.copy()
    same=True
    if not edges==edges_cp: same=False
    #print('1-ed_list', edges)
    #print('{:<10}'.format('strtng IDs'), id(edges), id(edges_cp), 'value equal?', same, 'len cp edges', len(edges_cp))
    
    if mf_color_map==None:
        mf_color_map = migration_timing(node_MF_cp, MF_colors)    
        gc_color_map = migration_timing(node_GC_cp, GC_colors)
    
    indices = target_MF_indexing(node_MF_cp, MF_subgroup)
    node_MF_cp, mf_color_map = node_pruning(indices, mf_color_map, node_MF_cp)
    edges_cp                 = edge_pruning(node_MF_cp, edges_cp)
    if not  edges==edges_cp: same=False
    #print('2-ed_list', edges)
    #print('{:<10}'.format('prunn IDs'), id(edges), id(edges_cp), 'value equal?', same, 'len cp edges', len(edges_cp))

    if shuffle: edges_cp = shuffling(node_GC_cp, node_MF_cp, edges_cp, gc_color_map)
    if not  edges==edges_cp: same=False
    #print('{:<10}'.format('shffln IDs'), id(edges), id(edges_cp), 'value equal?', same, 'len cp edges', len(edges_cp))
    #print('3-ed_list', edges)
    if draw_net: draw_network(gc_color_map, mf_color_map, edges_cp, node_GC_cp, node_MF_cp) 

    ratios=frequency_distribution(node_MF_cp, node_GC_cp, edges_cp, gc_color_map)    
    if plot_frequency_dist: plot_degree_dist(ratios)        
    if not  edges==edges_cp: same=False
    #print('{:<10}'.format('freq dst'), id(edges), id(edges_cp), 'value equal?', same, 'len cp edges', len(edges_cp))
    
    total_early, total_mid, total_late = frecuency_MF(node_MF_cp, node_GC_cp, edges_cp, gc_color_map)
    if print_stat: print_connectivity_stats(total_early, total_mid, total_late, MF_subgroup, node_MF_cp)
    #print('variance:', np.var(ratios))    
    #if rt_ratio: return ratios
    #print('4-ed_list', edges)
    return ratios


def softmax(valueset): 
    #return np.exp(valueset)/np.exp(valueset).sum()
    return valueset/valueset.sum() # no Exponential

def random_networks(Num_GCs, Num_MFs, previous_MF_index=0, degree_connectivity=4, num_modules=2, degree_of_modularity=0):    
    expansion_rate=Num_GCs/Num_MFs
    node_GC=[]
    node_MF=[]
    previous_GC_index=int(previous_MF_index*expansion_rate)
    for gc in range(Num_GCs):        
        node_GC.append('G%s'%(gc+previous_GC_index))
    for mf in range(Num_MFs):        
        node_MF.append('M%s'%(mf+previous_MF_index))

    ed_list=[]
    #module_size=int(Num_GCs/num_modules)
    #module_sizes=[int(Num_MFs/num_modules)]*(num_modules-1)+[Num_MFs-int(Num_MFs/num_modules)*(num_modules-1)]
    for i in range(Num_GCs):
        if degree_of_modularity>0: #Syn. partner selection for modularity, preferential to close indices
            #prob_basis_val=np.exp(-1*(abs(i/expansion_rate-np.arange(Num_MFs))))            
            prob_basis_val=np.exp(-1*(abs((i/expansion_rate-np.arange(Num_MFs))*degree_of_modularity/100)))            
            modular_prob_assign = softmax(prob_basis_val)            
            synapse_partner=np.random.choice(Num_MFs, size=degree_connectivity, replace=False, p=modular_prob_assign)
            #print('prob_basis_val:', np.round(prob_basis_val,3))
            #print('modular_prob_assign:', np.round(modular_prob_assign,3))
            #print('synapse_partner', synapse_partner)
            #sys.exit()


        else: synapse_partner=np.random.choice(Num_MFs, size=degree_connectivity, replace=False)

        syns=sorted(synapse_partner)
        for j in syns:            
            ed_list.append(['M%s'%(j+previous_MF_index), 'G%s'%(i+previous_GC_index)])

    print('# GCs', len(node_GC))
    print('# MFs', len(node_MF))
    print('# edges(synapses)', len(ed_list))
    
    return node_GC, node_MF, ed_list




