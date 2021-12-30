#from function import *
from vpython import *
import numpy as np
import math
import matplotlib.pyplot as plt 
import sys
import time

''' 
#obsolete functions
def GC_destination(current_depth_IGL):
    strata_subdivision=['top', 'mid', 'bottom']
    prob_destination=np.random.choice(strata_subdivision, p=[0.06,0.12,0.82])        
    if prob_destination=='top':
        transit_distance=np.random.choice(int(current_depth_IGL*0.44)+1)
    elif prob_destination=='mid':
        transit_distance=np.random.choice(int(current_depth_IGL*0.44)+1)+current_depth_IGL*0.44
    elif prob_destination=='bottom':
        transit_distance=np.random.choice(int(current_depth_IGL*0.12)+1)+current_depth_IGL*0.88        
    if transit_distance<radius_GC:
        transit_distance=radius_GC
    #print('transit_distance: ', transit_distance)        
    return transit_distance
    
'''

class time_box:
    def __init__(self, pos, text):
        self.pos=pos
        self.text=text

class class_time_display:
    def __init__(self,area_length, height_PCL, time_division, time_exhibit='Embryonic stage',\
                 counter=0, vpython=True):
        #self.time_exhibit=time_exhibit
        self.counter=counter  #-(24*7* time_division)
        self.day=0
        self.start=False
        if vpython:
            self.display=label(pos=vector(area_length*2,height_PCL/2,0), text=time_exhibit)
        else:
            self.display=time_box(pos=vector(area_length*2,height_PCL/2,0), text=time_exhibit)

        self.time_division=time_division

    def time_count(self):
        if self.start==False:
            self.start=True
        else:
            self.counter+=1
            if (self.counter%(24*self.time_division)) ==0:
                self.day+=1
                self.display.text=('P%d'%self.day)
                #if self.day<15:
                #print('Day %d, h %d'%(self.day, self.counter/self.time_division))
                print('P%d'%self.day)
            if self.counter==20*24*self.time_division:
                print('Simulation time end, after processing....')


class layer_box:
    def __init__(self, pos):
        self.pos=pos

class moving_layer:
    def __init__(self, area_length, height_PCL, area_width, vpython=True):
        length=area_length/2
        width=area_width/2
        if vpython:
            self.expanding_border=box(pos=vector(length,height_PCL,width),\
                                size=vector(area_length,0.1,area_width),\
                                color=color.magenta)
        else:
            self.expanding_border=layer_box(pos=vector(length,height_PCL,width))
        
    def expand(self, depth):
        self.expanding_border.pos.y=depth   
        #self.expanding_border.pos.y-=height_PCL/(24*20)   


def layer_division(area_length, height_PCL, area_width, height_WM=0, labeling=False, vpython=True):
    length=area_length/2
    width=area_width/2
    if vpython:
        Intersection1=box(pos=vector(length,height_PCL,width), size=vector(area_length,0.1,area_width), color=color.blue)  ## btwn PCL & IGL    
        Intersection2=box(pos=vector(length,height_WM ,width), size=vector(area_length,0.1,area_width))  ## btwn IGL & WM
    else:
        Intersection1=layer_box(pos=vector(length,height_PCL,width))  ## btwn PCL & IGL    
        Intersection2=layer_box(pos=vector(length,height_WM ,width))  ## btwn IGL & WM


    if labeling==True and vpython==True:
        label(pos=vector(length,height_PCL,width), text='The bottom of PCL')
        label(pos=vector(length,height_WM,width), text='The bottom of IGL')


# Data management
import os
import numpy as np

def data_save(data_name, data):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    directory = cur_path+'/data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_dir=directory+'/'+data_name
    np.save(save_dir, data)
    print('data', data_name,'saved at',save_dir)
    
def data_load(data_name):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    directory = cur_path+'/data'
    #load= np.load(directory+'/'+ data_name+'.npy')
    load= np.load(directory+'/'+ data_name+'.npy', allow_pickle=True)
    return load


import pickle

def pickle(dataname, data):
    pickle.dump(data, open(dataname+".pkl", "w"))

def unpickle(dataname):
    pickle.load(open(dataname+".pkl"))

def exponentiated_exponential_function(x, x_var=0, time_division=1, lambda_=3.5, alpha_=5.5, sigma=100, k=100, graph=False):
    #with sigma variation    
    # Function range = 0.5*sigma
    if x<=x_var*time_division:
        return 0
    eef=k*alpha_*lambda_\
        *np.exp(-lambda_*(x-x_var*time_division)/(sigma*time_division))\
        *((1-np.exp(-lambda_*(x-x_var*time_division)/(sigma*time_division)))**(alpha_-1))    
    #if graph:
    #    plt.plot(x, eef)
    #    plt.show()
    return eef

def MF_activity_coordinator(time_step, time_division):
    t=200
    interval=50
    Early_MF_activity=exponentiated_exponential_function(time_step, \
                x_var=t-interval, time_division=time_division)+1
    Mid_MF_activity=exponentiated_exponential_function(time_step, \
                x_var=t, time_division=time_division )+1
    Late_MF_activity=exponentiated_exponential_function(time_step, \
                x_var=t+interval, time_division=time_division)+1
    MF_activities=[Early_MF_activity, Mid_MF_activity, Late_MF_activity]
    return MF_activities


def MF_activity_coordinator_separatedactivities(time_step, time_division):
    t=220
    k= 140
    s=60
    Early_MF_activity=exponentiated_exponential_function(time_step, \
                x_var=t-100, time_division=time_division, sigma=s, k=k)+1
    Mid_MF_activity=exponentiated_exponential_function(time_step, \
                x_var=t, time_division=time_division, sigma=s/3, k=k)+1
    Late_MF_activity=exponentiated_exponential_function(time_step, \
                x_var=t+50, time_division=time_division, sigma=s, k=k)+1
    MF_activities=[Early_MF_activity, Mid_MF_activity, Late_MF_activity]
    return MF_activities

"""
Blinking
"""
def blinking(cells, lamda, Time='Fixed'):    
    print('Blinking')
    beta=1/lamda
    while True:
        delta_t = np.random.exponential(beta)
        time.sleep(delta_t)
        #print('dt:', delta_t)
        #firing_cell=np.random.choice(cells, 10)
        firing_cell=np.random.choice(cells)
        firing_cell.rosette.color=color.cyan
        time.sleep(0.05)
        firing_cell.rosette.color=firing_cell.color


def blinking2(cells, lamda, color):        
    beta=1/lamda
    delta_t = np.random.exponential(beta)
    time.sleep(delta_t)
    #print('dt:', delta_t)
    firing_cell=np.random.choice(cells, int(lamda))
    #firing_cell=np.random.choice(cells)
    for i in firing_cell:
        i.rosette.color=color
    time.sleep(0.05)
    for i in firing_cell:
        i.rosette.color=i.color