import numpy as np
import sys
from matplotlib import pyplot as plt

UNIT_SCALE=1e-03 ##Unit scales = milli
TIME_UNIT=UNIT_SCALE
PARAM_SCALING=1/UNIT_SCALE #Constants for Time, Conductances, Capacitances are up scaled for computation, Potentials remain same

#lists for [a,d] :amplitudes, decay times pairs and, at the last, rho: common rise times
EXT_dlt=np.array([1.*1e-03, 1.*1e-03])          *PARAM_SCALING  #Time terms are upscaled by milli
EXT_rho=0.3*1e-03                               *PARAM_SCALING
#INH_dlt=np.array([1.*1e-03, 1.*1e-03])         *PARAM_SCALING
#INH_rho=0.3*1e-03                              *PARAM_SCA
EXT_amp=np.array([1.*1e-09, 0.3*1e-09])         *PARAM_SCALING*15  #Amp & conductances upscaled by milli
#INH_amp=np.array([1.*1e-09, 0.3*1e-09])         *PARAM_SCALING*1



membrane_capacitance=3.22*1e-12                 *PARAM_SCALING #C_m
leak_conductance=1.06*1e-09                     *PARAM_SCALING #G_m
INH_conductance=0.483*1e-09                     *PARAM_SCALING #G_GABA

leak_reversal_potential=-79.9*1e-03             #*PARAM_SCALING #E_m
INH_reversal_potential =-79.1*1e-03              #*PARAM_SCALING #E_INH (tonic)
EXT_reversal_potential=0                        #*PARAM_SCALING #E_EXT
reset_membrane_potential=-63*1e-03              #*PARAM_SCALING #V_r
activation_threshold=-40*1e-03                  #*PARAM_SCALING #V_t

refractory_interval=2*1e-03                     *PARAM_SCALING #tau



r_EXT=0.2
r_INH=0.2

learning_rate=1e-02
WINDOW_INTERVAL=40
def STDP_window_function(time_interval):    
    if time_interval>0: return learning_rate*np.exp(-time_interval/20)
    elif time_interval<0: return learning_rate*-np.exp(time_interval/20)

class Spiking_cells:
    def __init__(self,num_dend=1): 
        #each refers to Synaptic weghts of each of the components
        self.p_EXT_U=[]
        self.p_EXT_R=[]
        self.Synaptic_strenth=[] # for weight update for STDP, initialized as 1

        for dend in range(num_dend):
            self.p_EXT_U.append(r_EXT)
            self.p_EXT_R.append(1)
            self.Synaptic_strenth.append(1)

        self.num_dend=num_dend
        self.membrane_potential= reset_membrane_potential
        self.MEM_POT_NOSTP= reset_membrane_potential

    def p_EXT(self, ind_dend): return self.p_EXT_U[ind_dend]*self.p_EXT_R[ind_dend]
    

    def PLASTICITY(self, ind_dend, SPIKE):
        p_EXT_U=self.p_EXT_U[ind_dend]
        p_EXT_R=self.p_EXT_R[ind_dend]
        if SPIKE:       
            self.p_EXT_U[ind_dend]+=r_EXT*(1-p_EXT_U)
            self.p_EXT_R[ind_dend]=p_EXT_R*(1-p_EXT_U)
        else:
            self.p_EXT_U[ind_dend]-=(p_EXT_U-r_EXT)/50. #Potentiation Recovery time constants
            #self.p_INH_U[ind_dend]-=(p_INH_U-r_INH)/100. 
            self.p_EXT_R[ind_dend]-=(p_EXT_R-1)/100.     #Depression Recovery time constants
            #self.p_INH_R[ind_dend]-=(p_INH_R-1)/100.     #10,20,100,50


    def EXT_conductance_evaluation(self, t, indEXT_dend):
        wave=np.zeros(len(t))
        for ind, _ in enumerate(EXT_amp):
            wave+=EXT_amp[ind]*(np.exp(-t/EXT_dlt[ind])-np.exp(-t/EXT_rho))            
        return self.Synaptic_strenth[indEXT_dend]*self.p_EXT(indEXT_dend)*wave, r_EXT*wave

    def rate_of_change_membrane_voltage(self, Current_Mem_Pot, current_conductance_EXT, \
                                                               current_conductance_INH, STP=True):
                                                               #STP=True):

        dV=-1/membrane_capacitance*(\
                leak_conductance*(Current_Mem_Pot-leak_reversal_potential)\
                +INH_conductance*(Current_Mem_Pot-INH_reversal_potential)\
                +current_conductance_EXT *(Current_Mem_Pot-EXT_reversal_potential)\
                )
        
        if STP: self.membrane_potential+=dV*TIME_UNIT  # change per milli second
        else: self.MEM_POT_NOSTP+=dV*TIME_UNIT



def simulator(NUM_MFs, spike_pttn_per_bin, time_step, SPK_GC=None, RECORD_WEIGHT_CHANGE=False):
    TIME_RANGE=np.linspace(0,TIME_UNIT*time_step, time_step)
    '''Input spike plot'''
    #raster_plot(spike_pttn_per_bin, [0])
    #sys.EXTit()

    if SPK_GC==None: SPK_GC=Spiking_cells(num_dend=NUM_MFs)
    else: SPK_GC=SPK_GC
    ''' For recording the values of variables as np.zeros'''
    EXT_Conductance=np.zeros((NUM_MFs, time_step))
    INH_Conductance=np.zeros((NUM_MFs, time_step))
    mem_voltage=np.zeros(time_step)
    REFRACTORY_COUNT=0
    p_ofEXT=np.zeros((NUM_MFs,time_step))
    #p_ofINH=np.zeros((NUM_MFs,time_step))

    EXT_NO_STP=np.zeros((NUM_MFs, time_step))
    mem_voltage2=np.zeros(time_step)
    REFRACTORY_COUNT2=0

    Num_OUTPUT_SPIKE=0
    Spike_record=np.zeros(time_step)
    if RECORD_WEIGHT_CHANGE:
        weight_change_record=np.zeros((SPK_GC.num_dend, time_step))

    ''' Simulation loop start '''
    for t, time in enumerate(TIME_RANGE):
        mem_voltage[t]=SPK_GC.membrane_potential

        for dend in range(SPK_GC.num_dend):
            p_ofEXT[dend][t]=SPK_GC.p_EXT(dend)
            #p_ofINH[dend][t]=  SPK_GC.p_INH(dend)


        if not REFRACTORY_COUNT==0:  #Refractory Period        
            SPK_GC.membrane_potential=reset_membrane_potential
            REFRACTORY_COUNT-=1
        else:        
            for dend in range(SPK_GC.num_dend):
                if spike_pttn_per_bin[dend][t]==1: #if there's a spike input for a dendrite
                    EXT_waveform_of_a_spike, _ =SPK_GC.EXT_conductance_evaluation(np.arange(time_step-t), dend)
                    #INH_waveform_of_a_spike, _ =SPK_GC.INH_conductance_evaluation(np.arange(time_step-t), dend)
                    EXT_Conductance[dend][t:]+=EXT_waveform_of_a_spike
                    INH_Conductance[dend][t]+=INH_conductance*(SPK_GC.membrane_potential-INH_reversal_potential)

                    for t_backward in [t-t_ for t_ in range(1, WINDOW_INTERVAL+1) if t-t_>=0]:                        
                        #if t_backward<0: raise Exception("Wrong time steps")
                        if Spike_record[t_backward]: 
                            SPK_GC.Synaptic_strenth[dend]+=STDP_window_function(t_backward-t)
                            #print('delta w in depression:', STDP_window_function(t_backward-t))



                SPK_GC.rate_of_change_membrane_voltage(SPK_GC.membrane_potential,
                                                     EXT_Conductance[dend][t], INH_Conductance[dend][t], STP=True)            
                                                     #EXT_Conductance[dend][t], STP=True)

            if SPK_GC.membrane_potential>=activation_threshold:
                #print('SPIKE at time', t)
                Spike_record[t]+=1
                for t_backward in [t-t_ for t_ in range(1, WINDOW_INTERVAL+1) if t-t_>=0]:
                    #if t_backward<0: raise Exception("Wrong time steps")
                    for dend in range(SPK_GC.num_dend):
                        if spike_pttn_per_bin[dend][t_backward]==1:
                            SPK_GC.Synaptic_strenth[dend]+=STDP_window_function(t-t_backward)
                            #print('delta w in potentiation:', STDP_window_function(t-t_backward))                        

                Num_OUTPUT_SPIKE+=1
                REFRACTORY_COUNT=refractory_interval
                SPK_GC.membrane_potential=0          # 0mv Represent spikes (choosed as a arbitrary high value)   
            elif SPK_GC.membrane_potential<=leak_reversal_potential:
                print('Limiting minimum potential by leak revers. Pot. at time', t)
                SPK_GC.membrane_potential=leak_reversal_potential

        for dend in range(SPK_GC.num_dend):
            SPK_GC.PLASTICITY(dend, spike_pttn_per_bin[dend][t]) #Short term plasticity
            if RECORD_WEIGHT_CHANGE: weight_change_record[dend][t]=SPK_GC.Synaptic_strenth[dend]


        # No STP simulation for Comparison-------------------------------
        '''
        mem_voltage2[t]=SPK_GC.MEM_POT_NOSTP

        if not REFRACTORY_COUNT2==0:  #Refractory Period        
            SPK_GC.MEM_POT_NOSTP=reset_membrane_potential
            REFRACTORY_COUNT2-=1    
        else:        
            for dend in range(SPK_GC.num_dend):
                if spike_pttn_per_bin[dend][t]==1:
                    _, EXT_WAVE_No_STP =SPK_GC.EXT_conductance_evaluation(np.arange(time_step-t), dend)
                    _, INH_WAVE_No_STP =SPK_GC.INH_conductance_evaluation(np.arange(time_step-t), dend)
                    EXT_NO_STP[dend][t:]+=EXT_WAVE_No_STP
                    INH_NO_STP[dend][t:]+=INH_WAVE_No_STP

                SPK_GC.rate_of_change_membrane_voltage(SPK_GC.MEM_POT_NOSTP, 
                                                    EXT_NO_STP[dend][t],INH_NO_STP[dend][t], STP=False)

            if SPK_GC.MEM_POT_NOSTP>=activation_threshold:
                print('SPIKE at time', t, 'For No STP')
                REFRACTORY_COUNT2=refractory_interval
                SPK_GC.MEM_POT_NOSTP=0   '''
    #print('Total spikes arrived:', np.sum(spike_pttn_per_bin), 'Total spike output:', Num_OUTPUT_SPIKE)    
    #return SPK_GC, EXT_Conductance, INH_Conductance, mem_voltage, p_ofEXT, p_ofINH, Num_OUTPUT_SPIKE, Spike_record
    if RECORD_WEIGHT_CHANGE: plot_STDP(SPK_GC.num_dend, time_step, weight_change_record)

    return SPK_GC, EXT_Conductance, INH_Conductance, mem_voltage, p_ofEXT, Num_OUTPUT_SPIKE, Spike_record

def plot_cell_dynamics(NUM_MFs, time_step, EXT_Conductance, INH_Conductance, mem_voltage):
    #raster_plot(spike_pttn_per_bin, [0])
    Plotting_Scale=1e-03 # of conducntance    

    fig, axs = plt.subplots(2+NUM_MFs, gridspec_kw={'height_ratios': [1]*(NUM_MFs)+[0.1]+[2]})
    for ind_mf in range(NUM_MFs):
        axs[ind_mf].plot(np.arange(time_step), EXT_Conductance[ind_mf]*Plotting_Scale, 'red', label="EXT")
        axs[ind_mf].plot(np.arange(time_step), INH_Conductance[ind_mf]*Plotting_Scale, 'g', label="INH")
        axs[ind_mf].axhline(y=630*1e-12*PARAM_SCALING*Plotting_Scale, color='c', linestyle=':', label='Peak amplitude') #of both condunctances        
        
        if ind_mf==3: axs[ind_mf].legend()

    axs[0].set_title('Conductances')

    #x,y = np.argwhere(spike_pttn_per_bin == 1).T
    #axs[NUM_MFs].scatter(y, x, marker='|', label='SPIKE arrivals')
    #axs[NUM_MFs].legend()

    axs[NUM_MFs+1].plot(np.arange(time_step), mem_voltage, label='Mem. Pot.')
    #axs[NUM_MFs].plot(np.arange(time_step), mem_voltage2, label='NO STP')
    axs[NUM_MFs+1].axhline(y=reset_membrane_potential, color='r', linestyle=':', label='Reset_pot')
    axs[NUM_MFs+1].axhline(y=activation_threshold, color='b', linestyle=':', label='Thrshld')
    axs[NUM_MFs+1].set_ylabel("""Mmbrn Ptnt'l level""")
    axs[NUM_MFs+1].legend()
    plt.show()

def plot_STP(NUM_MFs, time_step, p_ofEXT, p_ofINH=None):
    # For P development on condunctance terms of STP
    print(np.shape(np.arange(time_step)), np.shape(p_ofEXT))
    fig, axs = plt.subplots(NUM_MFs, gridspec_kw={'height_ratios': [1]*(NUM_MFs)})
    for ind_mf in range(NUM_MFs):
        axs[ind_mf].plot(np.arange(time_step), p_ofEXT[ind_mf], color='r', label="EXT")
        #axs[ind_mf].plot(np.arange(time_step), p_ofINH[ind_mf],color='g', label="INH")
        axs[ind_mf].axhline(y=r_EXT, color='r', linestyle=':', label='Init_EXT') 
        #axs[ind_mf].axhline(y=r_INH,color='g', linestyle=':', label='Init_INH')
        if ind_mf==NUM_MFs-1: axs[ind_mf].legend()


    axs[0].set_title('P development')
    plt.show()
    sys.exit()

def plot_STDP(NUM_DEND, time_step, RECORD_WEIGHT):
    # For P development on condunctance terms of STP
    print(np.shape(np.arange(time_step)), np.shape(RECORD_WEIGHT))
    fig, axs = plt.subplots(NUM_DEND, gridspec_kw={'height_ratios': [1]*(NUM_DEND)})
    for ind_mf in range(NUM_DEND):
        axs[ind_mf].plot(np.arange(time_step), RECORD_WEIGHT[ind_mf], color='r', label="RECORD_WEIGHT")        
        axs[ind_mf].axhline(y=1, color='r', linestyle=':', label='Init_Weight')
        axs[ind_mf].axhline(y=1+learning_rate, color='k', linestyle=':', label='Upper bound')
        axs[ind_mf].axhline(y=1-learning_rate, color='k', linestyle=':', label='Lower bound')
        if ind_mf==NUM_DEND-1: axs[ind_mf].legend()


    axs[0].set_title('Weight development')
    plt.show()
    #sys.exit()
