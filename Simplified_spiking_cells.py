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

learning_rate=1e-03
WINDOW_INTERVAL=40

class Spiking_cells:
    def __init__(self,num_dend=1): 
        #each refers to Synaptic weghts of each of the components
        self.p_EXT_U=    [r_EXT]*num_dend
        self.p_EXT_R=    [1]*    num_dend
        self.STDP_weight=[1]*num_dend # for weight update for STDP, initialized as 1

        self.num_dend=num_dend
        self.membrane_potential= reset_membrane_potential
        self.MEM_POT_NOSTP= reset_membrane_potential

    def Synapse_Pruning(self, NUM_PRUN=1):        
        ind_prun=np.where(self.STDP_weight==np.amin(self.STDP_weight))[0][:NUM_PRUN]

        self.STDP_weight=np.delete(self.STDP_weight, ind_prun)
        self.p_EXT_U=np.delete(self.p_EXT_U, ind_prun)
        self.p_EXT_R=np.delete(self.p_EXT_R, ind_prun)
        self.num_dend-=NUM_PRUN
        return ind_prun

    def STDP_window_function(self, time_interval, ind_dend):
        bounding=np.exp(-abs(1-self.STDP_weight[ind_dend]))
        #bounding=1
        sign= 1 if time_interval>0 else -1 if time_interval<0 else 0
        if sign==0: raise Exception("Interval for STDP window is 0")
        window_output=sign*np.exp(-sign*time_interval/40)
        #if window_output>0: window_output*=0.7
        return learning_rate*bounding*window_output
        
    
    def p_EXT(self, ind_dend): return self.p_EXT_U[ind_dend]*self.p_EXT_R[ind_dend]
    

    def PLASTICITY(self, ind_dend, SPIKE):
        EXT_U=self.p_EXT_U[ind_dend]
        EXT_R=self.p_EXT_R[ind_dend]
        
        if SPIKE:       
            #print('before plasticity:',EXT_U, EXT_R, self.p_EXT_U[ind_dend], self.p_EXT_R[ind_dend])
            self.p_EXT_U[ind_dend]+=r_EXT*(1-EXT_U)
            self.p_EXT_R[ind_dend]=EXT_R*(1-EXT_U)
            #print('After plasticity:',EXT_U, EXT_R, self.p_EXT_U[ind_dend], self.p_EXT_R[ind_dend])
            #sys.exit()
        else:
            self.p_EXT_U[ind_dend]-=(EXT_U-r_EXT)/50. #Potentiation Recovery time constants
            #self.p_INH_U[ind_dend]-=(p_INH_U-r_INH)/100. 
            self.p_EXT_R[ind_dend]-=(EXT_R-1)/100.     #Depression Recovery time constants
            #self.p_INH_R[ind_dend]-=(p_INH_R-1)/100.     #10,20,100,50
        


    def EXT_conductance_evaluation(self, t, ind_dend):
        wave=np.zeros(len(t))
        for ind, _ in enumerate(EXT_amp):
            wave+=EXT_amp[ind]*(np.exp(-t/EXT_dlt[ind])-np.exp(-t/EXT_rho))
        return self.STDP_weight[ind_dend]*self.p_EXT(ind_dend)*wave, r_EXT*wave

    def rate_of_change_membrane_voltage(self, current_conductance_EXT,
                                                               STP=True):
        Current_Mem_Pot=self.membrane_potential
        dV=-1/membrane_capacitance*(\
                leak_conductance*(Current_Mem_Pot-leak_reversal_potential)\
                +INH_conductance*(Current_Mem_Pot-INH_reversal_potential)\
                #+INH_conductance\
                +current_conductance_EXT *(Current_Mem_Pot-EXT_reversal_potential)\
                )
        
        #if STP: self.membrane_potential+=dV*TIME_UNIT  # change per milli second
        #else: self.MEM_POT_NOSTP+=dV*TIME_UNIT
        return dV*TIME_UNIT



def simulator(NUM_MFs, spike_pttn_per_bin, time_step, SPK_GC, RECORD_WEIGHT_CHANGE=False):
    TIME_RANGE=np.linspace(0,TIME_UNIT*time_step, time_step)
    
    '''Input spike plot'''
    #raster_plot(spike_pttn_per_bin, [0])
    
    ''' For recording the values of variables as np.zeros'''
    EXT_Conductance=np.zeros((NUM_MFs, time_step))
    INH_Conductance=np.ones((NUM_MFs, time_step))*INH_conductance
    mem_voltage=np.zeros(time_step)
    REFRACTORY_COUNT=0
    record_p_ofEXT=np.zeros((NUM_MFs,time_step))
    #p_ofINH=np.zeros((NUM_MFs,time_step))

    EXT_NO_STP=np.zeros((NUM_MFs, time_step))
    mem_voltage2=np.zeros(time_step)
    REFRACTORY_COUNT2=0
    
    Spike_record=np.zeros(time_step)
    if RECORD_WEIGHT_CHANGE:
        record_LTP=np.zeros((SPK_GC.num_dend, time_step))
        record_LTD=np.zeros((SPK_GC.num_dend, time_step))
        STDP_weight_change_record=np.zeros((SPK_GC.num_dend, time_step))

    ''' Simulation loop start '''
    for t, time in enumerate(TIME_RANGE):
        '''Recording'''
        mem_voltage[t]=SPK_GC.membrane_potential
        for dend in range(SPK_GC.num_dend): 
            record_p_ofEXT[dend][t]=SPK_GC.p_EXT(dend)
            if RECORD_WEIGHT_CHANGE: STDP_weight_change_record[dend][t]=SPK_GC.STDP_weight[dend]
            #p_ofINH[dend][t]=  SPK_GC.p_INH(dend)
        

        '''Computing'''
        if not REFRACTORY_COUNT==0:  #Refractory Period        
            SPK_GC.membrane_potential=reset_membrane_potential
            REFRACTORY_COUNT-=1
        else: # Compute, if not Refractory period
            dV=0
            for dend in range(SPK_GC.num_dend):            
                if spike_pttn_per_bin[dend][t]==1: #if there's spike input for each dendrite
                    EXT_waveform_of_a_spike, _ =SPK_GC.EXT_conductance_evaluation(np.arange(time_step-t), dend)
                    EXT_Conductance[dend][t:]+=EXT_waveform_of_a_spike
                    '''STDP, LTD'''
                    for t_backward in [t-t_ for t_ in range(1, WINDOW_INTERVAL+1) if t-t_>=0]:                    
                        if Spike_record[t_backward]: #If there's GC spikes before this MF spike                            
                            LTD_t=SPK_GC.STDP_window_function(t_backward-t, dend)
                            SPK_GC.STDP_weight[dend]+=LTD_t
                            if RECORD_WEIGHT_CHANGE: record_LTD[dend][t]+=LTD_t
                            #print('t:', t, 't_back:', t_backward)
                            #print('delta w in depression:', LTD_t)
                #dV+=SPK_GC.rate_of_change_membrane_voltage(EXT_Conductance[dend][t], STP=True)
            #SPK_GC.membrane_potential+=dV
            #Sum_Conductance_Input=np.sum(EXT_Conductance, axis=0)
            NORMALIZATION=4/SPK_GC.num_dend
            #NORMALIZATION=1
            Sum_Conductance_Input=np.sum(EXT_Conductance[:,t])*NORMALIZATION
            SPK_GC.membrane_potential+=SPK_GC.rate_of_change_membrane_voltage(
                                                     #Sum_Conductance_Input[t], STP=True)
                                                     Sum_Conductance_Input, STP=True)
            
            if SPK_GC.membrane_potential>=activation_threshold:
                #print('SPIKE at time', t)
                Spike_record[t]+=1
                '''STDP, LTP'''                
                for t_backward in [t-t_ for t_ in range(1, WINDOW_INTERVAL+1) if t-t_>=0]:                
                    for dend in range(SPK_GC.num_dend):                    
                        if spike_pttn_per_bin[dend][t_backward]:                            
                            LTP_t=SPK_GC.STDP_window_function(t-t_backward, dend)

                            SPK_GC.STDP_weight[dend]+=LTP_t
                            if RECORD_WEIGHT_CHANGE: record_LTP[dend][t]+=LTP_t
                            #print('t:', t, 't_back:', t_backward)
                            #print('delta w in potentiation:', LTP_t)
                
                REFRACTORY_COUNT=refractory_interval
                SPK_GC.membrane_potential=0          # 0mv Represent spikes (choosed as a arbitrary high value)   
            #elif SPK_GC.membrane_potential<=leak_reversal_potential:
            #    print('Limiting minimum potential by leak revers. Pot. at time', t)
            #    SPK_GC.membrane_potential=leak_reversal_potential            

        for dend in range(SPK_GC.num_dend):
            SPK_GC.PLASTICITY(dend, spike_pttn_per_bin[dend][t]) #Short term plasticity
            


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
    
    #return SPK_GC, EXT_Conductance, INH_Conductance, mem_voltage, record_p_ofEXT, p_ofINH, Spike_record
    if RECORD_WEIGHT_CHANGE: 
        plot_STDP(SPK_GC.num_dend, time_step, STDP_weight_change_record, 1)
        plot_STDP(SPK_GC.num_dend, time_step, record_LTD, 0, 'LTD')
        plot_STDP(SPK_GC.num_dend, time_step, record_LTP, 0, 'LTP')
        print('Total LTP:', np.sum(record_LTP), 'Total LTD:', np.sum(record_LTD))

    return SPK_GC, EXT_Conductance, INH_Conductance, mem_voltage, record_p_ofEXT, Spike_record

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

def plot_STP(NUM_MFs, time_step, record_p_ofEXT, p_ofINH=None):
    # For P development on condunctance terms of STP
    print(np.shape(np.arange(time_step)), np.shape(record_p_ofEXT))
    fig, axs = plt.subplots(NUM_MFs, gridspec_kw={'height_ratios': [1]*(NUM_MFs)})
    for ind_mf in range(NUM_MFs):
        axs[ind_mf].plot(np.arange(time_step), record_p_ofEXT[ind_mf], color='r', label="EXT")
        #axs[ind_mf].plot(np.arange(time_step), p_ofINH[ind_mf],color='g', label="INH")
        axs[ind_mf].axhline(y=r_EXT, color='r', linestyle=':', label='Init_EXT') 
        #axs[ind_mf].axhline(y=r_INH,color='g', linestyle=':', label='Init_INH')
        if ind_mf==NUM_MFs-1: axs[ind_mf].legend()


    axs[0].set_title('P development')
    plt.show()
    sys.exit()

def plot_STDP(NUM_DEND, time_step, RECORD_WEIGHT, original=1, Title='STDP Weight development'):
    # For P development on condunctance terms of STP
    print(np.shape(np.arange(time_step)), np.shape(RECORD_WEIGHT))
    fig, axs = plt.subplots(NUM_DEND, gridspec_kw={'height_ratios': [1]*(NUM_DEND)})
    for ind_mf in range(NUM_DEND):
        axs[ind_mf].plot(np.arange(time_step), RECORD_WEIGHT[ind_mf], color='r', label='RECORD_WEIGHT')
        axs[ind_mf].axhline(y=original, color='r', linestyle=':', label='Init_Weight')
        axs[ind_mf].axhline(y=original+learning_rate, color='k', linestyle=':', label='Upper bound')
        axs[ind_mf].axhline(y=original-learning_rate, color='k', linestyle=':', label='Lower bound')
        if ind_mf==NUM_DEND-1 and original==0: axs[ind_mf].legend()


    axs[0].set_title(Title)
    plt.show()
    #sys.exit()
