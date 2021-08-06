import numpy as np
import sys


UNIT_SCALE=1e-03 ##Unit scales = milli
TIME_UNIT=UNIT_SCALE
PARAM_SCALING=1/UNIT_SCALE #Constants for Time, Conductances, Capacitances are up scaled for computation, Potentials remain same

#lists for [a,d] :amplitudes, decay times pairs and, at the last, rho: common rise times
AMPA_d_dlt=np.array([0.3351*1e-03, 1.651*1e-03])                *PARAM_SCALING  #Time terms are upscaled by milli
AMPA_d_rho= 0.3274*1e-03                                        *PARAM_SCALING
AMPA_s_dlt=np.array([0.4*1e-03, 4.899*1e-03, 43.1*1e-03])       *PARAM_SCALING
AMPA_s_rho= 0.5548*1e-03                                        *PARAM_SCALING
NMDA_dlt=np.array([13.52*1e-03, 121.9*1e-03])                   *PARAM_SCALING
NMDA_rho=0.8647*1e-03                                           *PARAM_SCALING
GABA_dlt=np.array([13.52*1e-03, 121.9*1e-03])                   *PARAM_SCALING
GABA_rho=0.8647*1e-03                                           *PARAM_SCALING


AMPA_d_amp=np.array([3.724*1e-09, 0.3033*1e-09])                *PARAM_SCALING*1  #Amp & conductances upscaled by milli
AMPA_s_amp=np.array([0.2487*1e-09,0.2799*1e-09, 0.1268*1e-09])  *PARAM_SCALING*1
NMDA_amp=np.array([17*1e-09, 2.645*1e-09])                      *PARAM_SCALING*1
GABA_amp=np.array([0.1*1e-09, 0.2*1e-09])                       *PARAM_SCALING*1e-03



membrane_capacitance=3.22*1e-12                 *PARAM_SCALING #C_m
leak_conductance=1.06*1e-09                     *PARAM_SCALING #G_m
GABA_conductance=0.438*1e-09                    *PARAM_SCALING #G_GABAR (tonic)

leak_reversal_potential=-79.9*1e-03             #*PARAM_SCALING #E_m
GABA_reversal_potential=-79.1*1e-03             #*PARAM_SCALING #E_GABAR (tonic)
AMPA_reversal_potential=0                       #*PARAM_SCALING #E_AMPAR
NMDA_reversal_potential=0                       #*PARAM_SCALING #E_NMDAR
reset_membrane_potential=-63*1e-03              #*PARAM_SCALING #V_r
activation_threshold=-40*1e-03                  #*PARAM_SCALING #V_t

refractory_interval=2*1e-03                     *PARAM_SCALING #tau



r_AMPA_d=0.1249
r_AMPA_s=0.2792
r_NMDA  =0.0322
r_GABA =0.02

class Spiking_cells:
    def __init__(self,num_dend=1): 
        #each refers to Synaptic weghts of each of the components
        self.p_AMPA_d_U=[]
        self.p_AMPA_d_R=[]
        self.p_AMPA_s_U=[]
        self.p_AMPA_s_R=[]
        self.p_NMDA_U  =[]
        self.p_NMDA_R  =[]

        self.p_GABA_U  =[]
        self.p_GABA_R  =[]
        for dend in range(num_dend):
            self.p_AMPA_d_U.append(r_AMPA_d)
            self.p_AMPA_d_R.append(1)
            self.p_AMPA_s_U.append(r_AMPA_s)
            self.p_AMPA_s_R.append(1)
            self.p_NMDA_U  .append(r_NMDA)
            self.p_NMDA_R  .append(1)

            self.p_GABA_U  .append(r_GABA) #GABA added
            self.p_GABA_R  .append(1)

        self.num_dend=num_dend
        self.membrane_potential= reset_membrane_potential
        self.MEM_POT_NOSTP= reset_membrane_potential


    def p_AMPA_d(self, ind_dend): return self.p_AMPA_d_U[ind_dend]*self.p_AMPA_d_R[ind_dend]
    def p_AMPA_s(self, ind_dend): return self.p_AMPA_s_U[ind_dend]*self.p_AMPA_s_R[ind_dend]
    def p_NMDA(self,   ind_dend): return self.p_NMDA_U[ind_dend]*self.p_NMDA_R[ind_dend]
    def p_GABA(self,   ind_dend): return self.p_GABA_U[ind_dend]*self.p_GABA_R[ind_dend] #GABA added
    

    def PLASTICITY(self, ind_dend, SPIKE):
        p_AMPA_d_U=self.p_AMPA_d_U[ind_dend]
        p_AMPA_s_U=self.p_AMPA_s_U[ind_dend]
        p_NMDA_U  =self.p_NMDA_U  [ind_dend]
        p_AMPA_d_R=self.p_AMPA_d_R[ind_dend]
        p_AMPA_s_R=self.p_AMPA_s_R[ind_dend]
        p_NMDA_R  =self.p_NMDA_R  [ind_dend]

        p_GABA_U  =self.p_GABA_U  [ind_dend] #GABA added
        p_GABA_R  =self.p_GABA_R  [ind_dend]
        if SPIKE:
            self.p_AMPA_d_U[ind_dend]+=r_AMPA_d*(1-p_AMPA_d_U)
            self.p_AMPA_s_U[ind_dend]+=r_AMPA_s*(1-p_AMPA_s_U)
            self.p_NMDA_U  [ind_dend]+=r_NMDA  *(1-p_NMDA_U  )
            self.p_AMPA_d_R[ind_dend]=p_AMPA_d_R*(1-p_AMPA_d_U)
            self.p_AMPA_s_R[ind_dend]=p_AMPA_s_R*(1-p_AMPA_s_U)
            self.p_NMDA_R  [ind_dend]=p_NMDA_R  *(1-p_NMDA_U  )
            self.p_GABA_U  [ind_dend]+=r_GABA   *(1-p_GABA_U  )  #GABA added
            self.p_GABA_R  [ind_dend]= p_GABA_R *(1-p_GABA_U  )
        else:
            self.p_AMPA_d_U[ind_dend]-=(p_AMPA_d_U-r_AMPA_d)/6.394*1000*TIME_UNIT #a Potentiation Recovery time constant 
            self.p_AMPA_s_U[ind_dend]-=(p_AMPA_s_U-r_AMPA_s)/6.394*1000*TIME_UNIT # For NMDA, But since no info. for AMPA so.
            self.p_NMDA_U  [ind_dend]-=(p_NMDA_U  -r_NMDA  )/6.394*1000*TIME_UNIT # All for same
            self.p_AMPA_d_R[ind_dend]-=(p_AMPA_d_R-1)/131         *1000*TIME_UNIT
            self.p_AMPA_s_R[ind_dend]-=(p_AMPA_s_R-1)/14.85       *1000*TIME_UNIT
            self.p_NMDA_R  [ind_dend]-=(p_NMDA_R  -1)/236.1       *1000*TIME_UNIT
            self.p_GABA_U  [ind_dend]-=(p_GABA_U-r_GABA   )/6.394 *1000*TIME_UNIT#GABA added
            self.p_GABA_R  [ind_dend]-=(p_GABA_R -1)/100          *1000*TIME_UNIT



    def AMPA_conductance_evaluation(self, t, index_dend):
        d_wave=np.zeros(len(t))        
                
        for ind, _ in enumerate(AMPA_d_amp):            
            #d_wave+=AMPA_d_amp[ind]*(np.exp(-t/AMPA_d_rho)-np.exp(-t/AMPA_d_dlt[ind]))
            d_wave+=AMPA_d_amp[ind]*(np.exp(-t/AMPA_d_dlt[ind])-np.exp(-t/AMPA_d_rho))
            d_comp=(AMPA_d_amp[ind]*(np.exp(-t/AMPA_d_dlt[ind])-np.exp(-t/AMPA_d_rho)))
            #plt.plot(t, d_comp, label='d%s'%ind)
        #plt.legend()
        #plt.show()
        s_wave=np.zeros(len(t))
        for ind,_ in enumerate(AMPA_s_amp):            
            #s_wave+=AMPA_s_amp[ind]*(np.exp(-t/AMPA_s_rho)-np.exp(-t/AMPA_s_dlt[ind]))
            s_wave+=AMPA_s_amp[ind]*(np.exp(-t/AMPA_s_dlt[ind])-np.exp(-t/AMPA_s_rho))
            #s_comp=(AMPA_s_amp[ind]*(np.exp(-t/AMPA_s_dlt[ind])-np.exp(-t/AMPA_s_rho)))
            #plt.plot(t, s_comp, label='s%s'%ind)
        #plt.legend()
        #plt.show()
        #sys.exit()
    
        #print('t fo AMPA', t)
        #plt.plot(t, s_wave, label="spillover")
        #plt.plot(t, d_wave, label="direct")        
        #plt.title("AMPA wave for a spike")
        #plt.legend()
        #plt.show()
        #sys.exit()
        return self.p_AMPA_d(index_dend)*d_wave + self.p_AMPA_s(index_dend)*s_wave, r_AMPA_d*d_wave + r_AMPA_s*s_wave

    def block(self, STP=True):
        C1=2.07  *1e-03       #scaling term
        C2=0.015 *1e-03
        MG_out=1 *1e-03 
        z, T, F_per_R, delta_bind, delta_perm= 2, 308.15, 96485.3329/8.31446261815324, 0.35, 0.53  #scale free
        theta=z*F_per_R/T
        Current_Mem_Pot=self.membrane_potential if STP else self.MEM_POT_NOSTP
        theta_V= theta*Current_Mem_Pot
        block = (
            C1*np.exp(delta_bind*theta_V)+C2*np.exp(-delta_perm*theta_V)
            )\
                /(
                    C1*np.exp(delta_bind*theta_V)+C2*np.exp(-delta_perm*theta_V)\
                    +MG_out*np.exp(-delta_bind*theta_V)
                    )        
        import math
        if math.isnan(block) or block>1 or block<0:
            raise Exception('math.isnan(block) or block>1 or block<0\n', 
            'block val:', block, 'term: c1', C1*np.exp(delta_bind*theta), 'c2', C2*np.exp(-delta_perm*theta), 
            'mg', MG_out*np.exp(-delta_bind*theta), 'theta', theta, 'Mem Voltage:',Current_Mem_Pot)        
        return block

    def NMDA_conductance_evaluation(self, t, index_dend):
        NMDA_wave=np.zeros(len(t))
        for ind, _ in enumerate(NMDA_amp):                        
            #NMDA_wave+=NMDA_amp[ind]*abs(np.exp(-t/NMDA_rho)-np.exp(-t/NMDA_dlt[ind]))
            NMDA_wave+=NMDA_amp[ind]*(np.exp(-t/NMDA_dlt[ind])-np.exp(-t/NMDA_rho))
        return self.p_NMDA(index_dend)*NMDA_wave*self.block(STP=True), r_NMDA*NMDA_wave*self.block(STP=False)
    
    def GABA_conductance_evaluation(self, t, index_dend):
        GABA_wave=np.zeros(len(t))
        for ind, _ in enumerate(GABA_amp):                        
            GABA_wave+=GABA_amp[ind]*(np.exp(-t/GABA_dlt[ind])-np.exp(-t/GABA_rho))
        return self.p_GABA(index_dend)*GABA_wave, r_GABA*GABA_wave


    def rate_of_change_membrane_voltage(self, Current_Mem_Pot, current_conductance_AMPA, current_conductance_NMDA,\
                                            current_condunctance_GABA, STP=True):

        dV=-1/membrane_capacitance*(\
                leak_conductance*(Current_Mem_Pot-leak_reversal_potential)\
                #+GABA_conductance*(Current_Mem_Pot-GABA_reversal_potential)\
                +current_condunctance_GABA*(Current_Mem_Pot-GABA_reversal_potential)\
                +current_conductance_AMPA *(Current_Mem_Pot-AMPA_reversal_potential)\
                #block* moved to NMDA evaluation function
                +current_conductance_NMDA*(Current_Mem_Pot-NMDA_reversal_potential)\
                )
        
        if STP: self.membrane_potential+=dV*TIME_UNIT  # change per milli second
        else: self.MEM_POT_NOSTP+=dV*TIME_UNIT




from RateCodingModel import *
from matplotlib import pyplot as plt
time_step=1000
TIME_RANGE=np.linspace(0,TIME_UNIT*time_step, time_step)
NUM_MFs=4
spiking_rate=80




Spike_trains=[]
Spike_trains+=poisson_spike_generator(80, time_step, 1e-03)  # Firing rate spkies per sec,
Spike_trains+=poisson_spike_generator(80, time_step, 1e-03)
Spike_trains+=poisson_spike_generator(10, time_step, 1e-03)
Spike_trains+=poisson_spike_generator(10, time_step, 1e-03)

spike_pttn_per_bin=np.array(Spike_trains).reshape(NUM_MFs, time_step)

#print(np.shape(spike_pttn_per_bin))
#print(np.shape(spike_pttn_per_bin[0]))
#sys.exit()

'''Input spike plot'''
#raster_plot(spike_pttn_per_bin, [0])
#sys.exit()

SPK_GC=Spiking_cells(num_dend=NUM_MFs) 
''' For recording the values of variables as np.zeros'''
AMPA_Conductance=np.zeros((NUM_MFs, time_step))
NMDA_Conductance=np.zeros((NUM_MFs, time_step))
GABA_Conductance=np.zeros((NUM_MFs, time_step))

AMPA_NO_STP=np.zeros((NUM_MFs, time_step))
NMDA_NO_STP=np.zeros((NUM_MFs, time_step))
mem_voltage=np.zeros(time_step)
REFRACTORY_COUNT=0

mem_voltage2=np.zeros(time_step)
REFRACTORY_COUNT2=0

p_ofAMPA_d=np.zeros((NUM_MFs,time_step))
p_ofAMPA_s=np.zeros((NUM_MFs,time_step))
p_ofNMDA  =np.zeros((NUM_MFs,time_step))
p_ofGABA  =np.zeros((NUM_MFs,time_step))
''' Simulation loop start '''
for t, time in enumerate(TIME_RANGE):
    mem_voltage[t]=SPK_GC.membrane_potential

    for dend in range(SPK_GC.num_dend):
        p_ofAMPA_d[dend][t]=SPK_GC.p_AMPA_d(dend)
        p_ofAMPA_s[dend][t]=SPK_GC.p_AMPA_s(dend)
        p_ofNMDA[dend][t]=  SPK_GC.p_NMDA(dend)
        p_ofGABA[dend][t]=  SPK_GC.p_GABA(dend)
    

    if not REFRACTORY_COUNT==0:  #Refractory Period        
        SPK_GC.membrane_potential=reset_membrane_potential
        REFRACTORY_COUNT-=1
    else:        
        for dend in range(SPK_GC.num_dend):
            if spike_pttn_per_bin[dend][t]==1: #if there's a spike input for a dendrite
                AMPA_waveform_of_a_spike, _ =SPK_GC.AMPA_conductance_evaluation(np.arange(time_step-t), dend)
                NMDA_waveform_of_a_spike, _ =SPK_GC.NMDA_conductance_evaluation(np.arange(time_step-t), dend)
                GABA_waveform_of_a_spike, _ =SPK_GC.GABA_conductance_evaluation(np.arange(time_step-t), dend)
                AMPA_Conductance[dend][t:]+=AMPA_waveform_of_a_spike
                NMDA_Conductance[dend][t:]+=NMDA_waveform_of_a_spike
                GABA_Conductance[dend][t:]+=GABA_waveform_of_a_spike

            SPK_GC.rate_of_change_membrane_voltage(SPK_GC.membrane_potential,
                                                 AMPA_Conductance[dend][t], NMDA_Conductance[dend][t],\
                                                    GABA_Conductance[dend][t], STP=True)            

        if SPK_GC.membrane_potential>=activation_threshold:
            print('SPIKE at time', t)
            REFRACTORY_COUNT=refractory_interval
            SPK_GC.membrane_potential=0          # 0mv Represent spikes (choosed as a arbitrary high value)   
        elif SPK_GC.membrane_potential<=leak_reversal_potential:
            print('Limiting minimum potential by leak revers. Pot. at time', t)
            SPK_GC.membrane_potential=leak_reversal_potential

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
                _, AMPA_WAVE_No_STP=SPK_GC.AMPA_conductance_evaluation(np.arange(time_step-t), dend)
                _, NMDA_WAVE_No_STP=SPK_GC.NMDA_conductance_evaluation(np.arange(time_step-t), dend)
                AMPA_NO_STP[dend][t:]+=AMPA_WAVE_No_STP
                NMDA_NO_STP[dend][t:]+=NMDA_WAVE_No_STP
            
            SPK_GC.rate_of_change_membrane_voltage(SPK_GC.MEM_POT_NOSTP, 
                                                AMPA_NO_STP[dend][t],NMDA_NO_STP[dend][t], STP=False)

        if SPK_GC.MEM_POT_NOSTP>=activation_threshold:
            print('SPIKE at time', t, 'For No STP')
            REFRACTORY_COUNT2=refractory_interval
            SPK_GC.MEM_POT_NOSTP=0   '''




#plt.plot(np.arange(time_step), conductance)
#plt.title('Conductance_ No_NMDA, NO STP')
#plt.show()

#raster_plot(spike_pttn_per_bin, [0])

Plotting_Scale=1*UNIT_SCALE # of conducntance

print('Total spikes arrived:', np.sum(spike_pttn_per_bin), 'shape:', np.shape(spike_pttn_per_bin) )

fig, axs = plt.subplots(2+NUM_MFs, gridspec_kw={'height_ratios': [1]*(NUM_MFs)+[0.1]+[2]})



for ind_mf in range(NUM_MFs):
    axs[ind_mf].plot(np.arange(time_step), AMPA_Conductance[ind_mf]*Plotting_Scale, 'red', label="AMPA")
    axs[ind_mf].plot(np.arange(time_step), NMDA_Conductance[ind_mf]*Plotting_Scale, 'purple', label="NMDA")
    axs[ind_mf].plot(np.arange(time_step), GABA_Conductance[ind_mf]*Plotting_Scale, 'g', label="GABA")
    axs[ind_mf].axhline(y=630*1e-12*PARAM_SCALING*Plotting_Scale, color='c', linestyle=':', label='Peak amplitude') #of both condunctances
    #axs[ind_mf].plot(np.arange(time_step), AMPA_NO_STP[ind_mf]*Plotting_Scale, label="AMPA_NO")
    #axs[ind_mf].plot(np.arange(time_step), NMDA_NO_STP[ind_mf]*Plotting_Scale, label="NMDA_NO")
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

#sys.exit()
# For P development on condunctance terms of STP
fig, axs = plt.subplots(NUM_MFs, gridspec_kw={'height_ratios': [1]*(NUM_MFs)})
for ind_mf in range(NUM_MFs):
    axs[ind_mf].plot(np.arange(time_step), p_ofAMPA_d[ind_mf], color='r', label="AMPA_d")
    axs[ind_mf].plot(np.arange(time_step), p_ofAMPA_s[ind_mf], color='c', label="AMPA_s")
    axs[ind_mf].plot(np.arange(time_step), p_ofNMDA[ind_mf],   color='m', label="NMDA")
    axs[ind_mf].plot(np.arange(time_step), p_ofGABA[ind_mf],   color='g', label="GABA")
    axs[ind_mf].axhline(y=r_AMPA_d, color='r', linestyle=':', label='Init_AMPA_d')
    axs[ind_mf].axhline(y=r_AMPA_s, color='c', linestyle=':', label='Init_AMPA_s')
    axs[ind_mf].axhline(y=r_NMDA,   color='m', linestyle=':', label='Init_NMDA')    
    axs[ind_mf].axhline(y=r_GABA,   color='g', linestyle=':', label='Init_GABA')    
    if ind_mf==3: axs[ind_mf].legend()
    

axs[0].set_title('P development')
plt.show()
sys.exit()


Plotting_Scale=1 # of conducntance
plt.plot(np.arange(time_step), AMPA_Conductance*Plotting_Scale, 'red', label="AMPA")
plt.plot(np.arange(time_step), NMDA_Conductance*Plotting_Scale, 'purple', label="NMDA")
#plt.title('Conductance, NO STP, No Mg block')
plt.title('Conductance, NO STP')
#plt.legend(handles=[plt_AMPA,plt_NMDA])
plt.legend()
plt.show()


#plt.plot(np.arange(time_step), mem_voltage/UNIT_SCALE)
plt.plot(np.arange(time_step), mem_voltage)
plt.title('Membrane Voltage')
plt.show()
