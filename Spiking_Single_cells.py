import numpy as np
import sys
#lists for [a,d] :amplitudes, decay times pairs and, at the last, rho: common rise times

UNIT_SCALE=1e-03 ##Unit scales = milli
TIME_UNIT=UNIT_SCALE
PARAM_SCALING=1/UNIT_SCALE

AMPA_d_amp=np.array([3.724*1e-09, 0.3033*1e-09])                *PARAM_SCALING
AMPA_d_dlt=np.array([0.3351*1e-03, 1.651*1e-03])                *PARAM_SCALING
AMPA_d_rho= 0.3274*1e-03                                        *PARAM_SCALING
AMPA_s_amp=np.array([0.2487*1e-09,0.2799*1e-09, 0.1268*1e-09])  *PARAM_SCALING
AMPA_s_dlt=np.array([0.4*1e-03, 4.899*1e-03, 43.1*1e-03])       *PARAM_SCALING
AMPA_s_rho= 0.5548*1e-03                                        *PARAM_SCALING
NMDA_amp=np.array([17*1e-09, 2.645*1e-09])                      *PARAM_SCALING
NMDA_dlt=np.array([13.52*1e-03, 121.9*1e-03])                   *PARAM_SCALING
NMDA_rho=0.8647*1e-03                                           *PARAM_SCALING


#print('AMPA_d', AMPA_d)


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


#PARAM_SET1=[AMPA_d,AMPA_d_rho,AMPA_s,AMPA_s_rho, NMDA, NMDA_rho]
#PARAM_SET2=[membrane_capacitance,leak_conductance,leak_reversal_potential,\
#                GABA_conductance,GABA_reversal_potential,AMPA_reversal_potential,NMDA_reversal_potential,\
#                    reset_membrane_potential,activation_threshold,refractory_interval]

#for param in PARAM_SET1: param*=PARAM_SCALING
#for param in PARAM_SET2: param*=PARAM_SCALING

class Spiking_cells:
    def __init__(self,num_dend=1): 
        #each refers to Synaptic weghts of each of the components
        self.num_dend=num_dend
        self.p_AMPA_d=0.1249
        self.p_AMPA_s=0.2792
        self.p_NMDA  =0.0322
        self.membrane_potential= reset_membrane_potential

    def AMPA_conductance_evaluation(self, t):
        d_wave=np.zeros(len(t))        
        for ind, _ in enumerate(AMPA_d_amp):            
            #d_wave+=AMPA_d_amp[ind]*(np.exp(-t/AMPA_d_rho)-np.exp(-t/AMPA_d_dlt[ind]))
            d_wave+=AMPA_d_amp[ind]*abs(np.exp(-t/AMPA_d_rho)-np.exp(-t/AMPA_d_dlt[ind]))
        
        s_wave=np.zeros(len(t))
        for ind,_ in enumerate(AMPA_s_amp):            
            #s_wave+=AMPA_s_amp[ind]*(np.exp(-t/AMPA_s_rho)-np.exp(-t/AMPA_s_dlt[ind]))
            s_wave+=AMPA_s_amp[ind]*abs(np.exp(-t/AMPA_s_rho)-np.exp(-t/AMPA_s_dlt[ind]))
        #print('t fo AMPA', t)
        '''
        plt.plot(t, d_wave, label="direct")
        plt.plot(t, s_wave, label="spillover")
        plt.title("AMPA wave for a spike")
        plt.legend()
        plt.show()
        sys.exit()'''
        return self.p_AMPA_d*d_wave+ self.p_AMPA_s*s_wave

    def block(self):
        C1=2.07  *1e-03       #scaling term
        C2=0.015 *1e-03
        MG_out=1 *1e-03 
        z, T, F_per_R, delta_bind, delta_perm= 2, 308.15, 96485.3329/8.31446261815324, 0.35, 0.53  #scale free
        theta=z*F_per_R/T

        '''
        TUP=self.membrane_potential #True_unit_of_potentials_on_exponents, not necessary for numerators & denomenators as it is canceled out
        block =C1*np.exp(delta_bind*theta*TUP)+C2*np.exp(-delta_perm*theta*TUP)\
            /(C1*np.exp(delta_bind*theta*TUP)+C2*np.exp(-delta_perm*theta*TUP)\
                +MG_out*np.exp(-delta_bind*theta*TUP)
            )'''
        theta_V= theta*self.membrane_potential
        block = (
            C1*np.exp(delta_bind*theta_V)+C2*np.exp(-delta_perm*theta_V)
            )\
                /(
                    C1*np.exp(delta_bind*theta_V)+C2*np.exp(-delta_perm*theta_V)\
                    +MG_out*np.exp(-delta_bind*theta_V)
                    )
        #print('delta_bind', delta_bind*theta*self.membrane_potential*1e-03)
        #print('delta_perm', delta_perm*theta*self.membrane_potential*1e-03)
        #print('Mem Voltage:',self.membrane_potential)#, 'TUP:', TUP)
        import math
        #if math.isnan(block):
        #    print('block val:', block, 'term: c1', C1*np.exp(delta_bind*theta*TUP), 'c2', C2*np.exp(-delta_perm*theta*TUP), \
        #    'mg', MG_out*np.exp(-delta_bind*theta*TUP), 'theta', theta, 'Mem Voltage:',self.membrane_potential)
        #    sys.exit()
        if math.isnan(block):
            print('block val:', block, 'term: c1', C1*np.exp(delta_bind*theta), 'c2', C2*np.exp(-delta_perm*theta), \
            'mg', MG_out*np.exp(-delta_bind*theta), 'theta', theta, 'Mem Voltage:',self.membrane_potential)
            sys.exit()
        #print('bloack:', block)
        return block
        #return block

    def NMDA_conductance_evaluation(self, t):
        NMDA_wave=np.zeros(len(t))
        for ind, _ in enumerate(NMDA_amp):            
            #NMDA_wave+=NMDA_amp[0]*(np.exp(-t/NMDA_rho)-np.exp(-t/NMDA_dlt[ind]))
            NMDA_wave+=NMDA_amp[ind]*abs(np.exp(-t/NMDA_rho)-np.exp(-t/NMDA_dlt[ind]))*self.p_NMDA(ind)
        return NMDA_wave*self.block() #, self.p_NMDA*NMDA_wave #W/O Block term


    def rate_of_change_membrane_voltage(self, current_conductance_AMPA, current_conductance_NMDA):

        dV=-1/membrane_capacitance*(\
                leak_conductance*(self.membrane_potential-leak_reversal_potential)\
                +GABA_conductance*(self.membrane_potential-GABA_reversal_potential)\
                +current_conductance_AMPA *(self.membrane_potential-AMPA_reversal_potential)\
                #+self.block()*\
                +1*\
                    current_conductance_NMDA*(self.membrane_potential-NMDA_reversal_potential)\
                )
        '''
        #if abs(dVdt)>1e-1: print('1e-1 dVdt:', dVdt)
        if abs(dV)>0.01:
            pass 
            #, '1/C:', 1/membrane_capacitance\
            
            print('|dV|>0.1:', dV
                ,'leak term:', leak_conductance*(self.membrane_potential-leak_reversal_potential)\
                ,'GABA:',GABA_conductance*(self.membrane_potential-GABA_reversal_potential)\
                ,'AMPA:',current_conductance_AMPA *(self.membrane_potential-AMPA_reversal_potential)\
                ,'NMDA:',self.block()*current_conductance_NMDA*(self.membrane_potential-NMDA_reversal_potential)
            )

            print('Before', self.membrane_potential, 'added', dV,' *', TIME_UNIT)
            #sys.exit()

        #print('GABA C:',GABA_conductance, 'pot', self.membrane_potential, 'rev pot:', -GABA_reversal_potential\
        #    ,'term GABA:',GABA_conductance*(self.membrane_potential-GABA_reversal_potential)\
        #    , 'over capat:', GABA_conductance*(self.membrane_potential-GABA_reversal_potential)/membrane_capacitance) 
        
        
        #    )
        #sys.exit()
        '''
        
        self.membrane_potential+=dV*TIME_UNIT  # change per milli second
        #if abs(dV)>0.001: 
        #    pass
            #print('After', self.membrane_potential)
            #sys.exit()
    







from RateCodingModel import *
from matplotlib import pyplot as plt
time_step=1000
TIME_RANGE=np.linspace(0,TIME_UNIT*time_step, time_step)
NUM_MFs=4
spiking_rate=80



spike_pttn_per_bin=poisson_spike_generator(spiking_rate, time_step, 1e-03)  #20 spkies per sec,
spike_pttn_per_bin=np.array(spike_pttn_per_bin).reshape(len(spike_pttn_per_bin),-1)

#spike_pttn_per_bin=np.array(spike_pttn_per_bin).reshape(time_step,len(Spike_trains))

#print(np.shape(spike_pttn_per_bin))
#sys.exit()

'''Input spike plot'''
#raster_plot(spike_pttn_per_bin, [0])
#sys.exit()

SPK_GC=Spiking_cells(num_dend=1)
AMPA_Conductance=np.zeros(time_step)
NMDA_Conductance=np.zeros(time_step)
mem_voltage=np.zeros(time_step)

REFRACTORY_COUNT=0
for t, time in enumerate(TIME_RANGE):
    mem_voltage[t]=SPK_GC.membrane_potential
    
    if not REFRACTORY_COUNT==0:
        #print('Refractory Count:', REFRACTORY_COUNT,'mem pot:', SPK_GC.membrane_potential)
        SPK_GC.membrane_potential=reset_membrane_potential
        REFRACTORY_COUNT-=1

    else:
        if spike_pttn_per_bin[t]==1:
            Before=SPK_GC.membrane_potential
            #print('Before Membrane potential:', SPK_GC.membrane_potential)
            AMPA_waveform_of_a_spike=SPK_GC.AMPA_conductance_evaluation(np.arange(time_step-t))
            NMDA_waveform_of_a_spike, NMDA_waveform_WO_BLK=SPK_GC.NMDA_conductance_evaluation(np.arange(time_step-t))

            plt.plot(np.arange(time_step-t), NMDA_waveform_WO_BLK, marker='o', markersize=1, linestyle='dashed',label="NMDA W/O BLK")
            plt.plot(np.arange(time_step-t), AMPA_waveform_of_a_spike, label="AMPA(D+S)")
            plt.plot(np.arange(time_step-t), NMDA_waveform_of_a_spike, label="NMDA W BLK")
            
            plt.title("Conduncatnce wave for a spike")
            plt.legend()
            plt.show()
            sys.exit()

            AMPA_Conductance[t:]+=AMPA_waveform_of_a_spike
            NMDA_Conductance[t:]+=NMDA_waveform_of_a_spike

            #SPK_GC.NO_NMDA_rate_of_change_membrane_voltage(conductance[t])                    
        
        SPK_GC.rate_of_change_membrane_voltage(AMPA_Conductance[t], NMDA_Conductance[t])

        #if spike_pttn_per_bin[t]==1: 
        #    print('Membrane potential Before Spike:', SPK_GC.membrane_potential,\
        #         #'After spkie, Change:', SPK_GC.membrane_potential-Before)
        #         'After spkie:', SPK_GC.membrane_potential, 'SPIKE:', SPK_GC.membrane_potential>activation_threshold)
                 
        if SPK_GC.membrane_potential>=activation_threshold:            
            REFRACTORY_COUNT=refractory_interval
            SPK_GC.membrane_potential=reset_membrane_potential
            #print('Output Spike!, Refractory Count:', REFRACTORY_COUNT, "Mem pot:", SPK_GC.membrane_potential)
            #sys.exit()
        elif SPK_GC.membrane_potential<=leak_reversal_potential:
            SPK_GC.membrane_potential=leak_reversal_potential
            #print('MINIMUM potential')
        #print('No spike Membrane potential:', SPK_GC.membrane_potential, 'activation_threshold', activation_threshold)

    

        

    #if t>500:
    #    sys.exit()




#plt.plot(np.arange(time_step), conductance)
#plt.title('Conductance_ No_NMDA, NO STP')
#plt.show()

#raster_plot(spike_pttn_per_bin, [0])
'''
fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [2,1]})

#axs[0].plot(np.arange(time_step), AMPA_Conductance, 'red', label="AMPA")
#axs[0].title('AMPA_Conductance, NO STP')
#axs[0].legend()

axs[0].plot(np.arange(time_step), NMDA_Conductance, 'purple', label="NMDA")
#axs[0].title('NMDA_Conductance, NO STP')
axs[0].legend()

axs[1].plot(np.arange(time_step), spike_pttn_per_bin, label='SPIKE arrivals')
axs[1].legend()
plt.show()
'''

#xtick, ytick = list(range(0,len(data)+1,int(len(data)/10))), list(range(0, len(data[0])+1,int(len(data[0])/10)))
#ytick = list(range(-1e-08,1e-08,int(len(data)/10)))
#plt.yticks(ytick)
#plt.yticks(ytick, rotation=20)

#import matplotlib.ticker as ticker
#fig, ax = plt.subplots
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter())
#raster_plot(spike_pttn_per_bin, [0])

Plotting_Scale=1 # of conducntance

print('Total spikes arrived:', np.sum(spike_pttn_per_bin), 'shape:', np.shape(spike_pttn_per_bin) )

fig, axs = plt.subplots(3, gridspec_kw={'height_ratios': [1,2,2]})
#axs[0].plot(np.arange(time_step), spike_pttn_per_bin, label='SPIKE arrivals')
#axs[0].scatter(np.arange(time_step), spike_pttn_per_bin, marker='|', label='SPIKE arrivals')
x,y = np.argwhere(spike_pttn_per_bin == 1).T
axs[0].scatter(x,y, marker='|', label='SPIKE arrivals')
#ax.scatter(x,y , marker='|', color=spike_color, s=abs(int(3*log(num_data_point*num_GC)/2)))
axs[0].legend()

axs[1].plot(np.arange(time_step), AMPA_Conductance*Plotting_Scale, 'red', label="AMPA")
axs[1].plot(np.arange(time_step), NMDA_Conductance*Plotting_Scale, 'purple', label="NMDA")

#plt.title('Conductances , NO STP')
axs[1].legend()

axs[2].plot(np.arange(time_step), mem_voltage, label='Mem. Pot.')
axs[2].legend()

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
