from Simplified_spiking_cells import *

from RateCodingModel import *
from matplotlib import pyplot as plt



#Spike_trains+=poisson_spike_generator(50, time_step, 1e-03)
#Spike_trains+=poisson_spike_generator(50, time_step, 1e-03)
#Spike_trains+=poisson_spike_generator(50, time_step, 1e-03)

#spike_pttn_per_bin=np.array(Spike_trains).reshape(NUM_MFs, time_step)

#EXT_Conductance, INH_Conductance, mem_voltage, p_ofEXT, p_ofINH, Num_OUTPUT_SPIKE = simulator(NUM_MFs, spike_pttn_per_bin)
#plot_cell_dynamics(NUM_MFs, time_step, EXT_Conductance, INH_Conductance, mem_voltage)
#plot_STP(NUM_MFs, time_step, p_ofEXT, p_ofINH)
def spike_generation(Num_MFs, spike_rates, time_step):
    Spike_trains=[]
    for i in range(Num_MFs):
        Spike_trains+=poisson_spike_generator(spike_rates, time_step, 1e-03)  # Firing rate spkies per sec,
    spike_pttn_per_bin=np.array(Spike_trains).reshape(Num_MFs, time_step)
    return spike_pttn_per_bin




NUM_MFs=4
time_step=1000
SPK_GC=Spiking_cells(num_dend=NUM_MFs)

repetition=10
num_rates=20
RATE_FACTOR=5

VALUE_MATRIX=np.zeros((num_rates, repetition))

for i in range(num_rates):
    for j in range(repetition):
        spike_rates=(i+1)*RATE_FACTOR
        spike_pttn_per_bin=spike_generation(NUM_MFs, spike_rates, time_step)
        SPK_GC, _, _, _, _, _, Num_OUTPUT_SPIKE = simulator(NUM_MFs, spike_pttn_per_bin, time_step, SPK_GC)
        VALUE_MATRIX[i][j]=Num_OUTPUT_SPIKE        
        #print('i:',i, 'j:',j)
    print(i,'-th rates:',spike_rates)


VALUE_MATRIX_reverse=np.zeros((num_rates-1, repetition))
for i in reversed(range(num_rates-1)):
    for j in range(repetition):
        spike_rates=(i+1)*(RATE_FACTOR)
        spike_pttn_per_bin=spike_generation(NUM_MFs, spike_rates, time_step)
        SPK_GC, _, _, _, _, _, Num_OUTPUT_SPIKE = simulator(NUM_MFs, spike_pttn_per_bin, time_step, SPK_GC)
        VALUE_MATRIX_reverse[i][j]=Num_OUTPUT_SPIKE
    print('reverse', i,'-th rates:',spike_rates)
    
    #print(('Spike rates=',spike_rates, 'Num INPUT spikes:', np.sum(spike_pttn_per_bin), \
    #        'Num Output spikes:', Num_OUTPUT_SPIKE))

print(VALUE_MATRIX)
mean=np.sum(VALUE_MATRIX, axis=1)/repetition
std=np.std(VALUE_MATRIX, axis=1)

print('  rates','mean ', 'std', 'CV')
arr= np.column_stack((np.arange(1, num_rates+1).T*20, mean.T, std.T, (std/mean*100).T))
print(np.round(arr,1))


mean2=np.sum(VALUE_MATRIX_reverse, axis=1)/repetition
std2=np.std(VALUE_MATRIX_reverse, axis=1)
#print('  rates','mean ', 'std', 'CV')
arr2= np.column_stack((np.arange(1, num_rates).T*RATE_FACTOR, mean2.T, std2.T, (std2/mean2*100).T))
arr2=np.flip(arr2, axis=0)
print(np.round(arr2,1))


fig, axs = plt.subplots(2) #, gridspec_kw={'height_ratios': [1]+[1]})

axs[0].plot(np.arange(1, num_rates+1)*RATE_FACTOR, mean, label='Ascending mean')
axs[0].plot(np.arange(1, num_rates+1)*RATE_FACTOR, np.hstack((mean2, mean[-1])), label='Descending mean')
axs[0].set_title('mean')
axs[0].legend()

axs[1].plot(np.arange(1, num_rates+1)*RATE_FACTOR, (std/mean*100), label='Ascending CV')
axs[1].plot(np.arange(1, num_rates+1)*RATE_FACTOR, np.hstack(((std2/mean2*100), (std/mean*100)[-1])), label='Descending CV')
axs[1].set_title('CV')
axs[1].legend()
plt.show()