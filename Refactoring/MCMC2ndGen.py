
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy.stats import norm
#from lisaconstants import GM_SUN,c,ASTRONOMICAL_UNIT# Constant values set below to avoid importing lisaconstant
import zeus
from multiprocessing import Pool
#import h5py













# # Other Parameters Constant in Code	
f_s = 4.0
f_samp = f_s
#lagrange filter length
number_n=7
p = number_n//2
asd_nu = 28.8 # Hz/rtHz
cut_off= 0
tinit = 13000.0
data_file = 'LISA_Instrument_ESA_orbits_tcb_orbits_4_Hz_3600_sec.dat'
f_min = 5.0e-4 # (= 0.0009765625)
f_max = 0.1
central_freq=281600000000000.0


static=False
equalarmlength=False
keplerian=False
esa=True
matrix=True
tcb = True

import settings
settings.init(data_file,f_s,number_n,cut_off,tinit,central_freq,f_min,f_max,tcb)  

from TDI_functions import *
from delay_time_dependence import time_dependence
from filter_functions import *
from mcmc_functions import *













#........................................................................................
#...........................MCMC Portion.......................................
#........................................................................................



elements_data = np.genfromtxt('elements_from_Cartesian_4_Hz_3600_sec.dat')


initial_state_truth = np.array([elements_data[0],elements_data[1],elements_data[2],elements_data[3],elements_data[4],elements_data[5],elements_data[6],elements_data[7],elements_data[8],elements_data[9],elements_data[10],elements_data[11],elements_data[12],elements_data[13],elements_data[14],elements_data[15],elements_data[16],elements_data[17]])
#initial_state_truth = np.array([elements_data[0],elements_data[1],elements_data[2],elements_data[3],elements_data[4],elements_data[5],elements_data[6],elements_data[7],elements_data[8],elements_data[9],elements_data[10],elements_data[11],elements_data[15],elements_data[16],elements_data[17]])

print('initial_state_truth')
print(initial_state_truth)
#ndims = len(initial_state_truth) # number of parameters/dimensions

Nens = 37 # number of ensemble points
Nburnin = 100   # number of burn-in samples
Nsamples = 100000  # number of final posterior samples

'''
initial_state = np.array([np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),\
				np.random.uniform(low_inclination,high_inclination,size=Nens),np.random.uniform(low_inclination,high_inclination,size=Nens),np.random.uniform(low_inclination,high_inclination,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens),np.random.uniform(low_omega,high_omega,size=Nens),\
				np.random.uniform(low_omega,high_omega,size=Nens),np.random.uniform(low_omega,high_omega,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens)])

initial_state = np.array([np.random.uniform(elements_data[0]-1.0e4,elements_data[0]+1.0e4,size=Nens),np.random.uniform(elements_data[1]-1.0e4,elements_data[1]+1.0e4,size=Nens),np.random.uniform(elements_data[2]-1.0e4,elements_data[2]+1.0e4,size=Nens),np.random.uniform(elements_data[3]-1.0e-4,elements_data[3]+1.0e-4,size=Nens),np.random.uniform(elements_data[4]-1.0e-4,elements_data[4]+1.0e-4,size=Nens),np.random.uniform(elements_data[5]-1.0e-4,elements_data[5]+1.0e-4,size=Nens),\
				np.random.uniform(elements_data[6]-1.0e-4,elements_data[6]+1.0e-4,size=Nens),np.random.uniform(elements_data[7]-1.0e-4,elements_data[7]+1.0e-4,size=Nens),np.random.uniform(elements_data[8]-1.0e-4,elements_data[8]+1.0e-4,size=Nens), np.random.uniform(elements_data[9]-1.0e-5,elements_data[9]+1.0e-5,size=Nens), np.random.uniform(elements_data[10]-1.0e-5,elements_data[10]+1.0e-5,size=Nens), np.random.uniform(elements_data[11]-1.0e-5,elements_data[11]+1.0e-5,size=Nens),\
				np.random.uniform(elements_data[15]-1.0e-6,elements_data[15]+1.0e-6,size=Nens),np.random.uniform(elements_data[16]-1.0e-6,elements_data[16]+1.0e-6,size=Nens),np.random.uniform(elements_data[17]-1.0e-6,elements_data[17]+1.0e-6,size=Nens)])
'''

'''
initial_state = np.array([np.random.uniform(elements_data[0]-1.0e4,elements_data[0]+1.0e4,size=Nens),np.random.uniform(elements_data[1]-1.0e4,elements_data[1]+1.0e4,size=Nens),np.random.uniform(elements_data[2]-1.0e4,elements_data[2]+1.0e4,size=Nens),np.random.uniform(elements_data[3]-1.0e-4,elements_data[3]+1.0e-4,size=Nens),np.random.uniform(elements_data[4]-1.0e-4,elements_data[4]+1.0e-4,size=Nens),np.random.uniform(elements_data[5]-1.0e-4,elements_data[5]+1.0e-4,size=Nens),\
				np.random.uniform(elements_data[6]-1.0e-4,elements_data[6]+1.0e-4,size=Nens),np.random.uniform(elements_data[7]-1.0e-4,elements_data[7]+1.0e-4,size=Nens),np.random.uniform(elements_data[8]-1.0e-4,elements_data[8]+1.0e-4,size=Nens), np.random.uniform(elements_data[9]-1.0e-5,elements_data[9]+1.0e-5,size=Nens), np.random.uniform(elements_data[10]-1.0e-5,elements_data[10]+1.0e-5,size=Nens), np.random.uniform(elements_data[11]-1.0e-5,elements_data[11]+1.0e-5,size=Nens),\
				np.random.uniform(elements_data[12]-1.0e-6,elements_data[12]+1.0e-6,size=Nens),np.random.uniform(elements_data[13]-1.0e-6,elements_data[13]+1.0e-6,size=Nens),np.random.uniform(elements_data[14]-1.0e-6,elements_data[14]+1.0e-6,size=Nens),\
				np.random.uniform(elements_data[15]-1.0e-6,elements_data[15]+1.0e-6,size=Nens),np.random.uniform(elements_data[16]-1.0e-6,elements_data[16]+1.0e-6,size=Nens),np.random.uniform(elements_data[17]-1.0e-6,elements_data[17]+1.0e-6,size=Nens)])
'''
initial_state = np.array([np.random.uniform(elements_data[0]-1.0e-1,elements_data[0]+1.0e-1,size=Nens),np.random.uniform(elements_data[1]-1.0e-1,elements_data[1]+1.0e-1,size=Nens),np.random.uniform(elements_data[2]-1.0e-1,elements_data[2]+1.0e-1,size=Nens),np.random.uniform(elements_data[3]-1.0e-9,elements_data[3]+1.0e-9,size=Nens),np.random.uniform(elements_data[4]-1.0e-9,elements_data[4]+1.0e-9,size=Nens),np.random.uniform(elements_data[5]-1.0e-9,elements_data[5]+1.0e-9,size=Nens),\
				np.random.uniform(elements_data[6]-1.0e-9,elements_data[6]+1.0e-9,size=Nens),np.random.uniform(elements_data[7]-1.0e-9,elements_data[7]+1.0e-9,size=Nens),np.random.uniform(elements_data[8]-1.0e-9,elements_data[8]+1.0e-9,size=Nens), np.random.uniform(elements_data[9]-1.0e-9,elements_data[9]+1.0e-9,size=Nens), np.random.uniform(elements_data[10]-1.0e-9,elements_data[10]+1.0e-9,size=Nens), np.random.uniform(elements_data[11]-1.0e-9,elements_data[11]+1.0e-9,size=Nens),\
				np.random.uniform(elements_data[12]-1.0e-9,elements_data[12]+1.0e-9,size=Nens),np.random.uniform(elements_data[13]-1.0e-9,elements_data[13]+1.0e-9,size=Nens),np.random.uniform(elements_data[14]-1.0e-9,elements_data[14]+1.0e-9,size=Nens),\
				np.random.uniform(elements_data[15]-1.0e-9,elements_data[15]+1.0e-9,size=Nens),np.random.uniform(elements_data[16]-1.0e-9,elements_data[16]+1.0e-9,size=Nens),np.random.uniform(elements_data[17]-1.0e-9,elements_data[17]+1.0e-9,size=Nens)])

'''
initial_state = np.array([np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),\
				np.random.uniform(low_inclination,high_inclination,size=Nens),np.random.uniform(low_inclination,high_inclination,size=Nens),np.random.uniform(low_inclination,high_inclination,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens),\
				np.random.uniform(low_arg_per,high_arg_per,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens)])
'''
print('initial_state')
print(initial_state)
print('initial_state.T')
print(initial_state.T)
#sys.exit()
initial_state=initial_state.T
ndims = initial_state.shape[1]
#initial_state = np.append(initial_state,initial_state_truth)

initial_state = np.vstack([initial_state, initial_state_truth])
print('initial_state appended')
print(initial_state)
print('initial_state_truth')
print(initial_state_truth)
Nens+=1

semi_major_0=np.array([elements_data[0],elements_data[1],elements_data[2]])
eccentricity_0 = np.array([elements_data[3],elements_data[4],elements_data[5]])
inclination_0 = np.array([elements_data[6],elements_data[7],elements_data[8]])
m_init1_0 =np.array([elements_data[9],elements_data[10],elements_data[11]])
omega_init_0 = np.array([elements_data[12],elements_data[13],elements_data[14]])
#omega_init_0 = np.array([np.random.uniform(low_omega,high_omega),np.random.uniform(low_omega,high_omega),np.random.uniform(low_omega,high_omega)])
arg_per_0 = np.array([elements_data[15],elements_data[16],elements_data[17]])



'''
semi_major_0 = np.random.uniform(low_semi_major,high_semi_major,size=3)
m_init1_0 = np.random.uniform(low_minit1,high_minit1,size=3)
eccentricity_0 = np.random.uniform(low_eccentricity,high_eccentricity,size=3)
inclination_0 = np.random.uniform(low_inclination,high_inclination,size=3)
omega_init_0 = np.random.uniform(low_omega,high_omega,size=3)
arg_per_0 = np.random.uniform(low_arg_per,high_arg_per,size=3)
'''
print('semi_major_0')
print(semi_major_0)
print('eccentricity_0')
print(eccentricity_0)
print('inclination_0')
print(inclination_0)
print('m_init1_0')
print(m_init1_0)
print('omega_init_0')
print(omega_init_0)
print('arg_per_0')
print(arg_per_0)


#Get optimal einsum path]
orbital_L_3_p = time_dependence(m_init1_0,semi_major_0,eccentricity_0,inclination_0,omega_init_0,arg_per_0)
nested = nested_delay_application(orbital_L_3_p,np.array([0,1]))
test_filter = delay_polynomials(nested[0])
data_to_use_s13 = np.concatenate((np.zeros((test_filter[1],number_n+1)),settings.s13_coeffs),axis=0)
data_here_s13 = data_to_use_s13[:-test_filter[1]:]   
einsum_path_to_use = np.einsum_path('ij,ji->i',data_here_s13,test_filter[0], optimize='True')[0]
print('einsum_path_to_use')
print(einsum_path_to_use)
#Just plotting Li(t) to make sure truth values are working
'''
plt.plot(ppr_L_3_p,label='full truth')
plt.title('L3p')
plt.plot(orbital_L_3_p[5],label='orbital time dependence')
plt.legend()
plt.show()


plt.plot(ppr_L_1,label='full truth')
plt.title('L1')
plt.plot(orbital_L_3_p[0],label='orbital time dependence')
plt.legend()
plt.show()

plt.plot(ppr_L_1_p,label='full truth')
plt.title('L1p')
plt.plot(orbital_L_3_p[1],label='orbital time dependence')
plt.legend()
plt.show()
'''

'''
with h5py.File('saved_chains_zeus.h5', "r") as hf:
    samples = np.copy(hf['samples'])
    logprob_samples = np.copy(hf['logprob'])
    
initial_state_begin_here = samples[-1]
hf.close()
'''

start_time = time.time()



cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=10, discard=0.5)
cb1 = zeus.callbacks.SaveProgressCallback("saved_chains_zeus_light_mode_false.h5", ncheck=100)
if __name__ == "__main__": 
	with Pool() as pool:
		sampler = zeus.EnsembleSampler(Nens, ndims, target_log_prob_fn,mu=1e3,pool=pool,args=[einsum_path_to_use])
		sampler.run_mcmc(initial_state, Nsamples, callbacks=[cb0,cb1])
		#sampler.run_mcmc(initial_state_begin_here, Nsamples-len(samples))

		


	print("--- %s seconds using multiprocessing---" % (time.time() - start_time))

	plt.figure(figsize=(16,1.5*ndims))
	for n in range(ndims):
		plt.subplot2grid((ndims, 1), (n, 0))
		plt.plot(sampler.get_chain()[:,:,n],alpha=0.5)
		#plt.axhline(y=mu[n])
	plt.tight_layout()
	plt.show()

	chain = sampler.get_chain(flat=True, discard=2500)
	print('Percentiles')
	print (np.percentile(chain, [16, 50, 84], axis=0))
	print('Mean')
	print (np.mean(chain, axis=0))
	print('Standard Deviation')
	print (np.std(chain, axis=0))

	fig, axes = zeus.cornerplot(chain[::100], size=(16,16))

