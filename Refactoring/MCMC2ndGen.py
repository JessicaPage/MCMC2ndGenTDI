
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import zeus
from multiprocessing import Pool











class BayesTDI():

	def __init__(self, data_file, cut_off=0,f_s=4.0,t_init= 13100.0,f_min= 5.0e-4,f_max = 0.1,orbit_model='esa',orbital_elements_file='elements_from_Cartesian_4_Hz_3600_sec.dat',tcb=True,number_n=7,Nens = 37,Nburnin = 100,Nsamples = 100000):
		self.data_file = data_file
		self.f_s = f_s
		self.number_n = number_n
		self.cut_off = cut_off
		self.t_init = t_init
		self.f_min = f_min
		self.f_max = f_max
		self.central_freq = 281600000000000.0
		self.tcb=tcb
		self.orbit_model=orbit_model
		self.orbital_elements_file = orbital_elements_file
		self.Nens = Nens
		self.Nburnin = Nburnin
		self.Nsamples = Nsamples
		
		elements_data = np.genfromtxt(self.orbital_elements_file)


		initial_state_truth = np.array([elements_data[0],elements_data[1],elements_data[2],elements_data[3],elements_data[4],elements_data[5],elements_data[6],elements_data[7],elements_data[8],elements_data[9],elements_data[10],elements_data[11],elements_data[12],elements_data[13],elements_data[14],elements_data[15],elements_data[16],elements_data[17]])
		#initial_state_truth = np.array([elements_data[0],elements_data[1],elements_data[2],elements_data[3],elements_data[4],elements_data[5],elements_data[6],elements_data[7],elements_data[8],elements_data[9],elements_data[10],elements_data[11],elements_data[15],elements_data[16],elements_data[17]])




		self.initial_state = np.array([np.random.uniform(elements_data[0]-1.0e-1,elements_data[0]+1.0e-1,size=self.Nens),np.random.uniform(elements_data[1]-1.0e-1,elements_data[1]+1.0e-1,size=self.Nens),np.random.uniform(elements_data[2]-1.0e-1,elements_data[2]+1.0e-1,size=self.Nens),np.random.uniform(elements_data[3]-1.0e-9,elements_data[3]+1.0e-9,size=self.Nens),np.random.uniform(elements_data[4]-1.0e-9,elements_data[4]+1.0e-9,size=self.Nens),np.random.uniform(elements_data[5]-1.0e-9,elements_data[5]+1.0e-9,size=self.Nens),\
						np.random.uniform(elements_data[6]-1.0e-9,elements_data[6]+1.0e-9,size=self.Nens),np.random.uniform(elements_data[7]-1.0e-9,elements_data[7]+1.0e-9,size=self.Nens),np.random.uniform(elements_data[8]-1.0e-9,elements_data[8]+1.0e-9,size=self.Nens), np.random.uniform(elements_data[9]-1.0e-9,elements_data[9]+1.0e-9,size=self.Nens), np.random.uniform(elements_data[10]-1.0e-9,elements_data[10]+1.0e-9,size=self.Nens), np.random.uniform(elements_data[11]-1.0e-9,elements_data[11]+1.0e-9,size=self.Nens),\
						np.random.uniform(elements_data[12]-1.0e-9,elements_data[12]+1.0e-9,size=self.Nens),np.random.uniform(elements_data[13]-1.0e-9,elements_data[13]+1.0e-9,size=self.Nens),np.random.uniform(elements_data[14]-1.0e-9,elements_data[14]+1.0e-9,size=self.Nens),\
						np.random.uniform(elements_data[15]-1.0e-9,elements_data[15]+1.0e-9,size=self.Nens),np.random.uniform(elements_data[16]-1.0e-9,elements_data[16]+1.0e-9,size=self.Nens),np.random.uniform(elements_data[17]-1.0e-9,elements_data[17]+1.0e-9,size=self.Nens)])


		self.initial_state=self.initial_state.T
		self.ndims = self.initial_state.shape[1]

		self.initial_state = np.vstack([self.initial_state, initial_state_truth])

		self.Nens+=1

		self.semi_major_0=np.array([elements_data[0],elements_data[1],elements_data[2]])
		self.eccentricity_0 = np.array([elements_data[3],elements_data[4],elements_data[5]])
		self.inclination_0 = np.array([elements_data[6],elements_data[7],elements_data[8]])
		self.m_init1_0 =np.array([elements_data[9],elements_data[10],elements_data[11]])
		self.omega_init_0 = np.array([elements_data[12],elements_data[13],elements_data[14]])
		self.arg_per_0 = np.array([elements_data[15],elements_data[16],elements_data[17]])
		
	def run_zeus_mcmc(self,einsum_path_to_use):


		import settings
		settings.init(self.data_file,self.f_s,self.number_n,self.cut_off,self.t_init,self.central_freq,self.f_min,self.f_max,self.tcb)  
		from mcmc_functions import target_log_prob_fn

		#........................................................................................
		#...........................MCMC Portion.......................................
		#........................................................................................





		start_time = time.time()



		cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=10, discard=0.5)
		cb1 = zeus.callbacks.SaveProgressCallback("saved_chains_zeus_light_mode_false.h5", ncheck=100)
		if __name__ == "__main__": 
			with Pool() as pool:
				sampler = zeus.EnsembleSampler(self.Nens, self.ndims, target_log_prob_fn,mu=1e3,pool=pool,args=[einsum_path_to_use])
				sampler.run_mcmc(self.initial_state, self.Nsamples, callbacks=[cb0,cb1])
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

	def get_einsum_path(self):
	
		import settings
		settings.init(self.data_file,self.f_s,self.number_n,self.cut_off,self.t_init,self.central_freq,self.f_min,self.f_max,self.tcb)  
		from TDI_functions import nested_delay_application
		from delay_time_dependence import time_dependence
		from filter_functions import delay_polynomials
		#Get optimal einsum path]
		orbital_L_3_p = time_dependence(self.m_init1_0,self.semi_major_0,self.eccentricity_0,self.inclination_0,self.omega_init_0,self.arg_per_0)
		nested = nested_delay_application(orbital_L_3_p,np.array([0,1]))
		test_filter = delay_polynomials(nested[0])
		data_to_use_s13 = np.concatenate((np.zeros((test_filter[1],self.number_n+1)),settings.s13_coeffs),axis=0)
		data_here_s13 = data_to_use_s13[:-test_filter[1]:]   
		einsum_path_to_use = np.einsum_path('ij,ji->i',data_here_s13,test_filter[0], optimize='True')[0]
		print('einsum_path_to_use')
		print(einsum_path_to_use)
		#Just plotting Li(t) to make sure truth values are working


		return einsum_path_to_use


b1 = BayesTDI('LISA_Instrument_ESA_orbits_tcb_orbits_4_Hz_3600_sec.dat', cut_off=0,f_s=4.0,t_init= 13100.00,f_min= 5.0e-4,f_max = 0.1,orbit_model='esa',orbital_elements_file='elements_from_Cartesian_4_Hz_3600_sec.dat',tcb=True,number_n=7,Nens = 37,Nburnin = 100,Nsamples = 100000)
einsum_path_to_use = b1.get_einsum_path()
b1.run_zeus_mcmc(einsum_path_to_use)