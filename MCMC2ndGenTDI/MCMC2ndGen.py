
import sys
import numpy as np
import matplotlib.pyplot as plt


class BayesTDI():

	"""Class that performs MCMC to estimate the the time-varying spacecraft separations in 
	the LISA mission. This code generated the results of Page & Littenberg 2023 
	https://arxiv.org/abs/2305.14186. It samples over data simulated by LISA Instrument
	and LISA Orbits package, but the code takes LISA science, reference and test-mass 
	measurements in terms of frequency fluctuations.

	Args:
		data_file: File containing the LISA measurements with column headers 'sij', 'tauij',
			'epsij' for measurements received on S/C i from S/C j and well as truth values for S/C 
			separations Lij as 'mprs_ij'. Data generation code with correct header names is given in 
			'generate_data_ESA.ipynb'.
		cut_off(int): Number of samples to remove at beginning. Make sure it corresponds 
			to sample you quote in elements_from_Cartesian.ipynb if using ESA (numerical) orbit
			model for data simulation.
		f_s(float): Sampling rate in Hz.
		t_init(float,int): Initial time in seconds of data.
		f_min(float): Lower bound of LISA band used in log(likelihood) function. 
		f_max(float): Upper bound of LISA band used in log(likelihood) function. <0.1 Hz 
			is recommended. (See Bayesian TDI articles.) 
		orbit_model(str): Orbit model used in LISA Orbits to simulate the data. Orbit 
			model determines the parameterization of Lij(t).                                 #Haven't implemented Keplerian yet.
		orbital_elements_file: File that contains true initial values of the 6 Keplerian 
			orbital elements for all 3 S/C. Suggested for use in initializing the walkers.
			This file can be generated from elements_from_Cartesian.ipynb
		tcb(True, False): whether the data is simulated in the SSB frame or proper psuedo 
			ranges for delay estimation.
		number_n(int, even or odd): Filter length of FDI filter.
		Nens(int):Number of walkers you want to use -1. 
		Nburnin(int): Number of chains treated as burnin.
		Nsamples(int): Number of chains.
		chainfile(str): Name of file storing the samples. A backend .h5 file is also generated 

	
	"""
	def __init__(self, data_file, cut_off=0,f_s=4.0,t_init= 13100.0,f_min= 5.0e-4,f_max = 0.1,orbit_model='esa',orbital_elements_file='elements_from_Cartesian_4_Hz_3600_sec.dat',tcb=True,number_n=7,Nens = 37,Nburnin = 100,Nsamples = 100000,chainfile='chain.dat'):
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
		self.chainfile = chainfile
		
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
		"""The current sampler for realistic numerical LISA orbits. Currently uses the 
		Zeus ensemble sampler. Sampling data generated with realistic LISA orbits using 
		the 18 parameter model is a work in progress. This method using Zeus is the status so far.
		
		Arg: 
		
		einsum_path_to_use: speeds up NumPy einsum in FDI filter by finding optimum path. 
		See np.einsum docs for options if not calling get_einsum_path() here.
		"""
		import time
		import zeus
		from multiprocessing import Pool
		import settings
		settings.init(self.data_file,self.f_s,self.number_n,self.cut_off,self.t_init,self.central_freq,self.f_min,self.f_max,self.tcb)  
		from mcmc_functions import target_log_prob_fn

		#........................................................................................
		#...........................MCMC Portion.......................................
		#........................................................................................

		start_time = time.time()



		cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=10, discard=0.5)
		cb1 = zeus.callbacks.SaveProgressCallback(self.chainfile, ncheck=100)
		if __name__ == "__main__": 
			with Pool() as pool:
				sampler = zeus.EnsembleSampler(self.Nens, self.ndims, target_log_prob_fn,mu=1e3,pool=pool,args=[einsum_path_to_use])
				sampler.run_mcmc(self.initial_state, self.Nsamples, callbacks=[cb0,cb1])
				#sampler.run_mcmc(initial_state_begin_here, Nsamples-len(samples))

	


			print("--- %s seconds using multiprocessing---" % (time.time() - start_time))


	def get_einsum_path(self):
		"""Run this once before beginning the MCMC run. Calculates the optimum einsum 
		path for NumPy einsum in FDI filter functions.
		
		Returns: 
		
		einsum_path_to_use: speeds up NumPy einsum in FDI filter by finding optimum path. 
		See np.einsum docs for options if not calling get_einsum_path() here.
		"""	

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
if b1.orbit_model=='esa':
	b1.run_zeus_mcmc(einsum_path_to_use)
else:
	print('havent implemented Keplerian parameterization here yet.')