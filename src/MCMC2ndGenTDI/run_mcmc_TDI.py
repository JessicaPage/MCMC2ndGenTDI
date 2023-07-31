
import numpy as np
#import pkgutil
from . import settings
#import importlib.resources
#import os
#from . import data

#data_dir = importlib.resources('mcmc2ndgentdi.data')
#data_path = os.path.join(data, 'LISA_Instrument_Keplerian_orbits_ppr_orbits_4_Hz_3600_sec.dat')

'''
DATA_PATH = pkg_resources.resource_filename('<mcmc2ndgentdi>', 'data/')		
print('DATA_PATH')
print(DATA_PATH)
'''
'''
default_data_file = pkgutil.get_data("mcmc2ndgentdi.data","LISA_Instrument_Keplerian_orbits_ppr_orbits_4_Hz_3600_sec.dat")
print('default_data_file')
print(default_data_file)
'''
class BayesTDI():

	"""Class that performs MCMC to estimate the the time-varying spacecraft separations in the LISA mission. 
	
	This code generated the results of Page & Littenberg 2023 
	https://arxiv.org/abs/2305.14186. It samples over data simulated by LISA Instrument
	and LISA Orbits package, but the code takes LISA science, reference and test-mass 
	measurements in terms of frequency fluctuations.

	Args:
		data_file(str): File containing the LISA measurements with column headers 'sij', 'tauij',
			'epsij' for measurements received on S/C i from S/C j and well as truth values for S/C 
			separations Lij as 'mprs_ij'. Data generation code with correct header names is given in 
			'generate_data_ESA.ipynb' for LISA Instrument simulation used with ESA Orbits in LISA Orbits
			or 'generate_data.ipynb' for Keplerian orbits.
			Need file with headers formatted as 'sij' for science measurements, 'tauij' for reference 
			measurements and 'epsij' for test mass measurements, all three  in frequency fluctuations. 
			Also need each spacecraft SSB time with headers 'time_one', 'time_two', and 'time_three', along with SSB time 'time'.
			(default='LISA_Instrument_Keplerian_orbits_ppr_orbits_2_Hz_86400_sec.dat')
		cut_off(int): Number of samples to remove at beginning. Make sure it corresponds 
			to sample you quote in elements_from_Cartesian.ipynb if using ESA (numerical) orbit
			model for data simulation (default=0).
		f_s(float): Sampling rate in Hz (default=2.0).
		t_init(float): Initial time in seconds (default=0.0). 
		f_min(float): Lower bound of LISA band used in log(likelihood) function (default = 5.0e-4). 
		f_max(float): Upper bound of LISA band used in log(likelihood) function. <0.1 Hz 
			is recommended (default = 0.1).
		orbit_model(str): Orbit model used in LISA Orbits to simulate the data. Orbit 
			model determines the parameterization of Lij(t) (default = 'keplerian').                                
		orbital_elements_file(str): File that contains true initial values of the 6 Keplerian 
			orbital elements for all 3 S/C. Suggested for use in initializing the walkers.
			This file can be generated from elements_from_Cartesian.ipynb. (default = None)
		tcb(Bool): whether the data is simulated in the SSB frame or proper psuedo 
			ranges for delay estimation (default=False).
		number_n(int): Filter length of FDI filter (default=7).
		Nens(int):Number of walkers you want to use - 1 (default=37). 
		Nburnin(int): Number of chains treated as burnin (default=100).
		Nsamples(int): Number of chains (default=100000).
		chainfile(str): Name of .dat file storing the samples (default = 'chainfile.dat'). A backend .h5 file is also generated. 
	"""
	
	#def __init__(self, data_file=data_path, cut_off=0,f_s=2.0,t_init=0.0,f_min= 5.0e-4,f_max = 0.1,orbit_model='keplerian',orbital_elements_file=None,tcb=False,number_n=7,Nens = 37,Nburnin = 100,Nsamples = 100000,chainfile='chainfile.dat'): 
	def __init__(self, data_file='LISA_Instrument_Keplerian_orbits_ppr_orbits_4_Hz_3600_sec.dat', cut_off=0,f_s=2.0,t_init=0.0,f_min= 5.0e-4,f_max = 0.1,orbit_model='keplerian',orbital_elements_file=None,tcb=False,number_n=7,Nens = 37,Nburnin = 100,Nsamples = 100000,chainfile='chainfile.dat'): 
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
		self.data_file = data_file
				
		
		settings.init(self.data_file,self.f_s,self.number_n,self.cut_off,self.t_init,self.central_freq,self.f_min,self.f_max,self.tcb)  
		
		if self.orbit_model=='numerical':
		
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
			
		elif self.orbit_model=='keplerian':
			from .utils.delay_time_dependence_Keplerian import time_dependence
			self.arg_per_0 = -np.pi/2.0
			self.semi_major_0=settings.ASTRONOMICAL_UNIT
			self.m_init1_0 = 0.0
			self.initial_delays_in_time = time_dependence(self.m_init1_0,self.semi_major_0,self.arg_per_0)

			#initial_state_truth = np.array([elements_data[0],elements_data[1],elements_data[2],elements_data[3],elements_data[4],elements_data[5],elements_data[6],elements_data[7],elements_data[8],elements_data[9],elements_data[10],elements_data[11],elements_data[12],elements_data[13],elements_data[14],elements_data[15],elements_data[16],elements_data[17]])
			self.initial_state_truth = np.array([self.m_init1_0,self.semi_major_0,self.arg_per_0])

			
		else:
		
			print("Invalid orbit model.")
			
		
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
		from .utils.mcmc_functions import target_log_prob_fn

		#........................................................................................
		#...........................MCMC Portion.......................................
		#........................................................................................

		start_time = time.time()



		cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=10, discard=0.5)
		cb1 = zeus.callbacks.SaveProgressCallback(self.chainfile, ncheck=100)
		sampler = zeus.EnsembleSampler(self.Nens, self.ndims, target_log_prob_fn,mu=1e3,args=[einsum_path_to_use])
		sampler.run_mcmc(self.initial_state, self.Nsamples, callbacks=[cb0,cb1])

	
		print("--- %s seconds using zeus---" % (time.time() - start_time))

	def run_emcee_Keplerian_mcmc(self,einsum_path_to_use):
		"""The current sampler for realistic KeplerianLISA orbits. Currently uses the 
		emcee ensemble sampler. Sampling data generated with Keplerian LISA orbits using 
		the 3 parameter model. 
		
		Arg: 
		
		einsum_path_to_use: speeds up NumPy einsum in FDI filter by finding optimum path. 
		See np.einsum docs for options if not calling get_einsum_path() here.
		"""
		import time
		import emcee
		from .utils.mcmc_functions_Keplerian import target_log_prob_fn

		#........................................................................................
		#...........................MCMC Portion.......................................
		#........................................................................................

		start_time = time.time()

		#initial delays accepted into the chain
		accept = 1

		initial_state = np.array([np.random.uniform(self.m_init1_0-1.0e-5,self.m_init1_0+1.0e-5,size=self.Nens),np.random.uniform(1.49597e+11,1.49598e+11,size=self.Nens),np.random.uniform(self.arg_per_0-1.0e-6,self.arg_per_0+1.0e-6,size=self.Nens)])

		initial_state=initial_state.T
		self.ndims = initial_state.shape[1]

		initial_state = np.vstack([initial_state,self.initial_state_truth])
	
		self.Nens+=1

		self.ndims = initial_state.shape[1]



		filename = "samples_Keplerian_chain_omega_emcee_backend_testing_small_ball.h5"
		backend = emcee.backends.HDFBackend(filename)
		backend.reset(self.Nens, self.ndims)
		sampler = emcee.EnsembleSampler(self.Nens, self.ndims, target_log_prob_fn,backend=backend,args=[einsum_path_to_use])

		f = open("samples_Keplerian_chain_omega_emcee_testing_small_ball.dat", "w")
		f.close()

		max_n = self.Nsamples

		# We'll track how the average autocorrelation time estimate changes
		index_here = 0
		autocorr = np.empty(max_n)

		# This will be useful to testing convergence
		old_tau = np.inf



		for result in sampler.sample(initial_state, iterations=max_n,progress=True):
			#print('result')
			#print(result)

			position = result[0]
			f = open("samples_Keplerian_chain_omega_emcee_testing_small_ball.dat", "a")
			for k in range(position.shape[0]):
				f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,position[k]))))
			f.close()
	
			if sampler.iteration % 100:
				continue
	
			# Compute the autocorrelation time so far
			# Using tol=0 means that we'll always get an estimate even
			# if it isn't trustworthy
			tau = sampler.get_autocorr_time(tol=0)
			autocorr[index_here] = np.mean(tau)
			index_here += 1

			# Check convergence
			converged = np.all(tau * 100 < sampler.iteration)
			converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
			if converged:
				break
			old_tau = tau

		print("--- %s seconds using emcee---" % (time.time() - start_time))

		n = 100 * np.arange(1, index_here + 1)
		y = autocorr[:index_here]
		plt.plot(n, n / 100.0, "--k")
		plt.plot(n, y)
		plt.xlim(0, n.max())
		plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
		plt.xlabel("number of steps")
		plt.ylabel(r"mean $\hat{\tau}$")
		plt.show()


	def get_einsum_path(self):
		"""Run this once before beginning the MCMC run. Calculates the optimum einsum 
		path for NumPy einsum in FDI filter functions.
		
		Returns: 
		
		einsum_path_to_use: speeds up NumPy einsum in FDI filter by finding optimum path. 
		See np.einsum docs for options if not calling get_einsum_path() here.
		"""	

		#import settings
		#settings.init(self.data_file,self.f_s,self.number_n,self.cut_off,self.t_init,self.central_freq,self.f_min,self.f_max,self.tcb)  
		from .utils.TDI_functions import nested_delay_application
		from .utils.filter_functions import delay_polynomials
		#Get optimal einsum path]
		if self.orbit_model=='numerical':
			from .utils.delay_time_dependence import time_dependence

			orbital_L_3_p = time_dependence(self.m_init1_0,self.semi_major_0,self.eccentricity_0,self.inclination_0,self.omega_init_0,self.arg_per_0)
		elif self.orbit_model=='keplerian':
			from .utils.delay_time_dependence_Keplerian import time_dependence
			orbital_L_3_p = time_dependence(self.m_init1_0,self.semi_major_0,self.arg_per_0)

		nested = nested_delay_application(orbital_L_3_p,np.array([0,1]))
		test_filter = delay_polynomials(nested[0])
		data_to_use_s13 = np.concatenate((np.zeros((test_filter[1],self.number_n+1)),settings.s13_coeffs),axis=0)
		data_here_s13 = data_to_use_s13[:-test_filter[1]:]   
		einsum_path_to_use = np.einsum_path('ij,ji->i',data_here_s13,test_filter[0], optimize='True')[0]
		print('einsum_path_to_use')
		print(einsum_path_to_use)
		#Just plotting Li(t) to make sure truth values are working


		return einsum_path_to_use
		
'''
b1=BayesTDI()

einsum_path_to_use = b1.get_einsum_path()
if b1.orbit_model=='esa':
	b1.run_zeus_mcmc(einsum_path_to_use)
else:
	b1.run_emcee_Keplerian_mcmc(einsum_path_to_use)
'''

