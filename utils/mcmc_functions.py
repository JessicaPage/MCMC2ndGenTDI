import numpy as np
from lisaconstants import c 
from .delay_time_dependence import time_dependence
from .TDI_functions import *
from .filter_functions import *

def likelihood_analytical_equal_arm(x,y,z):
	'''
	plt.loglog(np.fft.rfftfreq(length,1/f_s),np.abs(np.fft.rfft(s12,norm='ortho'))**2,label='s12 my code')
	plt.loglog(f_band,np.power(np.abs(x[0]+x[1]),2),label='FFT X2 My Code')
	plt.loglog(f_band,np.power(np.abs(y[0]+y[1]),2),label='FFT Y2 My Code')
	plt.loglog(f_band,np.power(np.abs(z[0]+z[1]),2),label='FFT Z2 My Code')
	plt.loglog(f_band, np.power((2 * np.pi * f_band) * asd_nu * 9e-12/central_freq,2), c='gray', ls='--')
	plt.loglog(f_band,a,label=r'$\Sigma_{00}$')
	plt.xlabel(r'Frequency (Hz)')
	plt.ylabel(r'PSD (Hz/Hz)')
	#plt.axvline(f_max,label='f max cut-off')
	#plt.xlim(1e-4,0.1)
	plt.legend()
	#plt.title('Linear Approximation')
	#plt.title('L(t) Nested Delays')
	#plt.title('Non-Moving Arms, Doppler, Ranging and Clock Noises Disabled')
	#plt.savefig('Non-Moving_Arms_Doppler_and_Ranging_and_Clock_disabled.png')
	plt.show()         
	'''

	chi_2 = 1/settings.determinant*(settings.A_*(x[0]**2+x[1]**2+y[0]**2+y[1]**2+z[0]**2+z[1]**2) + 2*settings.B_*(x[0]*y[0]+x[1]*y[1]+x[0]*z[0]+x[1]*z[1]+y[0]*z[0]+y[1]*z[1]))
	'''
	plt.semilogx(f_band,chi_2,label='FFT X2 My Code')
	plt.title('chi^2')
	plt.show()
	'''
	value = -1*np.sum(chi_2) - settings.log_term_factor - np.sum(settings.log_term_determinant)
	
	return value,np.sum(chi_2)	  
	
def target_log_prob_fn(state_current,einsum_path_to_use):

	semi_major = np.array([state_current[0],state_current[1],state_current[2]])
	eccentricity = np.array([state_current[3],state_current[4],state_current[5]])
	inclination = np.array([state_current[6],state_current[7],state_current[8]])
	m_init1 = np.array([state_current[9],state_current[10],state_current[11]])
	omega_init =np.array([state_current[12],state_current[13],state_current[14]])
	arg_per = np.array([state_current[15],state_current[16],state_current[17]])

	
	'''
	semi_major = np.array([state_current_0,state_current_1,state_current_2])
	eccentricity = np.array([state_current_3,state_current_4,state_current_5])
	inclination = np.array([state_current_6,state_current_7,state_current_8])
	m_init1 = np.array([state_current_9,state_current_10,state_current_11])
	omega_init =np.array([state_current_12,state_current_13,state_current_14])
	arg_per  = np.array([state_current_15,state_current_16,state_current_17])
	'''
	delays_in_time =time_dependence(m_init1,semi_major,eccentricity,inclination,omega_init,arg_per)


	x_combo = x_combo_2_0(delays_in_time,einsum_path_to_use)
	y_combo = y_combo_2_0(delays_in_time,einsum_path_to_use)
	z_combo = z_combo_2_0(delays_in_time,einsum_path_to_use)

	likelihood,chi_2_here = likelihood_analytical_equal_arm(x_combo,y_combo,z_combo)

	prior = prior_minit1(m_init1)*prior_semi_major(semi_major)*prior_omega(omega_init)*prior_arg_per(arg_per)*prior_eccentricity(eccentricity)*prior_inclination(inclination)
	#prior = prior_minit1(m_init1)*prior_semi_major(semi_major)*prior_arg_per(arg_per)*prior_eccentricity(eccentricity)*prior_inclination(inclination)

	#return model.log_prob(rate=rate, obs=poisson_samples) 
	return likelihood + np.log(prior)	
		  

#........................................................................................
#...........................MCMC Functions.......................................
#........................................................................................

	


def prior_minit1(val):
	val = np.array(val)
	if (val >= settings.low_minit1).all() and (val <= settings.high_minit1).all():
		return 1
	else:
		return 0

def prior_semi_major(val):
	val = np.array(val)
	if (val >= settings.low_semi_major).all() and (val <= settings.high_semi_major).all():
		return 1
	else:
		return 0


		
def prior_omega(val):
	val = np.array(val)
	if (val >= settings.low_omega).all()  and (val <= settings.high_omega).all():
		return 1
	else:
		return 0
		
def prior_arg_per(val):
	val = np.array(val)
	if (val >= settings.low_arg_per).all()  and (val <= settings.high_arg_per).all() :
		return 1
	else:
		return 0
		
def prior_eccentricity(val):
	val = np.array(val)
	if (val >= settings.low_eccentricity).all() and (val <= settings.high_eccentricity).all():
		return 1
	else:
		return 0
		
def prior_inclination(val):
	val = np.array(val)
	if (val >= settings.low_inclination).all() and (val <= settings.high_inclination).all():
		return 1
	else:
		return 0
		


