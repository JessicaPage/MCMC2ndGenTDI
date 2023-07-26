import numpy as np
from lisaconstants import c 
from .delay_time_dependence_Keplerian import time_dependence
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


	delays_in_time =time_dependence(state_current[0],state_current[1],state_current[2])


	x_combo = x_combo_2_0(delays_in_time,einsum_path_to_use)
	y_combo = y_combo_2_0(delays_in_time,einsum_path_to_use)
	z_combo = z_combo_2_0(delays_in_time,einsum_path_to_use)

	likelihood,chi_2_here = likelihood_analytical_equal_arm(x_combo,y_combo,z_combo)
	#print('likelihood in target log prob')
	#print(likelihood)
	prior = prior_minit1(state_current[0])*prior_semi_major(state_current[1])*prior_arg_per(state_current[2])

	#return model.log_prob(rate=rate, obs=poisson_samples) 
	return likelihood + np.log(prior)	
		  
		  

#........................................................................................
#...........................MCMC Functions.......................................
#........................................................................................

	


def prior_minit1(val_minit):
	#val_ar = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (settings.settings.high-settings.settings.low)*uniform.pdf(val[0],settings.settings.low,settings.settings.high-settings.settings.low) == 1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[1],settings.settings.low,settings.settings.high-settings.settings.low) ==1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[2],settings.settings.low,settings.settings.high-settings.settings.low) == 1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[3],settings.settings.low,settings.settings.high-settings.settings.low) ==1.0:
	if (val_minit >= settings.low_minit1) and (val_minit <= settings.high_minit1):
		return 1
	else:
		return 0

def prior_semi_major(val):
	val = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (settings.settings.high-settings.settings.low)*uniform.pdf(val[0],settings.settings.low,settings.settings.high-settings.settings.low) == 1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[1],settings.settings.low,settings.settings.high-settings.settings.low) ==1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[2],settings.settings.low,settings.settings.high-settings.settings.low) == 1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[3],settings.settings.low,settings.settings.high-settings.settings.low) ==1.0:
	if (val >= settings.low_semi_major) and (val <= settings.high_semi_major):
		return 1
	else:
		return 0


		
def prior_Omega_1(val):
	val = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (settings.settings.high-settings.settings.low)*uniform.pdf(val[0],settings.settings.low,settings.settings.high-settings.settings.low) == 1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[1],settings.settings.low,settings.settings.high-settings.settings.low) ==1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[2],settings.settings.low,settings.settings.high-settings.settings.low) == 1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[3],settings.settings.low,settings.settings.high-settings.settings.low) ==1.0:
	if (val >= settings.low_Omega_1) and (val <= settings.high_Omega_1):
		return 1
	else:
		return 0


	
def prior_arg_per(val):
	val = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (settings.settings.high-settings.settings.low)*uniform.pdf(val[0],settings.settings.low,settings.settings.high-settings.settings.low) == 1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[1],settings.settings.low,settings.settings.high-settings.settings.low) ==1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[2],settings.settings.low,settings.settings.high-settings.settings.low) == 1.0 and (settings.settings.high-settings.settings.low)*uniform.pdf(val[3],settings.settings.low,settings.settings.high-settings.settings.low) ==1.0:
	if (val >= settings.low_arg_per) and (val <= settings.high_arg_per):
		return 1
	else:
		return 0	

