import numpy as np
import math
from scipy.signal import kaiser,kaiser_beta
#from likelihood_functions import *
from lisaconstants import GM_SUN,c,ASTRONOMICAL_UNIT# Constant values set below to avoid importing lisaconstant


def difference_operator_powers(data_):

    """Return backwards difference operators of the LISA measurements at beginning of the 
    run prior to MCMC iterations."""

    difference_coefficients = np.zeros((number_n+1,length))
    difference_coefficients[0] = data_

    
    for i in np.arange(1,number_n+1):
        sum_for_this_power = np.zeros(length)
        for j in np.arange(i+1):

            data_rolled = np.roll(data_,j)
            data_rolled[:j] = 0.0
            sum_for_this_power = sum_for_this_power + (-1)**j*math.comb(i, j)*data_rolled

        difference_coefficients[i] = sum_for_this_power/np.math.factorial(i)

    return difference_coefficients.T



def S_y_proof_mass_new_frac_freq(f):

    pm_here =  np.power(2.4e-15,2)*(1+np.power(4.0e-4/f,2))*(1+np.power(f/8.0e-3,4))
    return pm_here*np.power(2*np.pi*f*c,-2)


def S_y_OMS_frac_freq(f):

    op_here =  np.power(1.5e-11,2)*np.power(2*np.pi*f/c,2)*(1+np.power(2.0e-3/f,4))
    return op_here


def covariance_equal_arm(f,Sy_OP,Sy_PM):
    
    a = 16*np.power(np.sin(2*np.pi*f*avg_L),2)*Sy_OP+(8*np.power(np.sin(4*np.pi*f*avg_L),2)+32*np.power(np.sin(2*np.pi*f*avg_L),2))*Sy_PM
    b_ = -4*np.sin(2*np.pi*f*avg_L)*np.sin(4*np.pi*f*avg_L)*(4*Sy_PM+Sy_OP)

    return 2*a,2*b_


def init(data_file,fs,numbern,cut_off,tinit,central_freq,f_min,f_max,tcb):

	global L_arm
	L_arm = 2.5e9
	global avg_L
	avg_L = L_arm/c
	global f_s
	f_s = fs
	global number_n
	number_n = numbern
	global t_init
	t_init = tinit
	global p
	p = number_n//2

	global is_tcb
	if tcb==True:
		is_tcb =True
	else:
		is_tcb=False
	data = np.genfromtxt(data_file,names=True)

	#times = np.arange(initial_length)/f_s
	times = data['time'][cut_off::]
	times_one = data['time_one'][cut_off::]
	times_two = data['time_two'][cut_off::]
	times_three = data['time_three'][cut_off::]
	'''
	#for LISA Orbits 2.1 
	times = data['time'][cut_off::]
	times_one = times + data['time_one'][cut_off::]
	times_two = times + data['time_two'][cut_off::]
	times_three = times + data['time_three'][cut_off::]
	'''
	global tcb_times
	tcb_times = np.array([times_one,times_two,times_three])


	global s31
	s31 = data['s31'][cut_off::]/central_freq
	global s21
	s21 = data['s21'][cut_off::]/central_freq
	global s32
	s32 = data['s32'][cut_off::]/central_freq
	global s12
	s12 = data['s12'][cut_off::]/central_freq
	global s23
	s23 = data['s23'][cut_off::]/central_freq
	global s13
	s13 = data['s13'][cut_off::]/central_freq


	global tau31
	tau31 = data['tau31'][cut_off::]/central_freq
	global tau21
	tau21 = data['tau21'][cut_off::]/central_freq
	global tau12
	tau12 = data['tau12'][cut_off::]/central_freq
	global tau32
	tau32 = data['tau32'][cut_off::]/central_freq
	global tau23
	tau23 = data['tau23'][cut_off::]/central_freq
	global tau13
	tau13 = data['tau13'][cut_off::]/central_freq


	global eps31
	eps31 = data['eps31'][cut_off::]/central_freq
	global eps21
	eps21 = data['eps21'][cut_off::]/central_freq
	global eps12
	eps12 = data['eps12'][cut_off::]/central_freq
	global eps32
	eps32 = data['eps32'][cut_off::]/central_freq
	global eps23
	eps23 = data['eps23'][cut_off::]/central_freq
	global eps13
	eps13 = data['eps13'][cut_off::]/central_freq


	
	global length
	length = len(s31)





	#constant array for calculating delay polynomials
	global ints
	ints = np.broadcast_to(np.arange(number_n),(length,number_n)).T


	global s32_coeffs
	s32_coeffs = difference_operator_powers(s32)
	global s31_coeffs
	s31_coeffs = difference_operator_powers(s31)
	global s12_coeffs
	s12_coeffs = difference_operator_powers(s12)
	global s13_coeffs
	s13_coeffs = difference_operator_powers(s13)
	global s21_coeffs
	s21_coeffs = difference_operator_powers(s21)
	global s23_coeffs
	s23_coeffs = difference_operator_powers(s23)
	global eps32_coeffs
	eps32_coeffs = difference_operator_powers(eps32)
	global eps31_coeffs
	eps31_coeffs = difference_operator_powers(eps31)
	global eps12_coeffs
	eps12_coeffs = difference_operator_powers(eps12)
	global eps13_coeffs
	eps13_coeffs = difference_operator_powers(eps13)
	global eps21_coeffs
	eps21_coeffs = difference_operator_powers(eps21)
	global eps23_coeffs
	eps23_coeffs = difference_operator_powers(eps23)
	global tau32_coeffs
	tau32_coeffs = difference_operator_powers(tau32)
	global tau31_coeffs
	tau31_coeffs = difference_operator_powers(tau31)
	global tau12_coeffs
	tau12_coeffs = difference_operator_powers(tau12)
	global tau13_coeffs
	tau13_coeffs = difference_operator_powers(tau13)
	global tau21_coeffs
	tau21_coeffs = difference_operator_powers(tau21)
	global tau23_coeffs
	tau23_coeffs = difference_operator_powers(tau23)

	del data
	global window
	window = kaiser(length,kaiser_beta(320))

	global f_band
	f_band = np.fft.rfftfreq(length,1/f_s)
	global indices_f_band
	indices_f_band = np.where(np.logical_and(f_band>=f_min, f_band<=f_max))
	f_band=f_band[indices_f_band]

	global Sy_PM
	Sy_PM = S_y_proof_mass_new_frac_freq(f_band)
	global Sy_OP
	Sy_OP = S_y_OMS_frac_freq(f_band)
	global a
	global b_
	a,b_ = covariance_equal_arm(f_band,Sy_OP,Sy_PM)
	global A_,B_
	#Needed in inverse calculation
	A_ = a**2 - b_**2
	B_ = b_**2 - a*b_
	global log_term_factor,determinant,log_term_determinant
	log_term_factor = 3*np.log(np.pi)
	determinant = a*A_+2*b_*B_
	log_term_determinant = np.log(determinant)
	
	global low_minit1
	low_minit1 = -np.pi
	global high_minit1
	high_minit1 = np.pi


	global low_semi_major
	low_semi_major = 1.493e11# where LISA Constants is 149597870700.0
	global high_semi_major #Estimate from Fig 6 Trajectory Design Paper (for 10 years; way conservative)
	high_semi_major = 1.496e11
	global low_omega 
	low_omega = 0.0
	global high_omega
	high_omega = 2.0*np.pi

	global low_arg_per
	low_arg_per = 0.0
	global high_arg_per 
	high_arg_per = np.pi

	global low_eccentricity # see fig 6 trajectory design paper
	low_eccentricity = 0.004	
	global high_eccentricity 
	high_eccentricity = 0.0055

	global low_inclination# see fig 6 trajectory design paper
	low_inclination = 0.39*np.pi/180.0
	global high_inclination 
	high_inclination = 0.6*np.pi/180.0
	
	global delta
	delta = 5.0/8.0 
	
	global Omega_1
	Omega_1 = np.pi/2.0
	
	global AU
	AU = ASTRONOMICAL_UNIT
	

