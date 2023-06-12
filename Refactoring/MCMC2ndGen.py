
import sys

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import kaiser,kaiser_beta
import time
from scipy.stats import norm
import numpy as np
from lisaconstants import GM_SUN,c,ASTRONOMICAL_YEAR,ASTRONOMICAL_UNIT,SUN_SCHWARZSCHILD_RADIUS,OBLIQUITY # Constant values set below to avoid importing lisaconstant
import zeus
from multiprocessing import Pool
import h5py


def difference_operator_powers(data):

    difference_coefficients = np.zeros((number_n+1,length))
    difference_coefficients[0] = data
    #delta_one = np.roll(data,1)
    #delta_one[0] = 0.0
    #difference_coefficients[1] = data-delta_one
    
    for i in np.arange(1,number_n+1):
        sum_for_this_power = np.zeros(length)
        for j in np.arange(i+1):

            data_rolled = np.roll(data,j)
            data_rolled[:j] = 0.0

            sum_for_this_power = sum_for_this_power + (-1)**j*math.comb(i, j)*data_rolled

        difference_coefficients[i] = sum_for_this_power/np.math.factorial(i)

    return difference_coefficients.T




def filters_lagrange_2_0(D):

    D=D*f_samp
    integer_part, d_frac = np.divmod(D,1)

    integer_part = integer_part-p
    d_frac = d_frac+p

    delay_polynomials = np.ones((number_n+1,length))

    factors = -1*d_frac+ints

    delay_polynomials[1:number_n+1] = np.cumprod(factors,axis=0)

    return delay_polynomials,int(integer_part[0])

def trim_data(data,filter_array):


    data=np.roll(data,filter_array[1],axis=0)
    data[:filter_array[1]] = 0.0
    val = np.einsum('ij,ji->i',data,filter_array[0],optimize=einsum_path_to_use[0])

    return val





def S_y_proof_mass_new_frac_freq(f):

    pm_here =  np.power(2.4e-15,2)*(1+np.power(4.0e-4/f,2))*(1+np.power(f/8.0e-3,4))
    return pm_here*np.power(2*np.pi*f*c,-2)


def S_y_OMS_frac_freq(f):

    op_here =  np.power(1.5e-11,2)*np.power(2*np.pi*f/c,2)*(1+np.power(2.0e-3/f,4))
    return op_here


def orbital_parameters(semi_major,inclination):


    orbital_freq=np.sqrt(GM_SUN/semi_major**3)

    cos_inclination = np.cos(inclination)
    sin_inclination = np.sin(inclination)

    return orbital_freq,cos_inclination,sin_inclination


def s_c_positions(psi_here,eccentricity,cos_inclination,sin_inclination,semi_major,orbital_freq,omega,arg_per,k):

	lambda_k = omega  + arg_per
	zeta_t = semi_major*(np.cos(psi_here) - eccentricity)
	eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.sin(psi_here)
	positions = np.empty((3,length))
	
	positions[0] = (np.cos(omega)*np.cos(arg_per) - np.sin(omega)*np.sin(arg_per)*cos_inclination)*zeta_t - (np.cos(omega)*np.sin(arg_per) + np.sin(omega)*np.cos(arg_per)*cos_inclination)*eta_t #x(t)
	positions[1] = (np.sin(omega)*np.cos(arg_per) + np.cos(omega)*np.sin(arg_per)*cos_inclination)*zeta_t - (np.sin(omega)*np.sin(arg_per) - np.cos(omega)*np.cos(arg_per)*cos_inclination)*eta_t #y(t)
	positions[2] = np.sin(arg_per)*sin_inclination*zeta_t + np.cos(arg_per)*sin_inclination*eta_t #z(t)

	return positions


def s_c_velocities(psi_here,eccentricity,cos_inclination,sin_inclination,semi_major,orbital_freq,omega,arg_per,k):

	psi_dot = orbital_freq/(1.0-eccentricity*np.cos(psi_here))
	lambda_k = omega  + arg_per
	zeta_t = semi_major*(np.cos(psi_here) - eccentricity)
	d_zeta_t = -1*semi_major*np.sin(psi_here)*psi_dot
	eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.sin(psi_here)
	d_eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.cos(psi_here)*psi_dot
	velocities = np.empty((3,length))

	velocities[0] = (np.cos(omega)*np.cos(arg_per) - np.sin(omega)*np.sin(arg_per)*cos_inclination)*d_zeta_t - (np.cos(omega)*np.sin(arg_per) + np.sin(omega)*np.cos(arg_per)*cos_inclination)*d_eta_t #x(t)
	velocities[1] = (np.sin(omega)*np.cos(arg_per) + np.cos(omega)*np.sin(arg_per)*cos_inclination)*d_zeta_t - (np.sin(omega)*np.sin(arg_per) - np.cos(omega)*np.cos(arg_per)*cos_inclination)*d_eta_t #y(t)
	velocities[2] = np.sin(arg_per)*sin_inclination*d_zeta_t + np.cos(arg_per)*sin_inclination*d_eta_t #z(t)

	return velocities



def s_c_accelerations(position_here,semi_major,orbital_freq):
    
    return -1.0*np.power(semi_major,3)*orbital_freq**2*position_here/np.power(np.sqrt(position_here[0]**2+position_here[1]**2+position_here[2]**2),3)



def shapiro(pos_i,pos_j):

    mag_pos_j = np.sqrt(pos_j[0]**2+pos_j[1]**2+pos_j[2]**2)     
    mag_pos_i = np.sqrt(pos_i[0]**2+pos_i[1]**2+pos_i[2]**2)    
    diff = pos_j-pos_i
    mag_diff = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

    return 2.0*GM_SUN/(c**2)*np.log((mag_pos_j + mag_pos_i + mag_diff)/(mag_pos_j+mag_pos_i-mag_diff))



def psi(m_init1,eccentricity,orbital_freq,k,t):
    m = m_init1 + orbital_freq*(t-t_init)

    psi_return = m + (eccentricity-np.power(eccentricity,3)/8.0)*np.sin(m) + 0.5*eccentricity**2*np.sin(2.0*m)  + 3.0/8*np.power(eccentricity,3)*np.sin(3.0*m)
    for i in np.arange(2):
        error =psi_return - eccentricity * np.sin(psi_return) - m
        psi_return -= error / (1.0 - eccentricity * np.cos(psi_return)) 

    return psi_return

def theta(k):
    return 2.0*np.pi*(k-1)/3.0

def delta_tau(psi_here,m_init1,eccentricity,orbital_freq,semi_major,k,t):
    psi_here_init = psi(m_init1,eccentricity,orbital_freq,k,t_init)

    return -3.0/2.0*(orbital_freq*semi_major/c)**2*(t-t_init) - 2.0*(orbital_freq*semi_major/c)**2*eccentricity/orbital_freq*(np.sin(psi_here)-np.sin(psi_here_init))



def time_dependence(m_init1,semi_major,eccentricity,inclination,omega_init,arg_per):


	delay_in_time = np.empty((6,length))
	if is_tcb==False:
		mprs = np.empty((6,length))

	orbital_freq,cos_inclination,sin_inclination = orbital_parameters(semi_major,inclination)

	for i,k,z in zip(np.arange(6),np.array([2,3,3,1,1,2]),np.array([3,2,1,3,2,1])):
		psi_i = psi(m_init1[z-1],eccentricity[z-1],orbital_freq[z-1],z,tcb_times[z-1])
		psi_j = psi(m_init1[k-1],eccentricity[k-1],orbital_freq[k-1],k,tcb_times[z-1])
		
		position_i = s_c_positions(psi_i,eccentricity[z-1],cos_inclination[z-1],sin_inclination[z-1],semi_major[z-1],orbital_freq[z-1],omega_init[z-1],arg_per[z-1],z)
		position_j = s_c_positions(psi_j,eccentricity[k-1],cos_inclination[k-1],sin_inclination[k-1],semi_major[k-1],orbital_freq[k-1],omega_init[k-1],arg_per[k-1],k)

		Dij = position_i-position_j


		magDij = np.sqrt(Dij[0]**2+Dij[1]**2+Dij[2]**2)


		velocity_j = s_c_velocities(psi_j,eccentricity[k-1],cos_inclination[k-1],sin_inclination[k-1],semi_major[k-1],orbital_freq[k-1],omega_init[k-1],arg_per[k-1],k)

		second_term = np.sum(Dij*velocity_j,axis=0)/(c**2)

		mag_v_j = np.sqrt(velocity_j[0]**2+velocity_j[1]**2+velocity_j[2]**2)
		third_term = magDij/(2.0*np.power(c,3))*(mag_v_j**2 + np.power(np.sum(velocity_j*Dij,axis=0)/magDij,2) -np.sum(s_c_accelerations(position_j,semi_major[k-1],orbital_freq[k-1])*Dij,axis=0))

		delay_in_time[i] = magDij/c + second_term + third_term +  shapiro(position_i,position_j)/c

		if is_tcb==False:

			mprs[i] = delay_in_time[i]+ delta_tau(psi_i,m_init1[z-1],eccentricity[z-1],orbital_freq[z-1],semi_major[z-1],z,tcb_times[z-1]) - delta_tau(psi_j,m_init1[k-1],eccentricity[k-1],orbital_freq[k-1],semi_major[k-1],k,tcb_times[z-1] - delay_in_time[i])
			
	if is_tcb==True:
		return delay_in_time
	else:
		return mprs





def nested_delay_application(delay_array_here,list_delays):
    number_delays = len(list_delays)
    
    delays = np.array([delay_array_here[j] for j in list_delays])


    delay_dot_array_here = np.gradient(delays,1/f_s,axis=1,edge_order=2)

    correction_factor =np.zeros(length)

    for i in np.arange(number_delays):
        for j in np.arange(i+1,number_delays):

            correction_factor+=delays[i]*delay_dot_array_here[j]          


    doppler_factor = np.sum(delay_dot_array_here,axis=0)

    commutative_sum = np.sum(delays,axis=0)



    return commutative_sum, np.gradient(commutative_sum,1/f_s), correction_factor


def x_combo_2_0(delay_array):



    L12 = nested_delay_application(delay_array,np.array([5]))
    L12_L21 = nested_delay_application(delay_array,np.array([5,4]))
    L12_L21_L13 = nested_delay_application(delay_array,np.array([5,4,2]))
    L12_L21_L13_L31 = nested_delay_application(delay_array,np.array([5,4,2,3]))
    L12_L21_L13_L31_L13 = nested_delay_application(delay_array,np.array([5,4,2,3,2]))
    L12_L21_L13_L31_L13_L31 = nested_delay_application(delay_array,np.array([5,4,2,3,2,3]))
    L12_L21_L13_L31_L13_L31_L12 = nested_delay_application(delay_array,np.array([5,4,2,3,2,3,5]))
    L12_L21_L13_L31_L13_L31_L12_L21 = nested_delay_application(delay_array,np.array([5,4,2,3,2,3,5,4]))

    L13 = nested_delay_application(delay_array,np.array([2]))
    L13_L31 = nested_delay_application(delay_array,np.array([2,3]))
    L13_L31_L12 = nested_delay_application(delay_array,np.array([2,3,5]))
    L13_L31_L12_L21 = nested_delay_application(delay_array,np.array([2,3,5,4]))
    L13_L31_L12_L21_L12 = nested_delay_application(delay_array,np.array([2,3,5,4,5]))
    L13_L31_L12_L21_L12_L21 = nested_delay_application(delay_array,np.array([2,3,5,4,5,4]))
    L13_L31_L12_L21_L12_L21_L13 = nested_delay_application(delay_array,np.array([2,3,5,4,5,4,2]))
    L13_L31_L12_L21_L12_L21_L13_L31 = nested_delay_application(delay_array,np.array([2,3,5,4,5,4,2,3]))

    filter_L12 = filters_lagrange_2_0(L12[0])
    filter_L12_L21 = filters_lagrange_2_0(L12_L21[0])
    filter_L12_L21_L13 = filters_lagrange_2_0(L12_L21_L13[0])
    filter_L12_L21_L13_L31 = filters_lagrange_2_0(L12_L21_L13_L31[0])
    filter_L12_L21_L13_L31_L13 = filters_lagrange_2_0(L12_L21_L13_L31_L13[0])
    filter_L12_L21_L13_L31_L13_L31 = filters_lagrange_2_0(L12_L21_L13_L31_L13_L31[0])
    filter_L12_L21_L13_L31_L13_L31_L12 = filters_lagrange_2_0(L12_L21_L13_L31_L13_L31_L12[0])    
    filter_L12_L21_L13_L31_L13_L31_L12_L21 = filters_lagrange_2_0(L12_L21_L13_L31_L13_L31_L12_L21[0])

    filter_L13 = filters_lagrange_2_0(L13[0])
    filter_L13_L31 = filters_lagrange_2_0(L13_L31[0])
    filter_L13_L31_L12 = filters_lagrange_2_0(L13_L31_L12[0])
    filter_L13_L31_L12_L21 = filters_lagrange_2_0(L13_L31_L12_L21[0])
    filter_L13_L31_L12_L21_L12 = filters_lagrange_2_0(L13_L31_L12_L21_L12[0])
    filter_L13_L31_L12_L21_L12_L21 = filters_lagrange_2_0(L13_L31_L12_L21_L12_L21[0])
    filter_L13_L31_L12_L21_L12_L21_L13 = filters_lagrange_2_0(L13_L31_L12_L21_L12_L21_L13[0])
    filter_L13_L31_L12_L21_L12_L21_L13_L31 = filters_lagrange_2_0(L13_L31_L12_L21_L12_L21_L13_L31[0])

    x_combo = np.zeros(length)
    x_combo = x_combo + (s12 + 0.5*(tau12-eps12)) 

    next_term = trim_data((tau21_coeffs-eps21_coeffs) + s21_coeffs,filter_L12) 
    x_combo = x_combo + ((1-L12[1])*(next_term + np.gradient(next_term,1/f_s)*L12[2]))

    next_term = trim_data(0.5*(tau12_coeffs-eps12_coeffs) + s13_coeffs + 0.5*(tau13_coeffs-eps13_coeffs) + 0.5*(tau12_coeffs-tau13_coeffs),filter_L12_L21) 
    x_combo = x_combo + ((1-L12_L21[1])*(next_term + np.gradient(next_term,1/f_s)*L12_L21[2]))

    next_term = trim_data(0.5*(tau31_coeffs-eps31_coeffs) + s31_coeffs + 0.5*(tau31_coeffs-eps31_coeffs),filter_L12_L21_L13) 
    x_combo = x_combo + ((1-L12_L21_L13[1])*(next_term + np.gradient(next_term,1/f_s)*L12_L21_L13[2]))

    next_term = trim_data((tau13_coeffs-eps13_coeffs) + s13_coeffs,filter_L12_L21_L13_L31) 
    x_combo = x_combo + ((1-L12_L21_L13_L31[1])*(next_term + np.gradient(next_term,1/f_s)*L12_L21_L13_L31[2]))

    next_term = trim_data((tau31_coeffs-eps31_coeffs) + s31_coeffs,filter_L12_L21_L13_L31_L13) 
    x_combo = x_combo + ((1-L12_L21_L13_L31_L13[1])*(next_term + np.gradient(next_term,1/f_s)*L12_L21_L13_L31_L13[2]))

    next_term = trim_data(0.5*(tau13_coeffs-eps13_coeffs) + 0.5*(tau13_coeffs-tau12_coeffs) + s12_coeffs + 0.5*(tau12_coeffs-eps12_coeffs),filter_L12_L21_L13_L31_L13_L31) 
    x_combo = x_combo + ((1-L12_L21_L13_L31_L13_L31[1])*(next_term + np.gradient(next_term,1/f_s)*L12_L21_L13_L31_L13_L31[2]))

    next_term = trim_data((tau21_coeffs-eps21_coeffs) + s21_coeffs,filter_L12_L21_L13_L31_L13_L31_L12) 
    x_combo = x_combo + ((1-L12_L21_L13_L31_L13_L31_L12[1])*(next_term + np.gradient(next_term,1/f_s)*L12_L21_L13_L31_L13_L31_L12[2]))

    next_term = trim_data(0.5*(tau12_coeffs-eps12_coeffs),filter_L12_L21_L13_L31_L13_L31_L12_L21) 
    x_combo = x_combo + ((1-L12_L21_L13_L31_L13_L31_L12_L21[1])*(next_term + np.gradient(next_term,1/f_s)*L12_L21_L13_L31_L13_L31_L12_L21[2]))

    #BEGIN NEGATIVE
    x_combo_minus = np.zeros(length)

    x_combo_minus = x_combo_minus + (s13 + 0.5*(tau13-eps13) + 0.5*(tau12-tau13)) 

    next_term = trim_data((tau31_coeffs-eps31_coeffs) + s31_coeffs,filter_L13) 
    x_combo_minus = x_combo_minus + ((1-L13[1])*(next_term + np.gradient(next_term,1/f_s)*L13[2]))

    next_term = trim_data(0.5*(tau13_coeffs-eps13_coeffs) + 0.5*(tau13_coeffs-tau12_coeffs) + s12_coeffs + 0.5*(tau12_coeffs-eps12_coeffs),filter_L13_L31) 
    x_combo_minus = x_combo_minus + ((1-L13_L31[1])*(next_term + np.gradient(next_term,1/f_s)*L13_L31[2]))

    next_term = trim_data((tau21_coeffs-eps21_coeffs) + s21_coeffs,filter_L13_L31_L12) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12[1])*(next_term + np.gradient(next_term,1/f_s)*L13_L31_L12[2]))

    next_term = trim_data((tau12_coeffs-eps12_coeffs) + s12_coeffs,filter_L13_L31_L12_L21) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21[1])*(next_term + np.gradient(next_term,1/f_s)*L13_L31_L12_L21[2]))

    next_term = trim_data((tau21_coeffs-eps21_coeffs) + s21_coeffs,filter_L13_L31_L12_L21_L12) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21_L12[1])*(next_term + np.gradient(next_term,1/f_s)*L13_L31_L12_L21_L12[2]))

    next_term = trim_data(0.5*(tau12_coeffs-eps12_coeffs) + s13_coeffs + 0.5*(tau13_coeffs-eps13_coeffs) + 0.5*(tau12_coeffs-tau13_coeffs),filter_L13_L31_L12_L21_L12_L21) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21_L12_L21[1])*(next_term + np.gradient(next_term,1/f_s)*L13_L31_L12_L21_L12_L21[2]))

    next_term = trim_data((tau31_coeffs-eps31_coeffs) + s31_coeffs,filter_L13_L31_L12_L21_L12_L21_L13) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21_L12_L21_L13[1])*(next_term + np.gradient(next_term,1/f_s)*L13_L31_L12_L21_L12_L21_L13[2]))

    next_term = trim_data(0.5*(tau13_coeffs-eps13_coeffs) + 0.5*(tau13_coeffs-tau12_coeffs),filter_L13_L31_L12_L21_L12_L21_L13_L31) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21_L12_L21_L13_L31[1])*(next_term + np.gradient(next_term,1/f_s)*L13_L31_L12_L21_L12_L21_L13_L31[2]))

    x_combo = x_combo - x_combo_minus
    '''
    #np.savetxt('x_combo_full.dat',x_combo)
    plt.plot(s31,label = 's31')
    plt.plot(x_combo,label='x combo')
    plt.plot(window*x_combo,label = 'Kaiser windowed')
    plt.legend()
    plt.show()    
    '''

    
    x_f = np.fft.rfft(window*x_combo,norm='ortho')[indices_f_band]

    return [np.real(x_f),np.imag(x_f)]




def y_combo_2_0(delay_array):

    L23 = nested_delay_application(delay_array,np.array([1]))
    L23_L32 = nested_delay_application(delay_array,np.array([1,0]))
    L23_L32_L21 = nested_delay_application(delay_array,np.array([1,0,4]))
    L23_L32_L21_L12 = nested_delay_application(delay_array,np.array([1,0,4,5]))
    L23_L32_L21_L12_L21 = nested_delay_application(delay_array,np.array([1,0,4,5,4]))
    L23_L32_L21_L12_L21_L12 = nested_delay_application(delay_array,np.array([1,0,4,5,4,5]))
    L23_L32_L21_L12_L21_L12_L23 = nested_delay_application(delay_array,np.array([1,0,4,5,4,5,1]))
    L23_L32_L21_L12_L21_L12_L23_L32 = nested_delay_application(delay_array,np.array([1,0,4,5,4,5,1,0]))

    L21 = nested_delay_application(delay_array,np.array([4]))
    L21_L12 = nested_delay_application(delay_array,np.array([4,5]))
    L21_L12_L23 = nested_delay_application(delay_array,np.array([4,5,1]))
    L21_L12_L23_L32 = nested_delay_application(delay_array,np.array([4,5,1,0]))
    L21_L12_L23_L32_L23 = nested_delay_application(delay_array,np.array([4,5,1,0,1]))
    L21_L12_L23_L32_L23_L32 = nested_delay_application(delay_array,np.array([4,5,1,0,1,0]))
    L21_L12_L23_L32_L23_L32_L21 = nested_delay_application(delay_array,np.array([4,5,1,0,1,0,4]))
    L21_L12_L23_L32_L23_L32_L21_L12 = nested_delay_application(delay_array,np.array([4,5,1,0,1,0,4,5]))


    filter_L23 = filters_lagrange_2_0(L23[0])
    filter_L23_L32 = filters_lagrange_2_0(L23_L32[0])
    filter_L23_L32_L21 = filters_lagrange_2_0(L23_L32_L21[0])
    filter_L23_L32_L21_L12 = filters_lagrange_2_0(L23_L32_L21_L12[0])
    filter_L23_L32_L21_L12_L21 = filters_lagrange_2_0(L23_L32_L21_L12_L21[0])
    filter_L23_L32_L21_L12_L21_L12 = filters_lagrange_2_0(L23_L32_L21_L12_L21_L12[0])
    filter_L23_L32_L21_L12_L21_L12_L23 = filters_lagrange_2_0(L23_L32_L21_L12_L21_L12_L23[0])    
    filter_L23_L32_L21_L12_L21_L12_L23_L32 = filters_lagrange_2_0(L23_L32_L21_L12_L21_L12_L23_L32[0])

    filter_L21 = filters_lagrange_2_0(L21[0])
    filter_L21_L12 = filters_lagrange_2_0(L21_L12[0])
    filter_L21_L12_L23 = filters_lagrange_2_0(L21_L12_L23[0])
    filter_L21_L12_L23_L32 = filters_lagrange_2_0(L21_L12_L23_L32[0])
    filter_L21_L12_L23_L32_L23 = filters_lagrange_2_0(L21_L12_L23_L32_L23[0])
    filter_L21_L12_L23_L32_L23_L32 = filters_lagrange_2_0(L21_L12_L23_L32_L23_L32[0])
    filter_L21_L12_L23_L32_L23_L32_L21 = filters_lagrange_2_0(L21_L12_L23_L32_L23_L32_L21[0])
    filter_L21_L12_L23_L32_L23_L32_L21_L12 = filters_lagrange_2_0(L21_L12_L23_L32_L23_L32_L21_L12[0])

    y_combo = np.zeros(length)
    y_combo = y_combo + (s23 + 0.5*(tau23-eps23)) 

    next_term = trim_data((tau32_coeffs-eps32_coeffs) + s32_coeffs,filter_L23) 
    y_combo = y_combo + ((1-L23[1])*(next_term + np.gradient(next_term,1/f_s)*L23[2]))

    next_term = trim_data(0.5*(tau23_coeffs-eps23_coeffs) + s21_coeffs + 0.5*(tau21_coeffs-eps21_coeffs) + 0.5*(tau23_coeffs-tau21_coeffs),filter_L23_L32) 
    y_combo = y_combo + ((1-L23_L32[1])*(next_term + np.gradient(next_term,1/f_s)*L23_L32[2]))

    next_term = trim_data(0.5*(tau12_coeffs-eps12_coeffs) + s12_coeffs + 0.5*(tau12_coeffs-eps12_coeffs),filter_L23_L32_L21) 
    y_combo = y_combo + ((1-L23_L32_L21[1])*(next_term + np.gradient(next_term,1/f_s)*L23_L32_L21[2]))

    next_term = trim_data((tau21_coeffs-eps21_coeffs) + s21_coeffs,filter_L23_L32_L21_L12) 
    y_combo = y_combo + ((1-L23_L32_L21_L12[1])*(next_term + np.gradient(next_term,1/f_s)*L23_L32_L21_L12[2]))

    next_term = trim_data((tau12_coeffs-eps12_coeffs) + s12_coeffs,filter_L23_L32_L21_L12_L21) 
    y_combo = y_combo + ((1-L23_L32_L21_L12_L21[1])*(next_term + np.gradient(next_term,1/f_s)*L23_L32_L21_L12_L21[2]))

    next_term = trim_data(0.5*(tau21_coeffs-eps21_coeffs) + 0.5*(tau21_coeffs-tau23_coeffs) + s23_coeffs + 0.5*(tau23_coeffs-eps23_coeffs),filter_L23_L32_L21_L12_L21_L12) 
    y_combo = y_combo + ((1-L23_L32_L21_L12_L21_L12[1])*(next_term + np.gradient(next_term,1/f_s)*L23_L32_L21_L12_L21_L12[2]))

    next_term = trim_data((tau32_coeffs-eps32_coeffs) + s32_coeffs,filter_L23_L32_L21_L12_L21_L12_L23) 
    y_combo = y_combo + ((1-L23_L32_L21_L12_L21_L12_L23[1])*(next_term + np.gradient(next_term,1/f_s)*L23_L32_L21_L12_L21_L12_L23[2]))

    next_term = trim_data(0.5*(tau23_coeffs-eps23_coeffs),filter_L23_L32_L21_L12_L21_L12_L23_L32) 
    y_combo = y_combo + ((1-L23_L32_L21_L12_L21_L12_L23_L32[1])*(next_term + np.gradient(next_term,1/f_s)*L23_L32_L21_L12_L21_L12_L23_L32[2]))

    #BEGIN NEGATIVE
    y_combo_minus = np.zeros(length)

    y_combo_minus = y_combo_minus + (s21 + 0.5*(tau21-eps21) + 0.5*(tau23-tau21)) 

    next_term = trim_data((tau12_coeffs-eps12_coeffs) + s12_coeffs,filter_L21) 
    y_combo_minus = y_combo_minus + ((1-L21[1])*(next_term + np.gradient(next_term,1/f_s)*L21[2]))

    next_term = trim_data(0.5*(tau21_coeffs-eps21_coeffs) + 0.5*(tau21_coeffs-tau23_coeffs) + s23_coeffs + 0.5*(tau23_coeffs-eps23_coeffs),filter_L21_L12) 
    y_combo_minus = y_combo_minus + ((1-L21_L12[1])*(next_term + np.gradient(next_term,1/f_s)*L21_L12[2]))

    next_term = trim_data((tau32_coeffs-eps32_coeffs) + s32_coeffs,filter_L21_L12_L23) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23[1])*(next_term + np.gradient(next_term,1/f_s)*L21_L12_L23[2]))

    next_term = trim_data((tau23_coeffs-eps23_coeffs) + s23_coeffs,filter_L21_L12_L23_L32) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32[1])*(next_term + np.gradient(next_term,1/f_s)*L21_L12_L23_L32[2]))

    next_term = trim_data((tau32_coeffs-eps32_coeffs) + s32_coeffs,filter_L21_L12_L23_L32_L23) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32_L23[1])*(next_term + np.gradient(next_term,1/f_s)*L21_L12_L23_L32_L23[2]))

    next_term = trim_data(0.5*(tau23_coeffs-eps23_coeffs) + s21_coeffs + 0.5*(tau21_coeffs-eps21_coeffs) + 0.5*(tau23_coeffs-tau21_coeffs),filter_L21_L12_L23_L32_L23_L32) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32_L23_L32[1])*(next_term + np.gradient(next_term,1/f_s)*L21_L12_L23_L32_L23_L32[2]))

    next_term = trim_data((tau12_coeffs-eps12_coeffs) + s12_coeffs,filter_L21_L12_L23_L32_L23_L32_L21) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32_L23_L32_L21[1])*(next_term + np.gradient(next_term,1/f_s)*L21_L12_L23_L32_L23_L32_L21[2]))

    next_term = trim_data(0.5*(tau21_coeffs-eps21_coeffs) + 0.5*(tau21_coeffs-tau23_coeffs),filter_L21_L12_L23_L32_L23_L32_L21_L12) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32_L23_L32_L21_L12[1])*(next_term + np.gradient(next_term,1/f_s)*L21_L12_L23_L32_L23_L32_L21_L12[2]))

    y_combo = y_combo - y_combo_minus
    
    '''
    #np.savetxt('y_combo_full.dat',y_combo)
    plt.plot(s12,label = 's12_')
    plt.plot(y_combo,label='y combo')
    plt.plot(window*y_combo,label = 'Kaiser windowed')
    plt.legend()
    plt.show()    
    '''

    
    y_f = np.fft.rfft(window*y_combo,norm='ortho')[indices_f_band]

    return [np.real(y_f),np.imag(y_f)]


# In[25]:


def z_combo_2_0(delay_array):

    L31 = nested_delay_application(delay_array,np.array([3]))
    L31_L13 = nested_delay_application(delay_array,np.array([3,2]))
    L31_L13_L32 = nested_delay_application(delay_array,np.array([3,2,0]))
    L31_L13_L32_L23 = nested_delay_application(delay_array,np.array([3,2,0,1]))
    L31_L13_L32_L23_L32 = nested_delay_application(delay_array,np.array([3,2,0,1,0]))
    L31_L13_L32_L23_L32_L23 = nested_delay_application(delay_array,np.array([3,2,0,1,0,1]))
    L31_L13_L32_L23_L32_L23_L31 = nested_delay_application(delay_array,np.array([3,2,0,1,0,1,3]))
    L31_L13_L32_L23_L32_L23_L31_L13 = nested_delay_application(delay_array,np.array([3,2,0,1,0,1,3,2]))

    L32 = nested_delay_application(delay_array,np.array([0]))
    L32_L23 = nested_delay_application(delay_array,np.array([0,1]))
    L32_L23_L31 = nested_delay_application(delay_array,np.array([0,1,3]))
    L32_L23_L31_L13 = nested_delay_application(delay_array,np.array([0,1,3,2]))
    L32_L23_L31_L13_L31 = nested_delay_application(delay_array,np.array([0,1,3,2,3]))
    L32_L23_L31_L13_L31_L13 = nested_delay_application(delay_array,np.array([0,1,3,2,3,2]))
    L32_L23_L31_L13_L31_L13_L32 = nested_delay_application(delay_array,np.array([0,1,3,2,3,2,0]))
    L32_L23_L31_L13_L31_L13_L32_L23 = nested_delay_application(delay_array,np.array([0,1,3,2,3,2,0,1]))


    filter_L31 = filters_lagrange_2_0(L31[0])
    filter_L31_L13 = filters_lagrange_2_0(L31_L13[0])
    filter_L31_L13_L32 = filters_lagrange_2_0(L31_L13_L32[0])
    filter_L31_L13_L32_L23 = filters_lagrange_2_0(L31_L13_L32_L23[0])
    filter_L31_L13_L32_L23_L32 = filters_lagrange_2_0(L31_L13_L32_L23_L32[0])
    filter_L31_L13_L32_L23_L32_L23 = filters_lagrange_2_0(L31_L13_L32_L23_L32_L23[0])
    filter_L31_L13_L32_L23_L32_L23_L31 = filters_lagrange_2_0(L31_L13_L32_L23_L32_L23_L31[0])    
    filter_L31_L13_L32_L23_L32_L23_L31_L13 = filters_lagrange_2_0(L31_L13_L32_L23_L32_L23_L31_L13[0])

    filter_L32 = filters_lagrange_2_0(L32[0])
    filter_L32_L23 = filters_lagrange_2_0(L32_L23[0])
    filter_L32_L23_L31 = filters_lagrange_2_0(L32_L23_L31[0])
    filter_L32_L23_L31_L13 = filters_lagrange_2_0(L32_L23_L31_L13[0])
    filter_L32_L23_L31_L13_L31 = filters_lagrange_2_0(L32_L23_L31_L13_L31[0])
    filter_L32_L23_L31_L13_L31_L13 = filters_lagrange_2_0(L32_L23_L31_L13_L31_L13[0])
    filter_L32_L23_L31_L13_L31_L13_L32 = filters_lagrange_2_0(L32_L23_L31_L13_L31_L13_L32[0])
    filter_L32_L23_L31_L13_L31_L13_L32_L23 = filters_lagrange_2_0(L32_L23_L31_L13_L31_L13_L32_L23[0])

        
    z_combo = np.zeros(length)
    z_combo = z_combo + (s31 + 0.5*(tau31-eps31)) 

    next_term = trim_data((tau13_coeffs-eps13_coeffs) + s13_coeffs,filter_L31) 
    z_combo = z_combo + ((1-L31[1])*(next_term + np.gradient(next_term,1/f_s)*L31[2]))

    next_term = trim_data(0.5*(tau31_coeffs-eps31_coeffs) + s32_coeffs + 0.5*(tau32_coeffs-eps32_coeffs) + 0.5*(tau31_coeffs-tau32_coeffs),filter_L31_L13) 
    z_combo = z_combo + ((1-L31_L13[1])*(next_term + np.gradient(next_term,1/f_s)*L31_L13[2]))

    next_term = trim_data(0.5*(tau23_coeffs-eps23_coeffs) + s23_coeffs + 0.5*(tau23_coeffs-eps23_coeffs),filter_L31_L13_L32) 
    z_combo = z_combo + ((1-L31_L13_L32[1])*(next_term + np.gradient(next_term,1/f_s)*L31_L13_L32[2]))

    next_term = trim_data((tau32_coeffs-eps32_coeffs) + s32_coeffs,filter_L31_L13_L32_L23) 
    z_combo = z_combo + ((1-L31_L13_L32_L23[1])*(next_term + np.gradient(next_term,1/f_s)*L31_L13_L32_L23[2]))

    next_term = trim_data((tau23_coeffs-eps23_coeffs) + s23_coeffs,filter_L31_L13_L32_L23_L32) 
    z_combo = z_combo + ((1-L31_L13_L32_L23_L32[1])*(next_term + np.gradient(next_term,1/f_s)*L31_L13_L32_L23_L32[2]))

    next_term = trim_data(0.5*(tau32_coeffs-eps32_coeffs) + 0.5*(tau32_coeffs-tau31_coeffs) + s31_coeffs + 0.5*(tau31_coeffs-eps31_coeffs),filter_L31_L13_L32_L23_L32_L23) 
    z_combo = z_combo + ((1-L31_L13_L32_L23_L32_L23[1])*(next_term + np.gradient(next_term,1/f_s)*L31_L13_L32_L23_L32_L23[2]))

    next_term = trim_data((tau13_coeffs-eps13_coeffs) + s13_coeffs,filter_L31_L13_L32_L23_L32_L23_L31) 
    z_combo = z_combo + ((1-L31_L13_L32_L23_L32_L23_L31[1])*(next_term + np.gradient(next_term,1/f_s)*L31_L13_L32_L23_L32_L23_L31[2]))

    next_term = trim_data(0.5*(tau31_coeffs-eps31_coeffs),filter_L31_L13_L32_L23_L32_L23_L31_L13) 
    z_combo = z_combo + ((1-L31_L13_L32_L23_L32_L23_L31_L13[1])*(next_term + np.gradient(next_term,1/f_s)*L31_L13_L32_L23_L32_L23_L31_L13[2]))

    #BEGIN NEGATIVE
    z_combo_minus = np.zeros(length)

    z_combo_minus = z_combo_minus + (s32 + 0.5*(tau32-eps32) + 0.5*(tau31-tau32)) 

    next_term = trim_data((tau23_coeffs-eps23_coeffs) + s23_coeffs,filter_L32) 
    z_combo_minus = z_combo_minus + ((1-L32[1])*(next_term + np.gradient(next_term,1/f_s)*L32[2]))

    next_term = trim_data(0.5*(tau32_coeffs-eps32_coeffs) + 0.5*(tau32_coeffs-tau31_coeffs) + s31_coeffs + 0.5*(tau31_coeffs-eps31_coeffs),filter_L32_L23) 
    z_combo_minus = z_combo_minus + ((1-L32_L23[1])*(next_term + np.gradient(next_term,1/f_s)*L32_L23[2]))

    next_term = trim_data((tau13_coeffs-eps13_coeffs) + s13_coeffs,filter_L32_L23_L31) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31[1])*(next_term + np.gradient(next_term,1/f_s)*L32_L23_L31[2]))

    next_term = trim_data((tau31_coeffs-eps31_coeffs) + s31_coeffs,filter_L32_L23_L31_L13) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13[1])*(next_term + np.gradient(next_term,1/f_s)*L32_L23_L31_L13[2]))

    next_term = trim_data((tau13_coeffs-eps13_coeffs) + s13_coeffs,filter_L32_L23_L31_L13_L31) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13_L31[1])*(next_term + np.gradient(next_term,1/f_s)*L32_L23_L31_L13_L31[2]))

    next_term = trim_data(0.5*(tau31_coeffs-eps31_coeffs) + s32_coeffs + 0.5*(tau32_coeffs-eps32_coeffs) + 0.5*(tau31_coeffs-tau32_coeffs),filter_L32_L23_L31_L13_L31_L13) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13_L31_L13[1])*(next_term + np.gradient(next_term,1/f_s)*L32_L23_L31_L13_L31_L13[2]))

    next_term = trim_data((tau23_coeffs-eps23_coeffs) + s23_coeffs,filter_L32_L23_L31_L13_L31_L13_L32) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13_L31_L13_L32[1])*(next_term + np.gradient(next_term,1/f_s)*L32_L23_L31_L13_L31_L13_L32[2]))

    next_term = trim_data(0.5*(tau32_coeffs-eps32_coeffs) + 0.5*(tau32_coeffs-tau31_coeffs),filter_L32_L23_L31_L13_L31_L13_L32_L23) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13_L31_L13_L32_L23[1])*(next_term + np.gradient(next_term,1/f_s)*L32_L23_L31_L13_L31_L13_L32_L23[2]))

    z_combo = z_combo - z_combo_minus
    '''
    #np.savetxt('z_combo_full.dat',z_combo)
    plt.plot(s12,label = 's12')
    plt.plot(z_combo,label='z combo')
    plt.plot(window*z_combo,label = 'Kaiser windowed')
    plt.legend()
    plt.show()    
    '''

    
    z_f = np.fft.rfft(window*z_combo,norm='ortho')[indices_f_band]

    return [np.real(z_f),np.imag(z_f)]





def covariance_equal_arm(f,Sy_OP,Sy_PM):
    
    a = 16*np.power(np.sin(2*np.pi*f*avg_L),2)*Sy_OP+(8*np.power(np.sin(4*np.pi*f*avg_L),2)+32*np.power(np.sin(2*np.pi*f*avg_L),2))*Sy_PM
    b_ = -4*np.sin(2*np.pi*f*avg_L)*np.sin(4*np.pi*f*avg_L)*(4*Sy_PM+Sy_OP)

    return 2*a,2*b_



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

	chi_2 = 1/determinant*(A_*(x[0]**2+x[1]**2+y[0]**2+y[1]**2+z[0]**2+z[1]**2) + 2*B_*(x[0]*y[0]+x[1]*y[1]+x[0]*z[0]+x[1]*z[1]+y[0]*z[0]+y[1]*z[1]))
	'''
	plt.semilogx(f_band,chi_2,label='FFT X2 My Code')
	plt.title('chi^2')
	plt.show()
	'''
	value = -1*np.sum(chi_2) - log_term_factor - np.sum(log_term_determinant)
	
	return value,np.sum(chi_2)	  
	
def target_log_prob_fn(state_current):

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


	x_combo = x_combo_2_0(delays_in_time)
	y_combo = y_combo_2_0(delays_in_time)
	z_combo = z_combo_2_0(delays_in_time)

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
	if (val >= low_minit1).all() and (val <= high_minit1).all():
		return 1
	else:
		return 0

def prior_semi_major(val):
	val = np.array(val)
	if (val >= low_semi_major).all() and (val <= high_semi_major).all():
		return 1
	else:
		return 0


		
def prior_omega(val):
	val = np.array(val)
	if (val >= low_omega).all()  and (val <= high_omega).all():
		return 1
	else:
		return 0
		
def prior_arg_per(val):
	val = np.array(val)
	if (val >= low_arg_per).all()  and (val <= high_arg_per).all() :
		return 1
	else:
		return 0
		
def prior_eccentricity(val):
	val = np.array(val)
	if (val >= low_eccentricity).all() and (val <= high_eccentricity).all():
		return 1
	else:
		return 0
		
def prior_inclination(val):
	val = np.array(val)
	if (val >= low_inclination).all() and (val <= high_inclination).all():
		return 1
	else:
		return 0
		

# # Other Parameters Constant in Code	
f_s = 4.0
f_samp = f_s
#lagrange filter length
number_n_data = 7
number_n = number_n_data
p = number_n//2
asd_nu = 28.8 # Hz/rtHz


f_min = 5.0e-4 # (= 0.0009765625)
f_max = 0.1
central_freq=281600000000000.0
L_arm = 2.5e9
avg_L = L_arm/c

static=False
equalarmlength=False
keplerian=False
esa=True




matrix=True

is_tcb = True


low_minit1 = -np.pi
high_minit1 = np.pi


low_semi_major = 1.493e11 # where LISA Constants is 149597870700.0
high_semi_major = 1.496e11 #Estimate from Fig 6 Trajectory Design Paper (for 10 years; way conservative)

low_omega = 0.0
high_omega = 2.0*np.pi

low_arg_per = 0.0
high_arg_per = np.pi

low_eccentricity = 0.004 # see fig 6 trajectory design paper
high_eccentricity = 0.0055

low_inclination = 0.39*np.pi/180.0# see fig 6 trajectory design paper
high_inclination = 0.6*np.pi/180.0




#data_dir = '/Users/jessica/Desktop/Project_2/Tdi_2.0/Production/testing_codes_not_production/'

if static==True:
    data=np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/tryer_faster_codes/LISA_Instrument_RR_disable_all_but_laser_lock_six_static_orbits_tps_ppr_orbits_pyTDI_size_mprs_to_file.dat',names=True)

elif equalarmlength==True:
    data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_equalarmlength_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
elif keplerian==True:
    if is_tcb==True:
        data = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_orbits_tcb_ltt_orbits_mprs_and_dpprs_to_file_1_hour_NO_AA_filter_NEW.dat',names=True)
    else:
        data = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_ESA_orbits_tcb_ltt_orbits_mprs_and_dpprs_to_file_1_hour_4_Hz_NO_AA_filter_NEW.dat',names=True)

elif esa==True:
    data = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_ESA_orbits_tcb_ltt_orbits_mprs_and_dpprs_to_file_1_hour_4_Hz_NO_AA_filter_NEW.dat',names=True)

#data =  np.genfromtx
initial_length = len(data['s31'])

cut_off = 0
#cut_off=int(1e3)
end_cut_off = initial_length+1





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
tcb_times = np.array([times_one,times_two,times_three])
#semi_major_0=np.array([elements_data[0]+elements_data[18]*times,elements_data[1]+elements_data[19]*times,elements_data[2]+elements_data[20]*times])


#times = np.genfromtxt('tau.dat')[cut_off::]
print('times')
print(times)
print('times one')
print(times_one)
#times=times[:initial_length-cut_off:]
#times=times[cut_off::]
print('times[0]')
print(times[0])

#t_init = 0.0
#t_init = 2019686400.0
#t_init = 2019704400.0
#t_init = times[0]
#t_init = 2073211130.8175
#t_init = 2167920000.0
#t_init = 2146928400.0
#t_init = 2083770000.0
t_init =13100.0


'''
print('tinit')
print(t_init)
print('tinit')
print(times[0])
#t_init = times[0]

print('tcb_times')
print((tcb_times[0]-times)*-1)
print('from file')
print(data['time_one'][cut_off::])
#sys.exit()
'''
#alpha_ = 2*np.pi/period*times 

s31 = data['s31'][cut_off::]/central_freq
s21 = data['s21'][cut_off::]/central_freq
s32 = data['s32'][cut_off::]/central_freq
s12 = data['s12'][cut_off::]/central_freq
s23 = data['s23'][cut_off::]/central_freq
s13 = data['s13'][cut_off::]/central_freq

print('s31 flags')
print(s31.flags)


tau31 = data['tau31'][cut_off::]/central_freq
tau21 = data['tau21'][cut_off::]/central_freq
tau12 = data['tau12'][cut_off::]/central_freq
tau32 = data['tau32'][cut_off::]/central_freq
tau23 = data['tau23'][cut_off::]/central_freq
tau13 = data['tau13'][cut_off::]/central_freq

eps31 = data['eps31'][cut_off::]/central_freq
eps21 = data['eps21'][cut_off::]/central_freq
eps12 = data['eps12'][cut_off::]/central_freq
eps32 = data['eps32'][cut_off::]/central_freq
eps23 = data['eps23'][cut_off::]/central_freq
eps13 = data['eps13'][cut_off::]/central_freq

ppr_L_1 = data['mprs_32'][cut_off::]
ppr_L_1_p = data['mprs_23'][cut_off::]
ppr_L_2 = data['mprs_13'][cut_off::]
ppr_L_2_p = data['mprs_31'][cut_off::]
ppr_L_3 = data['mprs_21'][cut_off::]
ppr_L_3_p = data['mprs_12'][cut_off::]

length = len(s31)
print('length')
print(length)

#constant array for calculating delay polynomials
ints = np.broadcast_to(np.arange(number_n),(length,number_n)).T
#print('ints')
#print(ints)
#ints = np.arange(number_n)

#Coefficients in Lagrange Time-varying Filter (Pre-Processing)
s32_coeffs = difference_operator_powers(s32)
s31_coeffs = difference_operator_powers(s31)
s12_coeffs = difference_operator_powers(s12)
s13_coeffs = difference_operator_powers(s13)
s21_coeffs = difference_operator_powers(s21)
s23_coeffs = difference_operator_powers(s23)

eps32_coeffs = difference_operator_powers(eps32)
eps31_coeffs = difference_operator_powers(eps31)
eps12_coeffs = difference_operator_powers(eps12)
eps13_coeffs = difference_operator_powers(eps13)
eps21_coeffs = difference_operator_powers(eps21)
eps23_coeffs = difference_operator_powers(eps23)

tau32_coeffs = difference_operator_powers(tau32)
tau31_coeffs = difference_operator_powers(tau31)
tau12_coeffs = difference_operator_powers(tau12)
tau13_coeffs = difference_operator_powers(tau13)
tau21_coeffs = difference_operator_powers(tau21)
tau23_coeffs = difference_operator_powers(tau23)



del data
#del f


'''   
positions_1 = np.genfromtxt('s_c_positions_1.dat')[cut_off::]    

positions_2 = np.genfromtxt('s_c_positions_2.dat')[cut_off::]       
positions_3 = np.genfromtxt('s_c_positions_3.dat')[cut_off::]    

velocity_1 = np.genfromtxt('s_c_velocity_1.dat')[cut_off::]    
velocity_2 = np.genfromtxt('s_c_velocity_2.dat')[cut_off::]    
velocity_3 = np.genfromtxt('s_c_velocity_3.dat')[cut_off::]    



#sys.exit()
print('positions_1.T')    
print(positions_1.T)
print('positions_2.T')    
print(positions_2.T)
#sys.exit()

print('velocity_1')    
print(velocity_1.T)

print('velocity_1 grad')    
print(np.gradient(positions_1.T,1/f_s,axis=1))

one_x = positions_1.T[0]
one_y = positions_1.T[1]
one_z = positions_1.T[2]

two_x = positions_2.T[0]
two_y = positions_2.T[1]
two_z = positions_2.T[2]

three_x = positions_3.T[0]
three_y = positions_3.T[1]
three_z = positions_3.T[2]
'''

# # Windowing and Noise Covariance Matrix Parameters

# In[26]:



cut_data_length = len(s31)
#beg_ind,end_ind = cut_data(L_3,L_2,L_1,L_3_p,L_2_p,L_1_p,f_s,length)
#cut_data_length = len(s31[beg_ind::])
#window = tukey(cut_data_length,0.4)

window = kaiser(cut_data_length,kaiser_beta(320))

f_band = np.fft.rfftfreq(cut_data_length,1/f_s)
indices_f_band = np.where(np.logical_and(f_band>=f_min, f_band<=f_max))
f_band=f_band[indices_f_band]

Sy_PM = S_y_proof_mass_new_frac_freq(f_band)
Sy_OP = S_y_OMS_frac_freq(f_band)
a,b_ = covariance_equal_arm(f_band,Sy_OP,Sy_PM)
#Needed in inverse calculation
A_ = a**2 - b_**2
B_ = b_**2 - a*b_

log_term_factor = 3*np.log(np.pi)
determinant = a*A_+2*b_*B_
log_term_determinant = np.log(determinant)






#........................................................................................
#...........................MCMC Portion.......................................
#........................................................................................



elements_data = np.genfromtxt('elements_from_Cartesian_hour.dat')


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
test_filter = filters_lagrange_2_0(nested[0])
data_to_use_s13 = np.concatenate((np.zeros((test_filter[1],number_n+1)),s13_coeffs),axis=0)
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
		sampler = zeus.EnsembleSampler(Nens, ndims, target_log_prob_fn,mu=1e3,pool=pool)
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

