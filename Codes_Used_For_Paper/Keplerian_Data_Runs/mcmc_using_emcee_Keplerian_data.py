#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
#from mpmath import binomial,log,pi,ceil
import numpy as np
#import matplotlib.pyplot as plt
import math
from scipy.signal import kaiser,kaiser_beta
#from mpmath import *
import time
from scipy.stats import norm
#from astropy import constants as const
import numpy as np
#from scipy.signal import butter,filtfilt
#from scipy.special import gamma,binom,comb
#import h5py   
#import nexusformat.nexus as nx
import matplotlib.pyplot as plt
import emcee
from lisaconstants import GM_SUN,c,ASTRONOMICAL_YEAR,ASTRONOMICAL_UNIT
#import mpld3
#mpld3.enable_notebook()

#from sincinterpol import interp
#from shift_roll import roll_zeropad
#import scipy.sparse as sp
#import bottleneck as bn
#from numba import jit
#import timeit


# # CREATING FILTERS

# In[2]:


#@jit(cache=True)
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


# In[ ]:





# In[3]:


def filters_lagrange_2_0(D):

    #start_time = time.process_time()
    D=D*f_samp
    integer_part, d_frac = np.divmod(D,1)

    integer_part = integer_part-p
    d_frac = d_frac+p

    delay_polynomials = np.ones((number_n+1,length))

    #factors = np.array([-1*d_frac+i for i in ints])
    factors = -1*d_frac+ints

    delay_polynomials[1:number_n+1] = np.cumprod(factors,axis=0)
    #print("--- %s seconds for filters lagrange---" % (time.process_time() - start_time))

    return delay_polynomials,int(integer_part[0])

# In[4]:


#@jit(cache=True)
def trim_data(data,filter_array):


    #data=np.roll(data,filter_array[1],axis=0)
    #data[:filter_array[1]] = 0.0
    #val = np.einsum('ij,ji->i',data,filter_array[0],optimize=einsum_path_to_use[0])
    #val =  np.einsum('ij,ji->i',data,filter_array[0],optimize=einsum_path_to_use)
    #val = np.einsum('ij,ji->i',np.concatenate((np.zeros((filter_array[1],number_n+1)),data),axis=0)[:-filter_array[1]:],filter_array[0],optimize=einsum_path_to_use)

    #return val
    return np.einsum('ij,ji->i',np.concatenate((np.zeros((filter_array[1],number_n+1)),data),axis=0)[:-filter_array[1]:],filter_array[0],optimize=einsum_path_to_use)

    #print("--- %s seconds for trim_data---" % (time.process_time() - start_time))




def S_y_proof_mass_new_frac_freq(f):

    pm_here =  np.power(2.4e-15,2)*(1+np.power(4.0e-4/f,2))*(1+np.power(f/8.0e-3,4))
    return pm_here*np.power(2*np.pi*f*c,-2)


# In[6]:

def S_y_OMS_frac_freq(f):

    op_here =  np.power(1.5e-11,2)*np.power(2*np.pi*f/c,2)*(1+np.power(2.0e-3/f,4))
    return op_here

# In[7]:


#@jit(numba.types.float64(numba.types.float64),nopython=True)
#@jit(nopython=True)

def theta(k):
    return 2.0*np.pi*(k-1)/3


# In[13]:


#@jit(nopython=True)
#@jit(numba.types.float64[:](numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64[:]),nopython=True)

def psi(m_init1,eccentricity,orbital_freq,k,t):
    m = m_init1 + orbital_freq*(t-t_init) - theta(k)
    psi_return = m + (eccentricity-np.power(eccentricity,3)/8)*np.sin(m) + 0.5*eccentricity**2*np.sin(2*m)        + 3.0/8 *np.power(eccentricity,3)*np.sin(3*m)
    for i in np.arange(2):
        error =psi_return - eccentricity * np.sin(psi_return) - m
        psi_return -= error / (1 - eccentricity * np.cos(psi_return)) 

    return psi_return
    #psi_one = psi_zero - (psi_zero - eccentricity*np.sin(psi_zero) - m)/(1 - eccentricity*np.cos(psi_zero))
    #return psi_one
    #return psi_zero
    #return psi_one - (psi_one - eccentricity*np.sin(psi_one) - m)/(1 - eccentricity*np.cos(psi_one))


# In[14]:


#@jit(nopython=True)
def orbital_parameters(semi_major):
    alpha = L_arm/(2.0*semi_major)
    nu = np.pi/3 + delta*alpha
    eccentricity = np.sqrt(1 + 4.0*alpha*np.cos(nu)/np.sqrt(3) + 4.0*alpha**2/3) - 1

    orbital_freq=np.sqrt(GM_SUN/semi_major**3)

    tan_inclination = alpha*np.sin(nu)/(np.sqrt(3.0)/2.0 + alpha*np.cos(nu))
    cos_inclination = 1 / np.sqrt(1 + tan_inclination**2)
    sin_inclination = tan_inclination*cos_inclination

    return alpha,nu,eccentricity,orbital_freq,cos_inclination,sin_inclination


# In[15]:


#@jit(numba.types.float64[:,:](numba.types.float64[:],numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64),nopython=True)
#@jit(nopython=True,parallel=True)
def s_c_positions(psi_here,eccentricity,cos_inclination,sin_inclination,semi_major,Omega_1,arg_per,k):
	#lambda_k = Omega_1 + theta(k) + arg_per
	omega = Omega_1+theta(k)
	#print('k')
	#print(eccentricity)
	#lambda_k = omega  + arg_per
	zeta_t = semi_major*(np.cos(psi_here) - eccentricity)
	eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.sin(psi_here)
	#psi_here = psi(m_init1,eccentricity,orbital_freq,k,t)
	positions = np.empty((3,length))

	positions[0] = (np.cos(omega)*np.cos(arg_per) - np.sin(omega)*np.sin(arg_per)*cos_inclination)*zeta_t - (np.cos(omega)*np.sin(arg_per) + np.sin(omega)*np.cos(arg_per)*cos_inclination)*eta_t #x(t)
	positions[1] = (np.sin(omega)*np.cos(arg_per) + np.cos(omega)*np.sin(arg_per)*cos_inclination)*zeta_t - (np.sin(omega)*np.sin(arg_per) - np.cos(omega)*np.cos(arg_per)*cos_inclination)*eta_t #y(t)
	positions[2] = np.sin(arg_per)*sin_inclination*zeta_t + np.cos(arg_per)*sin_inclination*eta_t #z(t)
	'''
	xref = semi_major*cos_inclination*(np.cos(psi_here)-eccentricity)
	yref = semi_major*np.sqrt(1.0-eccentricity**2)*np.sin(psi_here )
	zref = -1*semi_major*sin_inclination*(np.cos(psi_here)-eccentricity)

	x_t = np.cos(lambda_k)*xref - np.sin(lambda_k)*yref
	y_t = np.sin(lambda_k)*xref + np.cos(lambda_k)*yref

	positions[0] = x_t
	positions[1] = y_t
	positions[2] = zref
	'''
	return positions

	#return np.array([x_t,y_t,zref])


# In[16]:


#@jit(nopython=True,parallel=True)
#@jit(numba.types.float64[:,:](numba.types.float64[:],numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64),nopython=True)

def s_c_velocities(psi_here,eccentricity,cos_inclination,sin_inclination,semi_major,orbital_freq,Omega_1,arg_per,k):
	#psi_here = psi(m_init1,eccentricity,orbital_freq,k,t)

	psi_dot = orbital_freq/(1.0-eccentricity*np.cos(psi_here))
	#psi_dot = np.gradient(psi_here,1/f_s,edge_order=2)

	#print('k')
	#print(eccentricity)
	#lambda_k = omega  + arg_per
	omega = Omega_1+theta(k)
	zeta_t = semi_major*(np.cos(psi_here) - eccentricity)
	d_zeta_t = -1*semi_major*np.sin(psi_here)*psi_dot
	eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.sin(psi_here)
	d_eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.cos(psi_here)*psi_dot
	#psi_here = psi(m_init1,eccentricity,orbital_freq,k,t)
	velocities = np.empty((3,length))

	velocities[0] = (np.cos(omega)*np.cos(arg_per) - np.sin(omega)*np.sin(arg_per)*cos_inclination)*d_zeta_t - (np.cos(omega)*np.sin(arg_per) + np.sin(omega)*np.cos(arg_per)*cos_inclination)*d_eta_t #x(t)
	velocities[1] = (np.sin(omega)*np.cos(arg_per) + np.cos(omega)*np.sin(arg_per)*cos_inclination)*d_zeta_t - (np.sin(omega)*np.sin(arg_per) - np.cos(omega)*np.cos(arg_per)*cos_inclination)*d_eta_t #y(t)
	velocities[2] = np.sin(arg_per)*sin_inclination*d_zeta_t + np.cos(arg_per)*sin_inclination*d_eta_t #z(t)
	'''
	vxref = -1.0*semi_major*psi_dot*cos_inclination*np.sin(psi_here)
	vyref = semi_major*psi_dot*np.sqrt(1.0-eccentricity**2)*np.cos(psi_here)
	vzref = semi_major*psi_dot*sin_inclination*np.sin(psi_here)

	#lambda_k = omega_1 + theta(k) + arg_per
	lambda_k = omega_1 + arg_per


	vx = np.cos(lambda_k)*vxref - np.sin(lambda_k)*vyref
	vy = np.sin(lambda_k)*vxref + np.cos(lambda_k)*vyref
	velocities[0] = vx
	velocities[1] = vy
	velocities[2] = vzref
	'''
	return velocities


# In[17]:


#@jit(nopython=True,parallel=True)
#@jit(numba.types.float64[:](numba.types.float64[:],numba.types.float64,numba.types.float64),nopython=True)

def s_c_accelerations(position_here,semi_major,orbital_freq):
    #position_here = s_c_positions(m_init1,eccentricity,cos_inclination,sin_inclination,semi_major,orbital_freq,k,t)
    
    #return -1*np.power(semi_major,3)*orbital_freq**2*position_here/np.power(np.linalg.norm(position_here,axis=0),3)
    return -1*np.power(semi_major,3)*orbital_freq**2*position_here/np.power(np.sqrt(position_here[0]**2+position_here[1]**2+position_here[2]**2),3)


# In[18]:


#@jit(nopython=True,parallel=True,cache=True)
#@jit(numba.types.float64[:](numba.types.float64[:],numba.types.float64[:]),nopython=True)

def shapiro(pos_i,pos_j):
    #return 2*GM_SUN/(c**2)*np.log((np.linalg.norm(pos_j,axis=0)+np.linalg.norm(pos_i,axis=0)+np.linalg.norm(pos_j-pos_i,axis=0))\
                                                         #/(np.linalg.norm(pos_j,axis=0)+np.linalg.norm(pos_i,axis=0)-np.linalg.norm(pos_j-pos_i,axis=0)))

    mag_pos_j = np.sqrt(pos_j[0]**2+pos_j[1]**2+pos_j[2]**2)     
    mag_pos_i = np.sqrt(pos_i[0]**2+pos_i[1]**2+pos_i[2]**2)    
    diff = pos_j-pos_i
    mag_diff = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

    return 2*GM_SUN/(c**2)*np.log((mag_pos_j + mag_pos_i + mag_diff)/(mag_pos_j+mag_pos_i-mag_diff))


# In[19]:


#@jit(nopython=True,parallel=True,cache=True)
#@jit(numba.types.float64(numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64[:]),nopython=True)

def delta_tau(psi_here,m_init1,eccentricity,orbital_freq,semi_major,k,t):
    #psi_here = psi(m_init1,eccentricity,orbital_freq,k,t)
    psi_here_init = psi(m_init1,eccentricity,orbital_freq,k,t_init)

    return -3.0/2.0*(orbital_freq*semi_major/c)**2*(t-t_init) - 2*(orbital_freq*semi_major/c)**2*eccentricity/orbital_freq*(np.sin(psi_here)-np.sin(psi_here_init))


# In[20]:


#@jit(nopython=True,parallel=True)
#@numba.cfunc("double(double)")
def time_dependence(m_init1,semi_major,arg_per):


    delay_in_time = np.empty((6,length))
    if is_tcb==False:
        mprs = np.empty((6,length))

    alpha,nu,eccentricity,orbital_freq,cos_inclination,sin_inclination = orbital_parameters(semi_major)
    for i,k,z in zip(np.arange(6),np.array([2,3,3,1,1,2]),np.array([3,2,1,3,2,1])):
        psi_i = psi(m_init1,eccentricity,orbital_freq,z,tcb_times[z-1])
        psi_j = psi(m_init1,eccentricity,orbital_freq,k,tcb_times[z-1])
        position_i = s_c_positions(psi_i,eccentricity,cos_inclination,sin_inclination,semi_major,Omega_1,arg_per,z)
        position_j = s_c_positions(psi_j,eccentricity,cos_inclination,sin_inclination,semi_major,Omega_1,arg_per,k)
        Dij = position_i-position_j


        magDij = np.sqrt(Dij[0]**2+Dij[1]**2+Dij[2]**2)

        velocity_j = s_c_velocities(psi_j,eccentricity,cos_inclination,sin_inclination,semi_major,orbital_freq,Omega_1,arg_per,k)
        second_term = np.sum(Dij*velocity_j,axis=0)/(c**2)

        #third_term = magDij/(2*np.power(c,3))*(np.linalg.norm(velocity_j,axis=0)**2 + np.power(np.sum(velocity_j*Dij,axis=0)/magDij,2) -np.sum(s_c_accelerations(position_j,semi_major,orbital_freq)*Dij,axis=0))
        third_term = magDij/(2*np.power(c,3))*(np.sqrt(velocity_j[0]**2+velocity_j[1]**2+velocity_j[2]**2)**2 + np.power(np.sum(velocity_j*Dij,axis=0)/magDij,2) -np.sum(s_c_accelerations(position_j,semi_major,orbital_freq)*Dij,axis=0))

        delay_in_time[i] = magDij/c + second_term + third_term +  shapiro(position_i,position_j)/c
        if is_tcb==False:
            mprs[i] = delay_in_time[i]+ delta_tau(psi_i,m_init1,eccentricity,orbital_freq,semi_major,z,tcb_times[z-1]) - delta_tau(psi_j,m_init1,eccentricity,orbital_freq,semi_major,k,tcb_times[z-1] - delay_in_time[i])


    if is_tcb==True:
        return delay_in_time
    else:
        return mprs





#Eq. (**) on page 1100 of notes
#@jit(cache=True)
def nested_delay_application(delay_array_here,list_delays):
    number_delays = len(list_delays)
    
    delays = np.array([delay_array_here[j] for j in list_delays])


    delay_dot_array_here = np.gradient(delays,1/f_s,axis=1)

    correction_factor =np.zeros(length)

    for i in np.arange(number_delays):
        for j in np.arange(i+1,number_delays):

            correction_factor+=delays[i]*delay_dot_array_here[j]          


    doppler_factor = np.sum(delay_dot_array_here,axis=0)

    commutative_sum = np.sum(delays,axis=0)


    #print("--- %s seconds for linear delay application---" % (time.process_time() - start_time))

    return commutative_sum, np.gradient(commutative_sum,1/f_s), correction_factor
    #return commutative_sum, 0.0, correction_factor

    #return commutative_sum, doppler_factor, correction_factor




# In[23]:


#@jit(nopython=True,parallel=True,fastmath=True,cache=True)
def x_combo_2_0(delay_array):


    #delay_array = np.zeros((6,length))
    #four_delays_needed = time_dependence([0,[0,['2','orbital','x')
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


# In[24]:


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



# In[19]:


def covariance_equal_arm(f,Sy_OP,Sy_PM):
    
    a = 16*np.power(np.sin(2*np.pi*f*avg_L),2)*Sy_OP+(8*np.power(np.sin(4*np.pi*f*avg_L),2)+32*np.power(np.sin(2*np.pi*f*avg_L),2))*Sy_PM
    b_ = -4*np.sin(2*np.pi*f*avg_L)*np.sin(4*np.pi*f*avg_L)*(4*Sy_PM+Sy_OP)

    return 2*a,2*b_


# In[20]:


#@jit(nopython=True,parallel=True,cache=True)
def likelihood_analytical_equal_arm(x,y,z):



	chi_2 = 1/determinant*(A_*(x[0]**2+x[1]**2+y[0]**2+y[1]**2+z[0]**2+z[1]**2) + 2*B_*(x[0]*y[0]+x[1]*y[1]+x[0]*z[0]+x[1]*z[1]+y[0]*z[0]+y[1]*z[1]))


	value = -1*np.sum(chi_2) - log_term_factor - np.sum(log_term_determinant)

	return value,np.sum(chi_2)	    
 
'''   
#........................................................................................
#...........................MCMC Functions.......................................
#........................................................................................
def proposal_minit1(mean, draw):

	
	if draw == 0:

		new_val = np.random.normal(mean,np.pi/2.0)
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val

	
	elif draw == 1:

		new_val = np.random.normal(mean,1.0e-6)
		return new_val
		
	elif draw == 2:

		new_val = np.random.normal(mean,1.0e-9)
		return new_val
		
	elif draw == 3:

		new_val = np.random.normal(mean,1.0e-3)
		return new_val

def proposal_minit1(mean, draw):

	if draw == 0:

		new_val = np.random.multivariate_normal(mean,(np.pi/2.0)**2*np.array([[1,0],[0,1]]))
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val
		
	elif draw == 1:

		new_val = np.random.multivariate_normal(mean,(1.0e-3)**2*np.array([[1,0.99],[0.99,1]]))
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val
		
	elif draw == 2:

		new_val = np.random.multivariate_normal(mean,(1.0e-6)**2*np.array([[1,0.99],[0.99,1]]))
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val
	

	

	
def proposal_semi_major(mean,draw):

	if draw == 0:

		new_val = np.random.normal(mean,1.0e5)
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val
	
	elif draw == 1:

		new_val = np.random.normal(mean,1.0e8)
		return new_val
	



def proposal_Omega_1(mean, draw):

	if draw == 0:

		new_val = np.random.normal(mean,np.pi/2.0)
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val
	
	elif draw == 1:

		new_val = np.random.normal(mean,1.0e-3)
		return new_val
	
	
	elif draw == 2:

		new_val = np.random.normal(mean,1.0e-6)
		return new_val
	
	elif draw == 3:

		new_val = np.random.normal(mean,1.0e-9)
		return new_val
		
def proposal_arg_per(mean, draw):

	if draw == 0:

		new_val = np.random.normal(mean,np.pi/2.0)
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val
	
	
	elif draw == 1:

		new_val = np.random.normal(mean,1.0e-5)
		return new_val
		
	elif draw == 2:

		new_val = np.random.normal(mean,1.0e-8)
		return new_val
		
'''
		
def prior_minit1(val_minit):
	#val_ar = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (high-low)*uniform.pdf(val[0],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[1],low,high-low) ==1.0 and (high-low)*uniform.pdf(val[2],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[3],low,high-low) ==1.0:
	if (val_minit >= low_minit1) and (val_minit <= high_minit1):
		return 1
	else:
		return 0

def prior_semi_major(val):
	val = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (high-low)*uniform.pdf(val[0],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[1],low,high-low) ==1.0 and (high-low)*uniform.pdf(val[2],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[3],low,high-low) ==1.0:
	if (val >= low_semi_major) and (val <= high_semi_major):
		return 1
	else:
		return 0


		
def prior_Omega_1(val):
	val = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (high-low)*uniform.pdf(val[0],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[1],low,high-low) ==1.0 and (high-low)*uniform.pdf(val[2],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[3],low,high-low) ==1.0:
	if (val >= low_Omega_1) and (val <= high_Omega_1):
		return 1
	else:
		return 0


	
def prior_arg_per(val):
	val = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (high-low)*uniform.pdf(val[0],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[1],low,high-low) ==1.0 and (high-low)*uniform.pdf(val[2],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[3],low,high-low) ==1.0:
	if (val >= low_arg_per) and (val <= high_arg_per):
		return 1
	else:
		return 0		
	
	
	
def target_log_prob_fn(state_current):


	delays_in_time =time_dependence(state_current[0],state_current[1],state_current[2])


	x_combo = x_combo_2_0(delays_in_time)
	y_combo = y_combo_2_0(delays_in_time)
	z_combo = z_combo_2_0(delays_in_time)

	likelihood,chi_2_here = likelihood_analytical_equal_arm(x_combo,y_combo,z_combo)
	#print('likelihood in target log prob')
	#print(likelihood)
	prior = prior_minit1(state_current[0])*prior_semi_major(state_current[1])*prior_arg_per(state_current[2])

	#return model.log_prob(rate=rate, obs=poisson_samples) 
	return likelihood + np.log(prior)	
		  
	
		
f_s = 2.0
#f_s = 4.0
f_samp = f_s
#lagrange filter length
number_n_data = 7
#number_n_data = 15
number_n = number_n_data
p = number_n//2
asd_nu = 28.8 # Hz/rtHz



f_min = 5.0e-4 # (= 0.0009765625)
f_max = 0.03
central_freq=281600000000000.0
L_arm = 2.5e9
avg_L = L_arm/c

static=False
equalarmlength=False
keplerian=True


matrix=True

is_tcb = False




# In[3]:


#lambda_init_0 = 0.0
#Omega1_init_0 = np.pi/2.0
Omega_1 = np.pi/2.0


arg_per_0 = -np.pi/2.0
semi_major_0=ASTRONOMICAL_UNIT
m_init1_0 = 0.0

delta = 5.0/8.0 #value of delta that minimizes constellation breathing. See LISA Orbits docs






data_dir = '/Users/jessica/Desktop/Project_2/Tdi_2.0/Production/orbit_and_data_files/'

if static==True:
    #data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_static_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
    #data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_equalarmlength_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
    data=np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/tryer_faster_codes/LISA_Instrument_RR_disable_all_but_laser_lock_six_static_orbits_tps_ppr_orbits_pyTDI_size_mprs_to_file.dat',names=True)

elif equalarmlength==True:
    data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_equalarmlength_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
elif keplerian==True:
    if is_tcb==True:
        data = np.genfromtxt(data_dir+'LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_orbits_tcb_ltt_orbits_mprs_and_dpprs_to_file_1_hour_NO_AA_filter_NEW.dat',names=True)
        #tcb_times = np.genfromtxt('tcb_sc_times.dat',names=True)
        #timer_deviations = np.genfromtxt('tcb_timer_deviations.dat',names=True)
    else:
        #data = np.genfromtxt(data_dir+'LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_1_hour_NO_AA_filter_NEW.dat',names=True)
        #data = np.genfromtxt(data_dir+'LISA_Instrument_RR_disable_all_but_laser_lock_six_Keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_1_days_3_Hz_NO_AA_filter_NEW.dat',names=True)
        data = np.genfromtxt(data_dir+'LISA_Instrument_RR_disable_all_but_laser_lock_six_Keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_1_days_2_Hz_NO_AA_filter_NEW.dat',names=True)
#data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_tps_ppr_orbits_pyTDI_size.dat',names=True)
#data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_equalarmlength_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
#data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_static_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
initial_length = len(data['s31'])

#cut_off = 0
cut_off=int(1e4)
end_cut_off = initial_length+1




'''
#times = np.arange(initial_length)/f_s
times = data['time'][cut_off::]
times_one = data['time_one']
times_two = data['time_two']
times_three = data['time_three']
'''



#for LISA Orbits 2.1 
times = data['time'][cut_off::]
times_one = times - data['time_one'][cut_off::]
times_two = times - data['time_two'][cut_off::]
times_three = times - data['time_three'][cut_off::]

tcb_times = np.array([times_one,times_two,times_three])


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
t_init = 2019704400.0


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



'''
#for number chain iterations
#number_chain = 50000
#number_chain = 1000000
number_chain = 100000
#number_chain  = 5000
checkfile = open('chainfile_testing_LISA_Instrument_Data_disable_all_but_laser_lock_six_static_tps_ppr_TDI_2_0_Keplerian_data_1_day_Keplerian_orbital_parameters_minit_semi_major_omega_Omega1_2Hz_N=7_fmin=5e-4_fmax=0_03_partly_blocked_proposals.dat','w')
checkfile.write("#likelihood" + " " + "chi_2_here" + " " + "minit1" + " " + "semi_major" + " " + "Omega_1" + " " + "arg_per" + " " + "Current_AR" + "\n")
k=1
'''
#MCMC Parameters
low_minit1 = -np.pi/2.0
high_minit1 = np.pi/2.0


low_semi_major = 1.494e11 # where LISA Constants is 149597870700.0
high_semi_major = 1.497e11 #Estimate from Fig 6 Trajectory Design Paper (for 10 years; way conservative)#Estimate from Fig 6 Trajectory Design Paper (for 10 years; way conservative)

low_Omega_1 = -2.0*np.pi
high_Omega_1 = 2.0*np.pi

low_arg_per = -np.pi
high_arg_per = 0.0

#Get optimal einsum path]
orbital_L_3_p = time_dependence(m_init1_0,semi_major_0,arg_per_0)
nested = nested_delay_application(orbital_L_3_p,np.array([0,1]))
test_filter = filters_lagrange_2_0(nested[0])
data_to_use_s13 = np.concatenate((np.zeros((test_filter[1],number_n+1)),s13_coeffs),axis=0)
data_here_s13 = data_to_use_s13[:-test_filter[1]:]   
einsum_path_to_use = np.einsum_path('ij,ji->i',data_here_s13,test_filter[0], optimize='True')[0]
print('einsum_path_to_use')
print(einsum_path_to_use)
# # Calculate Likelihood

# In[27]:

'''
initial_L_1 = L_1
initial_L_1_p = L_1_p
initial_L_2 = L_2
initial_L_3 = L_3
initial_L_2_p = L_2_p
initial_L_3_p = L_3_p

initial_L_1_dot = L_1_dot
initial_L_1_p_dot = L_1_p_dot
initial_L_2_dot = L_2_dot
initial_L_3_dot = L_3_dot
initial_L_2_p_dot = L_2_p_dot
initial_L_3_p_dot = L_3_p_dot
'''
#time mcmc computation time
start_time = time.time()

#initial delays accepted into the chain
accept = 1
initial_delays_in_time = time_dependence(m_init1_0,semi_major_0,arg_per_0)

#initial_state_truth = np.array([elements_data[0],elements_data[1],elements_data[2],elements_data[3],elements_data[4],elements_data[5],elements_data[6],elements_data[7],elements_data[8],elements_data[9],elements_data[10],elements_data[11],elements_data[12],elements_data[13],elements_data[14],elements_data[15],elements_data[16],elements_data[17]])
initial_state_truth = np.array([m_init1_0,semi_major_0,arg_per_0])
print('initial_state_truth')
print(initial_state_truth)
#ndims = len(initial_state_truth) # number of parameters/dimensions

Nens = 10  # number of ensemble points
Nburnin = 100   # number of burn-in samples
Nsamples = 10000  # number of final posterior samples
'''
initial_state = np.array([np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),np.random.uniform(low_eccentricity,high_eccentricity,size=Nens),\
				np.random.uniform(low_inclination,high_inclination,size=Nens),np.random.uniform(low_inclination,high_inclination,size=Nens),np.random.uniform(low_inclination,high_inclination,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens), np.random.uniform(low_minit1,high_minit1,size=Nens),np.random.uniform(low_omega,high_omega,size=Nens),\
				np.random.uniform(low_omega,high_omega,size=Nens),np.random.uniform(low_omega,high_omega,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens)])
'''

#initial_state = np.array([np.random.uniform(low_minit1,high_minit1,size=Nens),np.random.uniform(low_semi_major,high_semi_major,size=Nens),np.random.uniform(low_arg_per,high_arg_per,size=Nens)])
initial_state = np.array([np.random.uniform(m_init1_0-1.0e-5,m_init1_0+1.0e-5,size=Nens),np.random.uniform(1.49597e+11,1.49598e+11,size=Nens),np.random.uniform(arg_per_0-1.0e-6,arg_per_0+1.0e-6,size=Nens)])
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

ndims = initial_state.shape[1]
print('ndims')
print(ndims)

x_combo_initial_2_0 = x_combo_2_0(initial_delays_in_time)
y_combo_initial_2_0 = y_combo_2_0(initial_delays_in_time)
z_combo_initial_2_0 = z_combo_2_0(initial_delays_in_time)

filename = "samples_Keplerian_chain_omega_emcee_backend_testing_small_ball.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(Nens, ndims)
sampler = emcee.EnsembleSampler(Nens, ndims, target_log_prob_fn,backend=backend)

f = open("samples_Keplerian_chain_omega_emcee_testing_small_ball.dat", "w")
f.close()

max_n = Nsamples

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

n = 100 * np.arange(1, index_here + 1)
y = autocorr[:index_here]
plt.plot(n, n / 100.0, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$")
plt.show()

'''
#y_combo_initial_2_0 = y_combo_2_0([initial_L_1,initial_L_1_p,initial_L_2,initial_L_2_p,initial_L_3,initial_L_3_p],[initial_L_1_dot,initial_L_1_p_dot,initial_L_2_dot,initial_L_2_p_dot,initial_L_3_dot,initial_L_3_p_dot])
#y_combo_initial_2_0 = y_combo_2_0([initial_L_1,initial_L_1_p,initial_L_2,initial_L_2_p,initial_L_3,initial_L_3_p],[initial_L_1_dot,initial_L_1_p_dot,initial_L_2_dot,initial_L_2_p_dot,initial_L_3_dot,initial_L_3_p_dot])
#z_combo_initial_2_0 = z_combo_2_0([initial_L_1,initial_L_1_p,initial_L_2,initial_L_2_p,initial_L_3,initial_L_3_p],[initial_L_1_dot,initial_L_1_p_dot,initial_L_2_dot,initial_L_2_p_dot,initial_L_3_dot,initial_L_3_p_dot])
#old_likelihood,determ_here,chi_2_here = likelihood_analytical_equal_arm(x_combo_initial_2_0,y_combo_initial_2_0,z_combo_initial_2_0)
old_likelihood,chi_2_here = likelihood_analytical_equal_arm(x_combo_initial_2_0,y_combo_initial_2_0,z_combo_initial_2_0)

print('likelihood')
print(old_likelihood)
print('chi 2')
print(chi_2_here)
print('likelihood')
print(old_likelihood)
#sys.exit()
checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(m_init1_0) + " " + str(semi_major_0)  + " " + str(Omega1_init_0)  + " " + str(arg_per_0)  + " " + str(accept/k) + "\n")

old_minit1 = m_init1_0
old_semi_major = semi_major_0
old_Omega_1 = Omega1_init_0
old_arg_per = arg_per_0

x_combo_old = x_combo_initial_2_0
y_combo_old = y_combo_initial_2_0
z_combo_old = z_combo_initial_2_0

draw_count_semi_major = 0
draw_count_minit=0
draw_count_delta=0
draw_count_Omega_1 = 0
draw_count_arg_per = 0

#rest of mcmc chain
#for i, j, k in zip(L_1_chain,L_2_chain,counter):
while k <= number_chain:



	print('chain number')
	print(k)

	#...........................minit........................................
	#...........................minit........................................
	#...........................minit........................................	
	minit1_draw,arg_per_draw = proposal_minit1(np.array([old_minit1,old_arg_per]), draw_count_minit)
	delays_in_time_new = time_dependence(minit1_draw,old_semi_major,old_Omega_1,arg_per_draw)
	x_combo_new = x_combo_2_0(delays_in_time_new)
	y_combo_new = y_combo_2_0(delays_in_time_new)
	z_combo_new = z_combo_2_0(delays_in_time_new)
	new_likelihood,new_chi_2_here  = likelihood_analytical_equal_arm(x_combo_new,y_combo_new,z_combo_new)
	
	alpha = min(np.log(prior_minit1(minit1_draw,arg_per_draw))+new_likelihood-np.log(prior_minit1(old_minit1,old_arg_per))-old_likelihood,0)

	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_minit1 =minit1_draw #L_1_chain = np.append(L_1_chain,L_1_draw)
		old_arg_per = arg_per_draw #L_1_chain = np.append(L_1_chain,L_1_draw)

		old_likelihood = new_likelihood
		chi_2_here = new_chi_2_here
		x_combo_old=x_combo_new

		y_combo_old=y_combo_new
		z_combo_old=z_combo_new

		accept+=1
		#checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p)  + " " + str(old_L_1_dot) + " " + str(old_L_1_p_dot) + " " + str(old_L_2_dot) + " " + str(old_L_2_p_dot) + " " + str(old_L_3_dot) + " " + str(old_L_3_p_dot) + " " + str(accept/k) + "\n")
		checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(old_minit1) + " " + str(old_semi_major) + " " + str(old_Omega_1) + " " + str(old_arg_per) + " " + str(accept/k) + "\n")


	elif (alpha < u) or math.isnan(alpha):
		#checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p)  + " " + str(old_L_1_dot) + " " + str(old_L_1_p_dot) + " " + str(old_L_2_dot) + " " + str(old_L_2_p_dot) + " " + str(old_L_3_dot) + " " + str(old_L_3_p_dot) + " " + str(accept/k) + "\n")
		checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(old_minit1) + " " + str(old_semi_major) + " " + str(old_Omega_1) + " " + str(old_arg_per) + " " + str(accept/k) + "\n")


	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1	

	#...........................semi-major........................................
	#...........................semi-major........................................
	#...........................semi-major........................................	
	semi_major_draw = proposal_semi_major(old_semi_major, draw_count_semi_major)
	delays_in_time_new = time_dependence(old_minit1,semi_major_draw,old_Omega_1,old_arg_per)
	x_combo_new = x_combo_2_0(delays_in_time_new)
	y_combo_new = y_combo_2_0(delays_in_time_new)
	z_combo_new = z_combo_2_0(delays_in_time_new)
	new_likelihood,new_chi_2_here  = likelihood_analytical_equal_arm(x_combo_new,y_combo_new,z_combo_new)
	
	alpha = min(np.log(prior_semi_major(semi_major_draw))+new_likelihood-np.log(prior_semi_major(old_semi_major))-old_likelihood,0)

	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_semi_major = semi_major_draw #L_1_chain = np.append(L_1_chain,L_1_draw)

		old_likelihood = new_likelihood
		chi_2_here = new_chi_2_here
		x_combo_old=x_combo_new
		y_combo_old=y_combo_new
		z_combo_old=z_combo_new

		accept+=1
		#checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p)  + " " + str(old_L_1_dot) + " " + str(old_L_1_p_dot) + " " + str(old_L_2_dot) + " " + str(old_L_2_p_dot) + " " + str(old_L_3_dot) + " " + str(old_L_3_p_dot) + " " + str(accept/k) + "\n")
		checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(old_minit1) + " " + str(old_semi_major) + " " + str(old_Omega_1) + " " + str(old_arg_per) + " " + str(accept/k) + "\n")


	elif (alpha < u) or math.isnan(alpha):
		#checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p)  + " " + str(old_L_1_dot) + " " + str(old_L_1_p_dot) + " " + str(old_L_2_dot) + " " + str(old_L_2_p_dot) + " " + str(old_L_3_dot) + " " + str(old_L_3_p_dot) + " " + str(accept/k) + "\n")
		checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(old_minit1) + " " + str(old_semi_major) + " " + str(old_Omega_1) + " " + str(old_arg_per) + " " + str(accept/k) + "\n")


	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1	
	
	#...........................Omega 1........................................
	#...........................Omega 1........................................
	#...........................Omega 1........................................	
	Omega_1_draw = proposal_Omega_1(old_Omega_1, draw_count_Omega_1)
	delays_in_time_new = time_dependence(old_minit1,old_semi_major,Omega_1_draw,old_arg_per)
	x_combo_new = x_combo_2_0(delays_in_time_new)
	y_combo_new = y_combo_2_0(delays_in_time_new)
	z_combo_new = z_combo_2_0(delays_in_time_new)
	new_likelihood,new_chi_2_here  = likelihood_analytical_equal_arm(x_combo_new,y_combo_new,z_combo_new)
	
	alpha = min(np.log(prior_Omega_1(Omega_1_draw))+new_likelihood-np.log(prior_Omega_1(old_Omega_1))-old_likelihood,0)

	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_Omega_1 = Omega_1_draw #L_1_chain = np.append(L_1_chain,L_1_draw)

		old_likelihood = new_likelihood
		chi_2_here = new_chi_2_here
		x_combo_old=x_combo_new
		y_combo_old=y_combo_new
		z_combo_old=z_combo_new

		accept+=1
		#checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p)  + " " + str(old_L_1_dot) + " " + str(old_L_1_p_dot) + " " + str(old_L_2_dot) + " " + str(old_L_2_p_dot) + " " + str(old_L_3_dot) + " " + str(old_L_3_p_dot) + " " + str(accept/k) + "\n")
		checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(old_minit1) + " " + str(old_semi_major) + " " + str(old_Omega_1) + " " + str(old_arg_per) + " " + str(accept/k) + "\n")


	elif (alpha < u) or math.isnan(alpha):
		#checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p)  + " " + str(old_L_1_dot) + " " + str(old_L_1_p_dot) + " " + str(old_L_2_dot) + " " + str(old_L_2_p_dot) + " " + str(old_L_3_dot) + " " + str(old_L_3_p_dot) + " " + str(accept/k) + "\n")
		checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(old_minit1) + " " + str(old_semi_major) + " " + str(old_Omega_1) + " " + str(old_arg_per) + " " + str(accept/k) + "\n")


	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1	
	
	#...........................arg per........................................
	#...........................arg per........................................
	#...........................arg per........................................	
	arg_per_draw = proposal_arg_per(old_arg_per, draw_count_arg_per)
	delays_in_time_new = time_dependence(old_minit1,old_semi_major,old_Omega_1,arg_per_draw)
	x_combo_new = x_combo_2_0(delays_in_time_new)
	y_combo_new = y_combo_2_0(delays_in_time_new)
	z_combo_new = z_combo_2_0(delays_in_time_new)
	new_likelihood,new_chi_2_here  = likelihood_analytical_equal_arm(x_combo_new,y_combo_new,z_combo_new)
	
	alpha = min(np.log(prior_arg_per(arg_per_draw))+new_likelihood-np.log(prior_arg_per(old_arg_per))-old_likelihood,0)

	u = np.log(np.random.uniform(0.000,1.000))

	if alpha >= u:
		old_arg_per = arg_per_draw #L_1_chain = np.append(L_1_chain,L_1_draw)

		old_likelihood = new_likelihood
		chi_2_here = new_chi_2_here
		x_combo_old=x_combo_new
		y_combo_old=y_combo_new
		z_combo_old=z_combo_new

		accept+=1
		#checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p)  + " " + str(old_L_1_dot) + " " + str(old_L_1_p_dot) + " " + str(old_L_2_dot) + " " + str(old_L_2_p_dot) + " " + str(old_L_3_dot) + " " + str(old_L_3_p_dot) + " " + str(accept/k) + "\n")
		checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(old_minit1) + " " + str(old_semi_major) + " " + str(old_Omega_1) + " " + str(old_arg_per) + " " + str(accept/k) + "\n")


	elif (alpha < u) or math.isnan(alpha):
		#checkfile.write(str(old_likelihood) + " " + str(determ_here) + " " + str(chi_2_here) + " " + str(old_L_1) + " " + str(old_L_1_p) + " " + str(old_L_2) + " " + str(old_L_2_p) + " " + str(old_L_3) + " " + str(old_L_3_p)  + " " + str(old_L_1_dot) + " " + str(old_L_1_p_dot) + " " + str(old_L_2_dot) + " " + str(old_L_2_p_dot) + " " + str(old_L_3_dot) + " " + str(old_L_3_p_dot) + " " + str(accept/k) + "\n")
		checkfile.write(str(old_likelihood) + " " + str(chi_2_here) + " " + str(old_minit1) + " " + str(old_semi_major) + " " + str(old_Omega_1) + " " + str(old_arg_per) + " " + str(accept/k) + "\n")


	else:
		print('something wrong in acceptance/rejection step')
	#next iteration
	k+=1	
	
	


	if draw_count_minit == 0 or draw_count_minit == 1:
		draw_count_minit+=1
	else:
		draw_count_minit = 0	

	
		
	#Now only drawing between 2 proposal distributions for L dot; excluding uniform draws because want to avoid finite range
	if draw_count_semi_major == 0:
	    draw_count_semi_major+=1
	
	else:
		draw_count_semi_major = 0	
		
	if draw_count_Omega_1 == 0 or draw_count_Omega_1 ==1 or draw_count_Omega_1 ==2:
		draw_count_Omega_1+=1
	
	else:
		draw_count_Omega_1 = 0	
	
	if draw_count_arg_per == 0 or draw_count_arg_per == 1:
		draw_count_arg_per+=1
	
	else:
		draw_count_arg_per = 0	


checkfile.close()
print('acceptance ratio')
print(accept/number_chain)

print('number of new proposals accepted')
print(accept)
'''
print("--- %s seconds ---" % (time.time() - start_time))