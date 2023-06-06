import sys
#from mpmath import binomial,log,pi,ceil
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import fftconvolve,kaiser,kaiser_beta
#from mpmath import *
import time
from scipy.stats import norm, multivariate_normal
from astropy import constants as const
import numpy as np
#from scipy.signal import butter,filtfilt
from scipy.special import gamma,binom,comb
import h5py   
#import nexusformat.nexus as nx
from matplotlib import rc
import matplotlib as mpl
#import mpld3
#mpld3.enable_notebook()

#from sincinterpol import interp
#from shift_roll import roll_zeropad
#import scipy.sparse as sp
#import bottleneck as bn
#from numba import jit
import timeit


# # CREATING FILTERS
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (8.6,8.6)
plt.rcParams['font.size'] = 20
mpl.rcParams['lines.linewidth'] = 1.5

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


# In[3]:


#def delay_factorial_polynomials(D):
#@jit(cache=True)

def filters_lagrange_2_0(D):

    #start_time = time.process_time()
    D=D*f_samp
    integer_part, d_frac = np.divmod(D,1)

    integer_part = integer_part-p
    d_frac = d_frac+p

    delay_polynomials = np.ones((number_n+1,length))

    #factors = np.array([-1*d_frac+i for i in ints])
    factors = -1*d_frac+ints

    '''
    for i in np.arange(1,number_n+1):

        delay_polynomials[i] = np.prod(factors[:i],axis=0)    
    '''
    delay_polynomials[1:number_n+1] = np.cumprod(factors,axis=0)
    #print("--- %s seconds for filters lagrange---" % (time.process_time() - start_time))

    return delay_polynomials,int(integer_part[0])


# In[4]:


def trim_data(data,filter_array):
    #start_time = time.process_time()

    if matrix==True:

        data=np.roll(data,filter_array[1],axis=0)
        data[:filter_array[1]] = 0.0
        #val = np.einsum('ij,ji->i',data,filter_array[0],optimize=einsum_path_to_use[0])
        val = np.einsum('ij,ji->i',data,filter_array[0],optimize=False)

        #print("--- %s seconds for trim_data---" % (time.process_time() - start_time))

        return val


    else:
        return fftconvolve(data,filter_array,'same')


# In[5]:


def S_y_proof_mass_new_frac_freq(f):

    #f[0]=f[1]
    pm_here =  np.power(3.0e-15,2)*(1+np.power(4.0e-4/f,2))*(1+np.power(f/8.0e-3,4))
    #*np.power(2*np.pi*f,-4)
    return pm_here*np.power(2*np.pi*f*const.c.value,-2)


# In[6]:


def S_y_OMS_frac_freq(f):
    #f[0]=f[1]

    op_here =  np.power(1.5e-11,2)*np.power(2*np.pi*f/const.c.value,2)*(1+np.power(2.0e-3/f,4))
    return op_here


# In[7]:



def time_dependence(delay_list,delay_dot_list,model):
    number_delays = len(delay_list)
    if model=='linear':
        delay_in_time = np.empty((number_delays,length))
        delay_dot_in_time = np.empty((number_delays,length))

        for i in np.arange(number_delays):
            delay_in_time[i] = delay_list[i] + delay_dot_list[i]*times
            delay_dot_in_time[i] = np.gradient(delay_in_time[i],1/f_s)
        return delay_in_time,delay_dot_in_time


            


# In[8]:


#Eq. (**) on page 1100 of notes
#@jit(cache=True)
def nested_delay_application(delay_array_here,delay_dot_array_here):
    

    number_delays = len(delay_array_here)

    delay_array_here, delay_dot_array_here = time_dependence(delay_array_here,delay_dot_array_here,'linear')

    '''
    for i in np.arange(number_delays):
        delay_array_here[i] = (1-delay_dot_array_here[i])*delay_array_here[i]                  
    '''


    correction_factor =np.zeros(length)

    for i in np.arange(number_delays):
        for j in np.arange(i+1,number_delays):

            correction_factor+=delay_array_here[i]*delay_dot_array_here[j]          


    doppler_factor = np.sum(delay_dot_array_here,axis=0)

    commutative_sum = np.sum(delay_array_here,axis=0)


    #print("--- %s seconds for linear delay application---" % (time.process_time() - start_time))

    return commutative_sum, np.gradient(commutative_sum,1/f_s), correction_factor
    #return commutative_sum, 0.0, correction_factor



# In[9]:


def x_combo_2_0(delay_array,delay_dot_array):

    L12 = nested_delay_application([delay_array[5]],[delay_dot_array[5]])
    L12_L21 = nested_delay_application([delay_array[5],delay_array[4]],[delay_dot_array[5],delay_dot_array[4]])
    L12_L21_L13 = nested_delay_application([delay_array[5],delay_array[4],delay_array[2]],[delay_dot_array[5],delay_dot_array[4],delay_dot_array[2]])
    L12_L21_L13_L31 = nested_delay_application([delay_array[5],delay_array[4],delay_array[2],delay_array[3]],[delay_dot_array[5],delay_dot_array[4],delay_dot_array[2],delay_dot_array[3]])
    L12_L21_L13_L31_L13 = nested_delay_application([delay_array[5],delay_array[4],delay_array[2],delay_array[3],delay_array[2]],[delay_dot_array[5],delay_dot_array[4],delay_dot_array[2],delay_dot_array[3],delay_dot_array[2]])
    L12_L21_L13_L31_L13_L31 = nested_delay_application([delay_array[5],delay_array[4],delay_array[2],delay_array[3],delay_array[2],delay_array[3]],[delay_dot_array[5],delay_dot_array[4],delay_dot_array[2],delay_dot_array[3],delay_dot_array[2],delay_dot_array[3]])
    L12_L21_L13_L31_L13_L31_L12 = nested_delay_application([delay_array[5],delay_array[4],delay_array[2],delay_array[3],delay_array[2],delay_array[3],delay_array[5]],[delay_dot_array[5],delay_dot_array[4],delay_dot_array[2],delay_dot_array[3],delay_dot_array[2],delay_dot_array[3],delay_dot_array[5]])
    L12_L21_L13_L31_L13_L31_L12_L21 = nested_delay_application([delay_array[5],delay_array[4],delay_array[2],delay_array[3],delay_array[2],delay_array[3],delay_array[5],delay_array[4]],[delay_dot_array[5],delay_dot_array[4],delay_dot_array[2],delay_dot_array[3],delay_dot_array[2],delay_dot_array[3],delay_dot_array[5],delay_dot_array[4]])

    L13 = nested_delay_application([delay_array[2]],[delay_dot_array[2]])
    L13_L31 = nested_delay_application([delay_array[2],delay_array[3]],[delay_dot_array[2],delay_dot_array[3]])
    L13_L31_L12 = nested_delay_application([delay_array[2],delay_array[3],delay_array[5]],[delay_dot_array[2],delay_dot_array[3],delay_dot_array[5]])
    L13_L31_L12_L21 = nested_delay_application([delay_array[2],delay_array[3],delay_array[5],delay_array[4]],[delay_dot_array[2],delay_dot_array[3],delay_dot_array[5],delay_dot_array[4]])
    L13_L31_L12_L21_L12 = nested_delay_application([delay_array[2],delay_array[3],delay_array[5],delay_array[4],delay_array[5]],[delay_dot_array[2],delay_dot_array[3],delay_dot_array[5],delay_dot_array[4],delay_dot_array[5]])
    L13_L31_L12_L21_L12_L21 = nested_delay_application([delay_array[2],delay_array[3],delay_array[5],delay_array[4],delay_array[5],delay_array[4]],[delay_dot_array[2],delay_dot_array[3],delay_dot_array[5],delay_dot_array[4],delay_dot_array[5],delay_dot_array[4]])
    L13_L31_L12_L21_L12_L21_L13 = nested_delay_application([delay_array[2],delay_array[3],delay_array[5],delay_array[4],delay_array[5],delay_array[4],delay_array[2]],[delay_dot_array[2],delay_dot_array[3],delay_dot_array[5],delay_dot_array[4],delay_dot_array[5],delay_dot_array[4],delay_dot_array[2]])
    L13_L31_L12_L21_L12_L21_L13_L31 = nested_delay_application([delay_array[2],delay_array[3],delay_array[5],delay_array[4],delay_array[5],delay_array[4],delay_array[2],delay_array[3]],[delay_dot_array[2],delay_dot_array[3],delay_dot_array[5],delay_dot_array[4],delay_dot_array[5],delay_dot_array[4],delay_dot_array[2],delay_dot_array[3]])


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


# In[10]:


def y_combo_2_0(delay_array,delay_dot_array):

    L23 = nested_delay_application([delay_array[1]],[delay_dot_array[1]])
    L23_L32 = nested_delay_application([delay_array[1],delay_array[0]],[delay_dot_array[1],delay_dot_array[0]])
    L23_L32_L21 = nested_delay_application([delay_array[1],delay_array[0],delay_array[4]],[delay_dot_array[1],delay_dot_array[0],delay_dot_array[4]])
    L23_L32_L21_L12 = nested_delay_application([delay_array[1],delay_array[0],delay_array[4],delay_array[5]],[delay_dot_array[1],delay_dot_array[0],delay_dot_array[4],delay_dot_array[5]])
    L23_L32_L21_L12_L21 = nested_delay_application([delay_array[1],delay_array[0],delay_array[4],delay_array[5],delay_array[4]],[delay_dot_array[1],delay_dot_array[0],delay_dot_array[4],delay_dot_array[5],delay_dot_array[4]])
    L23_L32_L21_L12_L21_L12 = nested_delay_application([delay_array[1],delay_array[0],delay_array[4],delay_array[5],delay_array[4],delay_array[5]],[delay_dot_array[1],delay_dot_array[0],delay_dot_array[4],delay_dot_array[5],delay_dot_array[4],delay_dot_array[5]])
    L23_L32_L21_L12_L21_L12_L23 = nested_delay_application([delay_array[1],delay_array[0],delay_array[4],delay_array[5],delay_array[4],delay_array[5],delay_array[1]],[delay_dot_array[1],delay_dot_array[0],delay_dot_array[4],delay_dot_array[5],delay_dot_array[4],delay_dot_array[5],delay_dot_array[1]])
    L23_L32_L21_L12_L21_L12_L23_L32 = nested_delay_application([delay_array[1],delay_array[0],delay_array[4],delay_array[5],delay_array[4],delay_array[5],delay_array[1],delay_array[0]],[delay_dot_array[1],delay_dot_array[0],delay_dot_array[4],delay_dot_array[5],delay_dot_array[4],delay_dot_array[5],delay_dot_array[1],delay_dot_array[0]])

    L21 = nested_delay_application([delay_array[4]],[delay_dot_array[4]])
    L21_L12 = nested_delay_application([delay_array[4],delay_array[5]],[delay_dot_array[4],delay_dot_array[5]])
    L21_L12_L23 = nested_delay_application([delay_array[4],delay_array[5],delay_array[1]],[delay_dot_array[4],delay_dot_array[5],delay_dot_array[1]])
    L21_L12_L23_L32 = nested_delay_application([delay_array[4],delay_array[5],delay_array[1],delay_array[0]],[delay_dot_array[4],delay_dot_array[5],delay_dot_array[1],delay_dot_array[0]])
    L21_L12_L23_L32_L23 = nested_delay_application([delay_array[4],delay_array[5],delay_array[1],delay_array[0],delay_array[1]],[delay_dot_array[4],delay_dot_array[5],delay_dot_array[1],delay_dot_array[0],delay_dot_array[1]])
    L21_L12_L23_L32_L23_L32 = nested_delay_application([delay_array[4],delay_array[5],delay_array[1],delay_array[0],delay_array[1],delay_array[0]],[delay_dot_array[4],delay_dot_array[5],delay_dot_array[1],delay_dot_array[0],delay_dot_array[1],delay_dot_array[0]])
    L21_L12_L23_L32_L23_L32_L21 = nested_delay_application([delay_array[4],delay_array[5],delay_array[1],delay_array[0],delay_array[1],delay_array[0],delay_array[4]],[delay_dot_array[4],delay_dot_array[5],delay_dot_array[1],delay_dot_array[0],delay_dot_array[1],delay_dot_array[0],delay_dot_array[4]])
    L21_L12_L23_L32_L23_L32_L21_L12 = nested_delay_application([delay_array[4],delay_array[5],delay_array[1],delay_array[0],delay_array[1],delay_array[0],delay_array[4],delay_array[5]],[delay_dot_array[4],delay_dot_array[5],delay_dot_array[1],delay_dot_array[0],delay_dot_array[1],delay_dot_array[0],delay_dot_array[4],delay_dot_array[5]])


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
    plt.plot(s12_,label = 's12_')
    plt.plot(y_combo,label='y combo')
    plt.plot(window*y_combo,label = 'Kaiser windowed')
    plt.legend()
    plt.show()
    '''
    y_f = np.fft.rfft(window*y_combo,norm='ortho')[indices_f_band]

    return [np.real(y_f),np.imag(y_f)]


# In[11]:


def z_combo_2_0(delay_array,delay_dot_array):

    L31 = nested_delay_application([delay_array[3]],[delay_dot_array[3]])
    L31_L13 = nested_delay_application([delay_array[3],delay_array[2]],[delay_dot_array[3],delay_dot_array[2]])
    L31_L13_L32 = nested_delay_application([delay_array[3],delay_array[2],delay_array[0]],[delay_dot_array[3],delay_dot_array[2],delay_dot_array[0]])
    L31_L13_L32_L23 = nested_delay_application([delay_array[3],delay_array[2],delay_array[0],delay_array[1]],[delay_dot_array[3],delay_dot_array[2],delay_dot_array[0],delay_dot_array[1]])
    L31_L13_L32_L23_L32 = nested_delay_application([delay_array[3],delay_array[2],delay_array[0],delay_array[1],delay_array[0]],[delay_dot_array[3],delay_dot_array[2],delay_dot_array[0],delay_dot_array[1],delay_dot_array[0]])
    L31_L13_L32_L23_L32_L23 = nested_delay_application([delay_array[3],delay_array[2],delay_array[0],delay_array[1],delay_array[0],delay_array[1]],[delay_dot_array[3],delay_dot_array[2],delay_dot_array[0],delay_dot_array[1],delay_dot_array[0],delay_dot_array[1]])
    L31_L13_L32_L23_L32_L23_L31 = nested_delay_application([delay_array[3],delay_array[2],delay_array[0],delay_array[1],delay_array[0],delay_array[1],delay_array[3]],[delay_dot_array[3],delay_dot_array[2],delay_dot_array[0],delay_dot_array[1],delay_dot_array[0],delay_dot_array[1],delay_dot_array[3]])
    L31_L13_L32_L23_L32_L23_L31_L13 = nested_delay_application([delay_array[3],delay_array[2],delay_array[0],delay_array[1],delay_array[0],delay_array[1],delay_array[3],delay_array[2]],[delay_dot_array[3],delay_dot_array[2],delay_dot_array[0],delay_dot_array[1],delay_dot_array[0],delay_dot_array[1],delay_dot_array[3],delay_dot_array[2]])

    L32 = nested_delay_application([delay_array[0]],[delay_dot_array[0]])
    L32_L23 = nested_delay_application([delay_array[0],delay_array[1]],[delay_dot_array[0],delay_dot_array[1]])
    L32_L23_L31 = nested_delay_application([delay_array[0],delay_array[1],delay_array[3]],[delay_dot_array[0],delay_dot_array[1],delay_dot_array[3]])
    L32_L23_L31_L13 = nested_delay_application([delay_array[0],delay_array[1],delay_array[3],delay_array[2]],[delay_dot_array[0],delay_dot_array[1],delay_dot_array[3],delay_dot_array[2]])
    L32_L23_L31_L13_L31 = nested_delay_application([delay_array[0],delay_array[1],delay_array[3],delay_array[2],delay_array[3]],[delay_dot_array[0],delay_dot_array[1],delay_dot_array[3],delay_dot_array[2],delay_dot_array[3]])
    L32_L23_L31_L13_L31_L13 = nested_delay_application([delay_array[0],delay_array[1],delay_array[3],delay_array[2],delay_array[3],delay_array[2]],[delay_dot_array[0],delay_dot_array[1],delay_dot_array[3],delay_dot_array[2],delay_dot_array[3],delay_dot_array[2]])
    L32_L23_L31_L13_L31_L13_L32 = nested_delay_application([delay_array[0],delay_array[1],delay_array[3],delay_array[2],delay_array[3],delay_array[2],delay_array[0]],[delay_dot_array[0],delay_dot_array[1],delay_dot_array[3],delay_dot_array[2],delay_dot_array[3],delay_dot_array[2],delay_dot_array[0]])
    L32_L23_L31_L13_L31_L13_L32_L23 = nested_delay_application([delay_array[0],delay_array[1],delay_array[3],delay_array[2],delay_array[3],delay_array[2],delay_array[0],delay_array[1]],[delay_dot_array[0],delay_dot_array[1],delay_dot_array[3],delay_dot_array[2],delay_dot_array[3],delay_dot_array[2],delay_dot_array[0],delay_dot_array[1]])


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


# In[12]:


def covariance_equal_arm(f,Sy_OP,Sy_PM):
    
    a = 16*np.power(np.sin(2*np.pi*f*avg_L),2)*Sy_OP+(8*np.power(np.sin(4*np.pi*f*avg_L),2)+32*np.power(np.sin(2*np.pi*f*avg_L),2))*Sy_PM
    b_ = -4*np.sin(2*np.pi*f*avg_L)*np.sin(4*np.pi*f*avg_L)*(4*Sy_PM+Sy_OP)

    return 2*a,2*b_


# In[13]:


#@jit(nopython=True,parallel=True,cache=True)
def likelihood_analytical_equal_arm(x,y,z):


	plt.loglog(np.fft.rfftfreq(length,1/f_s)[indices_f_band],np.abs(np.fft.rfft(s12,norm='ortho')[indices_f_band])**2,label=r'$s_{12}$',color='black')
	plt.loglog(f_band,np.power(np.abs(x[0]+x[1]),2),label=r'$X_{2}$',color='magenta')
	plt.loglog(f_band,np.power(np.abs(y[0]+y[1]),2),label=r'$Y_{2}$',color='brown')
	plt.loglog(f_band,np.power(np.abs(z[0]+z[1]),2),label=r'$Z_{2}$',color='red')
	#plt.loglog(f_band, np.power((2 * np.pi * f_band) * asd_nu * 9e-12/central_freq,2), c='gray', ls='--')
	plt.loglog(f_band,a,label=r'$\Sigma_{00}$',color='grey',linestyle='--')
	plt.xlabel(r'f (Hz)')
	plt.ylabel(r'PSD (Hz/Hz)')
	#plt.axvline(f_max,label='f max cut-off')
	#plt.xlim(1e-4,0.1)
	plt.legend(loc='best')
	#plt.title('Linear Approximation')
	#plt.title('L(t) Nested Delays')
	#plt.title('Non-Moving Arms, Doppler, Ranging and Clock Noises Disabled')
	plt.savefig('One_hour_linear_residual.png')
	#plt.savefig('One_Day_True_Positions.png')

	plt.show()    
	chi_2 = 1/determinant*(A_*(x[0]**2+x[1]**2+y[0]**2+y[1]**2+z[0]**2+z[1]**2) + 2*B_*(x[0]*y[0]+x[1]*y[1]+x[0]*z[0]+x[1]*z[1]+y[0]*z[0]+y[1]*z[1]))

	value = -1*np.sum(chi_2) - log_term_factor - np.sum(log_term_determinant)

	return value,np.sum(log_term_determinant),np.sum(chi_2)	

#........................................................................................
#...........................MCMC Functions.......................................
#........................................................................................
def proposal(mean, draw):

	
	if draw == 0:

		new_val = np.random.normal(mean,1e-2)
		#new_val = np.random.normal(mean,(1000/const.c.value))

		return new_val
	
	elif draw == 1:

		new_val = np.random.normal(mean,1e-4)
		return new_val
	
	elif draw == 2:

		new_val = np.random.normal(mean,1e-9)
		return new_val
	'''
	elif draw == 3:

		new_val = np.random.normal(mean,1e-12)
		return new_val
	'''
def proposal_dot(mean,draw):

	if draw==0:
	
		new_val = np.array(np.random.normal(mean,1e-10))
	elif draw==1:
	
		new_val = np.array(np.random.normal(mean,1e-11))
	elif draw==2:
	
		new_val = np.array(np.random.normal(mean,1e-12))
	'''
		
	elif draw==3:
		new_val = np.array(np.random.normal(mean,1e-11))
	'''
	return new_val


def prior(val):
	val = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (high-low)*uniform.pdf(val[0],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[1],low,high-low) ==1.0 and (high-low)*uniform.pdf(val[2],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[3],low,high-low) ==1.0:
	if (val >= low).all() and (val <= high).all():
		return 1
	else:
		return 0

def prior_dot(val):
	val = np.array(val)
	#return multivariate_normal.pdf(val,mean=[avg_L,avg_L,avg_L,avg_L],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#return multivariate_normal.pdf(val,mean=[L_3,L_2,L_3_p,L_2_p],cov=(1000/const.c.value)**2*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
	#if (high-low)*uniform.pdf(val[0],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[1],low,high-low) ==1.0 and (high-low)*uniform.pdf(val[2],low,high-low) == 1.0 and (high-low)*uniform.pdf(val[3],low,high-low) ==1.0:
	if (val >= low_dot).all() and (val <= high_dot).all():
		return 1
	else:
		return 0
	

# # Initial Parameters

# In[14]:


f_s = 4
f_samp = f_s
#lagrange filter length
number_n_data = 21
number_n_delays = 5
number_n = number_n_data
p = number_n//2
asd_nu = 28.8 # Hz/rtHz



f_min = 5.0e-4 # (= 0.0009765625)
f_max = 0.03
central_freq=281600000000000.0
L_arm = 2.5e9
avg_L = L_arm/const.c.value

static=False
equalarmlength=False
keplerian=True


is_linear=True

matrix=True

is_tcb = False



# # Load Data and Cut off Initial Samples

# In[15]:


if static==True:
    #data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_static_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
    #data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_equalarmlength_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
    data=np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/tryer_faster_codes/LISA_Instrument_RR_disable_all_but_laser_lock_six_static_orbits_tps_ppr_orbits_pyTDI_size_mprs_to_file.dat',names=True)

elif equalarmlength==True:
    data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_equalarmlength_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
elif keplerian==True:
    if is_tcb==True:
        data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_tcb_ltt_orbits.dat',names=True)

    else:
        #data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_tps_ppr_orbits_pyTDI_size.dat',names=True)
        #data=np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/tryer_faster_codes/LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_tps_ppr_orbits_pyTDI_size_mprs_to_file.dat',names=True)
        #data=np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/tryer_faster_codes/LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_tps_ppr_orbits_one_day_mprs_to_file.dat',names=True)
        #data=np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/tryer_faster_codes/LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_one_day.dat',names=True)
        #data=np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/tryer_faster_codes/LISA_Instrument_RR_disable_all_but_laser_lock_six_Keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_one_hour.dat',names=True)
        #data = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_9_hours_NO_AA_filter.dat',names=True)
        data = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_1_hour_NO_AA_filter.dat',names=True)
        #data = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_6_hour_NO_AA_filter.dat',names=True)

        #data= np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_orbits_tps_ppr_orbits_mprs_and_dpprs_to_file_1_day_NO_AA_filter.dat',names=True)
    #data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_tps_ppr_orbits_one_day.dat',names=True)

#data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_keplerian_tps_ppr_orbits_pyTDI_size.dat',names=True)
#data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_equalarmlength_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
#data =  np.genfromtxt('/Users/jessica/Desktop/Project_2/TDI_2.0/orbit_files/LISA_Instrument_RR_disable_all_but_laser_lock_six_static_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
initial_length = len(data['s31'])

cut_off = 0
#cut_off=int(1e3)
end_cut_off = initial_length+1

times = np.arange(initial_length)/f_s
#times=times[:initial_length-cut_off:]
times=times[cut_off::]


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


ppr_L_1 = data['mprs_32'][cut_off::]
ppr_L_1_p = data['mprs_23'][cut_off::]
ppr_L_2 = data['mprs_13'][cut_off::]
ppr_L_2_p = data['mprs_31'][cut_off::]
ppr_L_3 = data['mprs_21'][cut_off::]
ppr_L_3_p = data['mprs_12'][cut_off::]




if static ==True:
    d_ppr_L_1_dot = np.zeros(length-cut_off)[0]
    d_ppr_L_1_p_dot = np.zeros(length-cut_off)[0]
    d_ppr_L_2_dot = np.zeros(length-cut_off)[0]
    d_ppr_L_2_p_dot = np.zeros(length-cut_off)[0]
    d_ppr_L_3_dot = np.zeros(length-cut_off)[0]
    d_ppr_L_3_p_dot = np.zeros(length-cut_off)[0]
    
    
    L_1 = -1*ppr_L_1[0]
    print('L1')
    print(L_1)
    L_1_p = -1*ppr_L_1_p[0]
    L_2 =-1*ppr_L_2[0]
    L_2_p = -1*ppr_L_2_p[0]
    L_3 = -1*ppr_L_3[0]
    L_3_p = -1*ppr_L_3_p[0]
    
    L_1_dot=0.0
    L_1_p_dot=0.0
    L_2_dot=0.0
    L_2_p_dot=0.0
    L_3_dot=0.0
    L_3_p_dot=0.0


    
else:



    if is_linear==True:

        
        L_1 = ppr_L_1[0] 
        L_1_p = ppr_L_1_p[0] 
        L_2 = ppr_L_2[0]  
        L_2_p = ppr_L_2_p[0] 
        L_3 = ppr_L_3[0]  
        L_3_p = ppr_L_3_p[0] 

        L_1_dot = np.gradient(ppr_L_1,1/f_s)[0] 
        L_1_p_dot = np.gradient(ppr_L_1_p,1/f_s)[0] 
        L_2_dot = np.gradient(ppr_L_2,1/f_s)[0] 
        L_2_p_dot = np.gradient(ppr_L_2_p,1/f_s)[0] 
        L_3_dot = np.gradient(ppr_L_3,1/f_s)[0] 
        L_3_p_dot = np.gradient(ppr_L_3_p,1/f_s)[0]        
   

    elif is_linear==False:


        
        L_1 = ppr_L_1 
        L_1_p = ppr_L_1_p 
        L_2 = ppr_L_2 
        L_2_p = ppr_L_2_p
        L_3 = ppr_L_3 
        L_3_p = ppr_L_3_p

        L_1_dot = np.gradient(ppr_L_1,1/f_s)
        L_1_p_dot = np.gradient(ppr_L_1_p,1/f_s)
        L_2_dot = np.gradient(ppr_L_2,1/f_s)
        L_2_p_dot = np.gradient(ppr_L_2_p,1/f_s)
        L_3_dot = np.gradient(ppr_L_3,1/f_s)
        L_3_p_dot = np.gradient(ppr_L_3_p,1/f_s)
del data
#del f


    



# # Windowing and Noise Covariance Matrix Parameters

# In[18]:



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




plt.plot(times,ppr_L_1,label=r'$L_{32}(t)$ True time dependence')
plt.plot(times,time_dependence([L_1],[L_1_dot],'linear')[0][0],label=r'Linear $L_{32}(t)$ approximation')
plt.legend()
plt.xlabel('t (s)')
plt.ylabel(r'$L_{32}$ (s)')
plt.savefig('linear_time_dependence.png')
plt.show()


#plt.plot(times,ppr_L_1,label=r'$L_{32}(t)$ True time dependence')
plt.plot(times,ppr_L_1 - time_dependence([L_1],[L_1_dot],'linear')[0][0],color='k')
#plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'$\Delta L_{32}(t)$ [s]')
plt.savefig('linear_time_dependence_difference.png')
plt.show()


#........................................................................................
#...........................MCMC Portion.......................................
#........................................................................................



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
initial_L_1 = np.random.uniform(low,high)
initial_L_1_p = np.random.uniform(low,high)
initial_L_2 = np.random.uniform(low,high)
initial_L_2_p = np.random.uniform(low,high)
initial_L_3 = np.random.uniform(low,high)
initial_L_3_p = np.random.uniform(low,high)

initial_L_1_dot = np.random.uniform(low_dot,high_dot)
initial_L_1_p_dot = np.random.uniform(low_dot,high_dot)
initial_L_2_dot = np.random.uniform(low_dot,high_dot)
initial_L_2_p_dot = np.random.uniform(low_dot,high_dot)
initial_L_3_dot = np.random.uniform(low_dot,high_dot)
initial_L_3_p_dot = np.random.uniform(low_dot,high_dot)
'''
#initial delays accepted into the chain
accept = 1
x_combo_initial = x_combo_2_0([initial_L_1,initial_L_1_p,initial_L_2,initial_L_2_p,initial_L_3,initial_L_3_p],[initial_L_1_dot,initial_L_1_p_dot,initial_L_2_dot,initial_L_2_p_dot,initial_L_3_dot,initial_L_3_p_dot])
y_combo_initial = y_combo_2_0([initial_L_1,initial_L_1_p,initial_L_2,initial_L_2_p,initial_L_3,initial_L_3_p],[initial_L_1_dot,initial_L_1_p_dot,initial_L_2_dot,initial_L_2_p_dot,initial_L_3_dot,initial_L_3_p_dot])
z_combo_initial = z_combo_2_0([initial_L_1,initial_L_1_p,initial_L_2,initial_L_2_p,initial_L_3,initial_L_3_p],[initial_L_1_dot,initial_L_1_p_dot,initial_L_2_dot,initial_L_2_p_dot,initial_L_3_dot,initial_L_3_p_dot])






old_likelihood,determ_here,chi_2_here = likelihood_analytical_equal_arm(x_combo_initial,y_combo_initial,z_combo_initial)
print('likelihood')
print(old_likelihood)
sys.exit()
