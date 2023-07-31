import numpy as np
from .filter_functions import delay_polynomials, trim_data
from .. import settings

def nested_delay_application(delay_array_here,list_delays):
    number_delays = len(list_delays)
    
    delays = np.array([delay_array_here[j] for j in list_delays])


    delay_dot_array_here = np.gradient(delays,1/settings.f_s,axis=1,edge_order=2)

    correction_factor =np.zeros(settings.length)

    for i in np.arange(number_delays):
        for j in np.arange(i+1,number_delays):

            correction_factor+=delays[i]*delay_dot_array_here[j]          


    doppler_factor = np.sum(delay_dot_array_here,axis=0)

    commutative_sum = np.sum(delays,axis=0)



    return commutative_sum, np.gradient(commutative_sum,1/settings.f_s), correction_factor


def x_combo_2_0(delay_array,einsum_path_to_use):



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

    filter_L12 = delay_polynomials(L12[0])
    filter_L12_L21 = delay_polynomials(L12_L21[0])
    filter_L12_L21_L13 = delay_polynomials(L12_L21_L13[0])
    filter_L12_L21_L13_L31 = delay_polynomials(L12_L21_L13_L31[0])
    filter_L12_L21_L13_L31_L13 = delay_polynomials(L12_L21_L13_L31_L13[0])
    filter_L12_L21_L13_L31_L13_L31 = delay_polynomials(L12_L21_L13_L31_L13_L31[0])
    filter_L12_L21_L13_L31_L13_L31_L12 = delay_polynomials(L12_L21_L13_L31_L13_L31_L12[0])    
    filter_L12_L21_L13_L31_L13_L31_L12_L21 = delay_polynomials(L12_L21_L13_L31_L13_L31_L12_L21[0])

    filter_L13 = delay_polynomials(L13[0])
    filter_L13_L31 = delay_polynomials(L13_L31[0])
    filter_L13_L31_L12 = delay_polynomials(L13_L31_L12[0])
    filter_L13_L31_L12_L21 = delay_polynomials(L13_L31_L12_L21[0])
    filter_L13_L31_L12_L21_L12 = delay_polynomials(L13_L31_L12_L21_L12[0])
    filter_L13_L31_L12_L21_L12_L21 = delay_polynomials(L13_L31_L12_L21_L12_L21[0])
    filter_L13_L31_L12_L21_L12_L21_L13 = delay_polynomials(L13_L31_L12_L21_L12_L21_L13[0])
    filter_L13_L31_L12_L21_L12_L21_L13_L31 = delay_polynomials(L13_L31_L12_L21_L12_L21_L13_L31[0])

    x_combo = np.zeros(settings.length)
    x_combo = x_combo + (settings.s12 + 0.5*(settings.tau12-settings.eps12)) 

    next_term = trim_data((settings.tau21_coeffs-settings.eps21_coeffs) + settings.s21_coeffs,filter_L12,einsum_path_to_use) 
    x_combo = x_combo + ((1-L12[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L12[2]))

    next_term = trim_data(0.5*(settings.tau12_coeffs-settings.eps12_coeffs) + settings.s13_coeffs + 0.5*(settings.tau13_coeffs-settings.eps13_coeffs) + 0.5*(settings.tau12_coeffs-settings.tau13_coeffs),filter_L12_L21,einsum_path_to_use) 
    x_combo = x_combo + ((1-L12_L21[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L12_L21[2]))

    next_term = trim_data(0.5*(settings.tau31_coeffs-settings.eps31_coeffs) + settings.s31_coeffs + 0.5*(settings.tau31_coeffs-settings.eps31_coeffs),filter_L12_L21_L13,einsum_path_to_use) 
    x_combo = x_combo + ((1-L12_L21_L13[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L12_L21_L13[2]))

    next_term = trim_data((settings.tau13_coeffs-settings.eps13_coeffs) +settings.s13_coeffs,filter_L12_L21_L13_L31,einsum_path_to_use) 
    x_combo = x_combo + ((1-L12_L21_L13_L31[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L12_L21_L13_L31[2]))

    next_term = trim_data((settings.tau31_coeffs-settings.eps31_coeffs) + settings.s31_coeffs,filter_L12_L21_L13_L31_L13,einsum_path_to_use) 
    x_combo = x_combo + ((1-L12_L21_L13_L31_L13[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L12_L21_L13_L31_L13[2]))

    next_term = trim_data(0.5*(settings.tau13_coeffs-settings.eps13_coeffs) + 0.5*(settings.tau13_coeffs-settings.tau12_coeffs) + settings.s12_coeffs + 0.5*(settings.tau12_coeffs-settings.eps12_coeffs),filter_L12_L21_L13_L31_L13_L31,einsum_path_to_use) 
    x_combo = x_combo + ((1-L12_L21_L13_L31_L13_L31[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L12_L21_L13_L31_L13_L31[2]))

    next_term = trim_data((settings.tau21_coeffs-settings.eps21_coeffs) + settings.s21_coeffs,filter_L12_L21_L13_L31_L13_L31_L12,einsum_path_to_use) 
    x_combo = x_combo + ((1-L12_L21_L13_L31_L13_L31_L12[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L12_L21_L13_L31_L13_L31_L12[2]))

    next_term = trim_data(0.5*(settings.tau12_coeffs-settings.eps12_coeffs),filter_L12_L21_L13_L31_L13_L31_L12_L21,einsum_path_to_use) 
    x_combo = x_combo + ((1-L12_L21_L13_L31_L13_L31_L12_L21[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L12_L21_L13_L31_L13_L31_L12_L21[2]))

    #BEGIN NEGATIVE
    x_combo_minus = np.zeros(settings.length)

    x_combo_minus = x_combo_minus + (settings.s13 + 0.5*(settings.tau13-settings.eps13) + 0.5*(settings.tau12-settings.tau13)) 

    next_term = trim_data((settings.tau31_coeffs-settings.eps31_coeffs) + settings.s31_coeffs,filter_L13,einsum_path_to_use) 
    x_combo_minus = x_combo_minus + ((1-L13[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L13[2]))

    next_term = trim_data(0.5*(settings.tau13_coeffs-settings.eps13_coeffs) + 0.5*(settings.tau13_coeffs-settings.tau12_coeffs) + settings.s12_coeffs + 0.5*(settings.tau12_coeffs-settings.eps12_coeffs),filter_L13_L31,einsum_path_to_use) 
    x_combo_minus = x_combo_minus + ((1-L13_L31[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L13_L31[2]))

    next_term = trim_data((settings.tau21_coeffs-settings.eps21_coeffs) + settings.s21_coeffs,filter_L13_L31_L12,einsum_path_to_use) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L13_L31_L12[2]))

    next_term = trim_data((settings.tau12_coeffs-settings.eps12_coeffs) + settings.s12_coeffs,filter_L13_L31_L12_L21,einsum_path_to_use) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L13_L31_L12_L21[2]))

    next_term = trim_data((settings.tau21_coeffs-settings.eps21_coeffs) + settings.s21_coeffs,filter_L13_L31_L12_L21_L12,einsum_path_to_use) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21_L12[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L13_L31_L12_L21_L12[2]))

    next_term = trim_data(0.5*(settings.tau12_coeffs-settings.eps12_coeffs) + settings.s13_coeffs + 0.5*(settings.tau13_coeffs-settings.eps13_coeffs) + 0.5*(settings.tau12_coeffs-settings.tau13_coeffs),filter_L13_L31_L12_L21_L12_L21,einsum_path_to_use) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21_L12_L21[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L13_L31_L12_L21_L12_L21[2]))

    next_term = trim_data((settings.tau31_coeffs-settings.eps31_coeffs) + settings.s31_coeffs,filter_L13_L31_L12_L21_L12_L21_L13,einsum_path_to_use) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21_L12_L21_L13[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L13_L31_L12_L21_L12_L21_L13[2]))

    next_term = trim_data(0.5*(settings.tau13_coeffs-settings.eps13_coeffs) + 0.5*(settings.tau13_coeffs-settings.tau12_coeffs),filter_L13_L31_L12_L21_L12_L21_L13_L31,einsum_path_to_use) 
    x_combo_minus = x_combo_minus + ((1-L13_L31_L12_L21_L12_L21_L13_L31[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L13_L31_L12_L21_L12_L21_L13_L31[2]))

    x_combo = x_combo - x_combo_minus
    '''
    #np.savetxt('x_combo_full.dat',x_combo)
    plt.plot(s31,label = 's31')
    plt.plot(x_combo,label='x combo')
    plt.plot(window*x_combo,label = 'Kaiser windowed')
    plt.legend()
    plt.show()    
    '''

    
    x_f = np.fft.rfft(settings.window*x_combo,norm='ortho')[settings.indices_f_band]

    return [np.real(x_f),np.imag(x_f)]




def y_combo_2_0(delay_array,einsum_path_to_use):

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


    filter_L23 = delay_polynomials(L23[0])
    filter_L23_L32 = delay_polynomials(L23_L32[0])
    filter_L23_L32_L21 = delay_polynomials(L23_L32_L21[0])
    filter_L23_L32_L21_L12 = delay_polynomials(L23_L32_L21_L12[0])
    filter_L23_L32_L21_L12_L21 = delay_polynomials(L23_L32_L21_L12_L21[0])
    filter_L23_L32_L21_L12_L21_L12 = delay_polynomials(L23_L32_L21_L12_L21_L12[0])
    filter_L23_L32_L21_L12_L21_L12_L23 = delay_polynomials(L23_L32_L21_L12_L21_L12_L23[0])    
    filter_L23_L32_L21_L12_L21_L12_L23_L32 = delay_polynomials(L23_L32_L21_L12_L21_L12_L23_L32[0])

    filter_L21 = delay_polynomials(L21[0])
    filter_L21_L12 = delay_polynomials(L21_L12[0])
    filter_L21_L12_L23 = delay_polynomials(L21_L12_L23[0])
    filter_L21_L12_L23_L32 = delay_polynomials(L21_L12_L23_L32[0])
    filter_L21_L12_L23_L32_L23 = delay_polynomials(L21_L12_L23_L32_L23[0])
    filter_L21_L12_L23_L32_L23_L32 = delay_polynomials(L21_L12_L23_L32_L23_L32[0])
    filter_L21_L12_L23_L32_L23_L32_L21 = delay_polynomials(L21_L12_L23_L32_L23_L32_L21[0])
    filter_L21_L12_L23_L32_L23_L32_L21_L12 = delay_polynomials(L21_L12_L23_L32_L23_L32_L21_L12[0])

    y_combo = np.zeros(settings.length)
    y_combo = y_combo + (settings.s23 + 0.5*(settings.tau23-settings.eps23)) 

    next_term = trim_data((settings.tau32_coeffs-settings.eps32_coeffs) + settings.s32_coeffs,filter_L23,einsum_path_to_use) 
    y_combo = y_combo + ((1-L23[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L23[2]))

    next_term = trim_data(0.5*(settings.tau23_coeffs-settings.eps23_coeffs) + settings.s21_coeffs + 0.5*(settings.tau21_coeffs-settings.eps21_coeffs) + 0.5*(settings.tau23_coeffs-settings.tau21_coeffs),filter_L23_L32,einsum_path_to_use) 
    y_combo = y_combo + ((1-L23_L32[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L23_L32[2]))

    next_term = trim_data(0.5*(settings.tau12_coeffs-settings.eps12_coeffs) + settings.s12_coeffs + 0.5*(settings.tau12_coeffs-settings.eps12_coeffs),filter_L23_L32_L21,einsum_path_to_use) 
    y_combo = y_combo + ((1-L23_L32_L21[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L23_L32_L21[2]))

    next_term = trim_data((settings.tau21_coeffs-settings.eps21_coeffs) + settings.s21_coeffs,filter_L23_L32_L21_L12,einsum_path_to_use) 
    y_combo = y_combo + ((1-L23_L32_L21_L12[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L23_L32_L21_L12[2]))

    next_term = trim_data((settings.tau12_coeffs-settings.eps12_coeffs) + settings.s12_coeffs,filter_L23_L32_L21_L12_L21,einsum_path_to_use) 
    y_combo = y_combo + ((1-L23_L32_L21_L12_L21[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L23_L32_L21_L12_L21[2]))

    next_term = trim_data(0.5*(settings.tau21_coeffs-settings.eps21_coeffs) + 0.5*(settings.tau21_coeffs-settings.tau23_coeffs) + settings.s23_coeffs + 0.5*(settings.tau23_coeffs-settings.eps23_coeffs),filter_L23_L32_L21_L12_L21_L12,einsum_path_to_use) 
    y_combo = y_combo + ((1-L23_L32_L21_L12_L21_L12[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L23_L32_L21_L12_L21_L12[2]))

    next_term = trim_data((settings.tau32_coeffs-settings.eps32_coeffs) + settings.s32_coeffs,filter_L23_L32_L21_L12_L21_L12_L23,einsum_path_to_use) 
    y_combo = y_combo + ((1-L23_L32_L21_L12_L21_L12_L23[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L23_L32_L21_L12_L21_L12_L23[2]))

    next_term = trim_data(0.5*(settings.tau23_coeffs-settings.eps23_coeffs),filter_L23_L32_L21_L12_L21_L12_L23_L32,einsum_path_to_use) 
    y_combo = y_combo + ((1-L23_L32_L21_L12_L21_L12_L23_L32[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L23_L32_L21_L12_L21_L12_L23_L32[2]))

    #BEGIN NEGATIVE
    y_combo_minus = np.zeros(settings.length)

    y_combo_minus = y_combo_minus + (settings.s21 + 0.5*(settings.tau21-settings.eps21) + 0.5*(settings.tau23-settings.tau21)) 

    next_term = trim_data((settings.tau12_coeffs-settings.eps12_coeffs) + settings.s12_coeffs,filter_L21,einsum_path_to_use) 
    y_combo_minus = y_combo_minus + ((1-L21[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L21[2]))

    next_term = trim_data(0.5*(settings.tau21_coeffs-settings.eps21_coeffs) + 0.5*(settings.tau21_coeffs-settings.tau23_coeffs) + settings.s23_coeffs + 0.5*(settings.tau23_coeffs-settings.eps23_coeffs),filter_L21_L12,einsum_path_to_use) 
    y_combo_minus = y_combo_minus + ((1-L21_L12[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L21_L12[2]))

    next_term = trim_data((settings.tau32_coeffs-settings.eps32_coeffs) + settings.s32_coeffs,filter_L21_L12_L23,einsum_path_to_use) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L21_L12_L23[2]))

    next_term = trim_data((settings.tau23_coeffs-settings.eps23_coeffs) + settings.s23_coeffs,filter_L21_L12_L23_L32,einsum_path_to_use) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L21_L12_L23_L32[2]))

    next_term = trim_data((settings.tau32_coeffs-settings.eps32_coeffs) + settings.s32_coeffs,filter_L21_L12_L23_L32_L23,einsum_path_to_use) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32_L23[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L21_L12_L23_L32_L23[2]))

    next_term = trim_data(0.5*(settings.tau23_coeffs-settings.eps23_coeffs) + settings.s21_coeffs + 0.5*(settings.tau21_coeffs-settings.eps21_coeffs) + 0.5*(settings.tau23_coeffs-settings.tau21_coeffs),filter_L21_L12_L23_L32_L23_L32,einsum_path_to_use) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32_L23_L32[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L21_L12_L23_L32_L23_L32[2]))

    next_term = trim_data((settings.tau12_coeffs-settings.eps12_coeffs) + settings.s12_coeffs,filter_L21_L12_L23_L32_L23_L32_L21,einsum_path_to_use) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32_L23_L32_L21[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L21_L12_L23_L32_L23_L32_L21[2]))

    next_term = trim_data(0.5*(settings.tau21_coeffs-settings.eps21_coeffs) + 0.5*(settings.tau21_coeffs-settings.tau23_coeffs),filter_L21_L12_L23_L32_L23_L32_L21_L12,einsum_path_to_use) 
    y_combo_minus = y_combo_minus + ((1-L21_L12_L23_L32_L23_L32_L21_L12[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L21_L12_L23_L32_L23_L32_L21_L12[2]))

    y_combo = y_combo - y_combo_minus
    
    '''
    #np.savetxt('y_combo_full.dat',y_combo)
    plt.plot(s12,label = 's12_')
    plt.plot(y_combo,label='y combo')
    plt.plot(window*y_combo,label = 'Kaiser windowed')
    plt.legend()
    plt.show()    
    '''

    
    y_f = np.fft.rfft(settings.window*y_combo,norm='ortho')[settings.indices_f_band]

    return [np.real(y_f),np.imag(y_f)]


# In[25]:


def z_combo_2_0(delay_array,einsum_path_to_use):

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


    filter_L31 = delay_polynomials(L31[0])
    filter_L31_L13 = delay_polynomials(L31_L13[0])
    filter_L31_L13_L32 = delay_polynomials(L31_L13_L32[0])
    filter_L31_L13_L32_L23 = delay_polynomials(L31_L13_L32_L23[0])
    filter_L31_L13_L32_L23_L32 = delay_polynomials(L31_L13_L32_L23_L32[0])
    filter_L31_L13_L32_L23_L32_L23 = delay_polynomials(L31_L13_L32_L23_L32_L23[0])
    filter_L31_L13_L32_L23_L32_L23_L31 = delay_polynomials(L31_L13_L32_L23_L32_L23_L31[0])    
    filter_L31_L13_L32_L23_L32_L23_L31_L13 = delay_polynomials(L31_L13_L32_L23_L32_L23_L31_L13[0])

    filter_L32 = delay_polynomials(L32[0])
    filter_L32_L23 = delay_polynomials(L32_L23[0])
    filter_L32_L23_L31 = delay_polynomials(L32_L23_L31[0])
    filter_L32_L23_L31_L13 = delay_polynomials(L32_L23_L31_L13[0])
    filter_L32_L23_L31_L13_L31 = delay_polynomials(L32_L23_L31_L13_L31[0])
    filter_L32_L23_L31_L13_L31_L13 = delay_polynomials(L32_L23_L31_L13_L31_L13[0])
    filter_L32_L23_L31_L13_L31_L13_L32 = delay_polynomials(L32_L23_L31_L13_L31_L13_L32[0])
    filter_L32_L23_L31_L13_L31_L13_L32_L23 = delay_polynomials(L32_L23_L31_L13_L31_L13_L32_L23[0])

        
    z_combo = np.zeros(settings.length)
    z_combo = z_combo + (settings.s31 + 0.5*(settings.tau31-settings.eps31)) 

    next_term = trim_data((settings.tau13_coeffs-settings.eps13_coeffs) + settings.s13_coeffs,filter_L31,einsum_path_to_use) 
    z_combo = z_combo + ((1-L31[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L31[2]))

    next_term = trim_data(0.5*(settings.tau31_coeffs-settings.eps31_coeffs) + settings.s32_coeffs + 0.5*(settings.tau32_coeffs-settings.eps32_coeffs) + 0.5*(settings.tau31_coeffs-settings.tau32_coeffs),filter_L31_L13,einsum_path_to_use) 
    z_combo = z_combo + ((1-L31_L13[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L31_L13[2]))

    next_term = trim_data(0.5*(settings.tau23_coeffs-settings.eps23_coeffs) + settings.s23_coeffs + 0.5*(settings.tau23_coeffs-settings.eps23_coeffs),filter_L31_L13_L32,einsum_path_to_use) 
    z_combo = z_combo + ((1-L31_L13_L32[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L31_L13_L32[2]))

    next_term = trim_data((settings.tau32_coeffs-settings.eps32_coeffs) + settings.s32_coeffs,filter_L31_L13_L32_L23,einsum_path_to_use) 
    z_combo = z_combo + ((1-L31_L13_L32_L23[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L31_L13_L32_L23[2]))

    next_term = trim_data((settings.tau23_coeffs-settings.eps23_coeffs) + settings.s23_coeffs,filter_L31_L13_L32_L23_L32,einsum_path_to_use) 
    z_combo = z_combo + ((1-L31_L13_L32_L23_L32[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L31_L13_L32_L23_L32[2]))

    next_term = trim_data(0.5*(settings.tau32_coeffs-settings.eps32_coeffs) + 0.5*(settings.tau32_coeffs-settings.tau31_coeffs) + settings.s31_coeffs + 0.5*(settings.tau31_coeffs-settings.eps31_coeffs),filter_L31_L13_L32_L23_L32_L23,einsum_path_to_use) 
    z_combo = z_combo + ((1-L31_L13_L32_L23_L32_L23[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L31_L13_L32_L23_L32_L23[2]))

    next_term = trim_data((settings.tau13_coeffs-settings.eps13_coeffs) + settings.s13_coeffs,filter_L31_L13_L32_L23_L32_L23_L31,einsum_path_to_use) 
    z_combo = z_combo + ((1-L31_L13_L32_L23_L32_L23_L31[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L31_L13_L32_L23_L32_L23_L31[2]))

    next_term = trim_data(0.5*(settings.tau31_coeffs-settings.eps31_coeffs),filter_L31_L13_L32_L23_L32_L23_L31_L13,einsum_path_to_use) 
    z_combo = z_combo + ((1-L31_L13_L32_L23_L32_L23_L31_L13[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L31_L13_L32_L23_L32_L23_L31_L13[2]))

    #BEGIN NEGATIVE
    z_combo_minus = np.zeros(settings.length)

    z_combo_minus = z_combo_minus + (settings.s32 + 0.5*(settings.tau32-settings.eps32) + 0.5*(settings.tau31-settings.tau32)) 

    next_term = trim_data((settings.tau23_coeffs-settings.eps23_coeffs) + settings.s23_coeffs,filter_L32,einsum_path_to_use) 
    z_combo_minus = z_combo_minus + ((1-L32[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L32[2]))

    next_term = trim_data(0.5*(settings.tau32_coeffs-settings.eps32_coeffs) + 0.5*(settings.tau32_coeffs-settings.tau31_coeffs) + settings.s31_coeffs + 0.5*(settings.tau31_coeffs-settings.eps31_coeffs),filter_L32_L23,einsum_path_to_use) 
    z_combo_minus = z_combo_minus + ((1-L32_L23[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L32_L23[2]))

    next_term = trim_data((settings.tau13_coeffs-settings.eps13_coeffs) + settings.s13_coeffs,filter_L32_L23_L31,einsum_path_to_use) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L32_L23_L31[2]))

    next_term = trim_data((settings.tau31_coeffs-settings.eps31_coeffs) + settings.s31_coeffs,filter_L32_L23_L31_L13,einsum_path_to_use) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L32_L23_L31_L13[2]))

    next_term = trim_data((settings.tau13_coeffs-settings.eps13_coeffs) + settings.s13_coeffs,filter_L32_L23_L31_L13_L31,einsum_path_to_use) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13_L31[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L32_L23_L31_L13_L31[2]))

    next_term = trim_data(0.5*(settings.tau31_coeffs-settings.eps31_coeffs) + settings.s32_coeffs + 0.5*(settings.tau32_coeffs-settings.eps32_coeffs) + 0.5*(settings.tau31_coeffs-settings.tau32_coeffs),filter_L32_L23_L31_L13_L31_L13,einsum_path_to_use) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13_L31_L13[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L32_L23_L31_L13_L31_L13[2]))

    next_term = trim_data((settings.tau23_coeffs-settings.eps23_coeffs) + settings.s23_coeffs,filter_L32_L23_L31_L13_L31_L13_L32,einsum_path_to_use) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13_L31_L13_L32[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L32_L23_L31_L13_L31_L13_L32[2]))

    next_term = trim_data(0.5*(settings.tau32_coeffs-settings.eps32_coeffs) + 0.5*(settings.tau32_coeffs-settings.tau31_coeffs),filter_L32_L23_L31_L13_L31_L13_L32_L23,einsum_path_to_use) 
    z_combo_minus = z_combo_minus + ((1-L32_L23_L31_L13_L31_L13_L32_L23[1])*(next_term + np.gradient(next_term,1/settings.f_s)*L32_L23_L31_L13_L31_L13_L32_L23[2]))

    z_combo = z_combo - z_combo_minus
    '''
    #np.savetxt('z_combo_full.dat',z_combo)
    plt.plot(s12,label = 's12')
    plt.plot(z_combo,label='z combo')
    plt.plot(window*z_combo,label = 'Kaiser windowed')
    plt.legend()
    plt.show()    
    '''

    
    z_f = np.fft.rfft(settings.window*z_combo,norm='ortho')[settings.indices_f_band]

    return [np.real(z_f),np.imag(z_f)]


