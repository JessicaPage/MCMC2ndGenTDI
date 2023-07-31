import numpy as np
from .. import settings
from lisaconstants import GM_SUN,c,ASTRONOMICAL_UNIT# Constant values set below to avoid importing lisaconstant


def theta(k):
    return 2.0*np.pi*(k-1)/3


# In[13]:


#@jit(nopython=True)
#@jit(numba.types.float64[:](numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64,numba.types.float64[:]),nopython=True)

def psi(m_init1,eccentricity,orbital_freq,k,t):
    m = m_init1 + orbital_freq*(t-settings.t_init) - theta(k)
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
    alpha = settings.L_arm/(2.0*semi_major)
    nu = np.pi/3 + settings.delta*alpha
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
	#lambda_k = settings.Omega_1 + theta(k) + arg_per
	omega = settings.Omega_1+theta(k)
	#print('k')
	#print(eccentricity)
	#lambda_k = omega  + arg_per
	zeta_t = semi_major*(np.cos(psi_here) - eccentricity)
	eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.sin(psi_here)
	#psi_here = psi(m_init1,eccentricity,orbital_freq,k,t)
	positions = np.empty((3,settings.length))

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
	omega = settings.Omega_1+theta(k)
	zeta_t = semi_major*(np.cos(psi_here) - eccentricity)
	d_zeta_t = -1*semi_major*np.sin(psi_here)*psi_dot
	eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.sin(psi_here)
	d_eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.cos(psi_here)*psi_dot
	#psi_here = psi(m_init1,eccentricity,orbital_freq,k,t)
	velocities = np.empty((3,settings.length))

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
    psi_here_init = psi(m_init1,eccentricity,orbital_freq,k,settings.t_init)

    return -3.0/2.0*(orbital_freq*semi_major/c)**2*(t-settings.t_init) - 2*(orbital_freq*semi_major/c)**2*eccentricity/orbital_freq*(np.sin(psi_here)-np.sin(psi_here_init))


# In[20]:


#@jit(nopython=True,parallel=True)
#@numba.cfunc("double(double)")
def time_dependence(m_init1,semi_major,arg_per):


    delay_in_time = np.empty((6,settings.length))
    if settings.is_tcb==False:
        mprs = np.empty((6,settings.length))

    alpha,nu,eccentricity,orbital_freq,cos_inclination,sin_inclination = orbital_parameters(semi_major)
    for i,k,z in zip(np.arange(6),np.array([2,3,3,1,1,2]),np.array([3,2,1,3,2,1])):
        psi_i = psi(m_init1,eccentricity,orbital_freq,z,settings.tcb_times[z-1])
        psi_j = psi(m_init1,eccentricity,orbital_freq,k,settings.tcb_times[z-1])
        position_i = s_c_positions(psi_i,eccentricity,cos_inclination,sin_inclination,semi_major,settings.Omega_1,arg_per,z)
        position_j = s_c_positions(psi_j,eccentricity,cos_inclination,sin_inclination,semi_major,settings.Omega_1,arg_per,k)
        Dij = position_i-position_j


        magDij = np.sqrt(Dij[0]**2+Dij[1]**2+Dij[2]**2)

        velocity_j = s_c_velocities(psi_j,eccentricity,cos_inclination,sin_inclination,semi_major,orbital_freq,settings.Omega_1,arg_per,k)
        second_term = np.sum(Dij*velocity_j,axis=0)/(c**2)

        #third_term = magDij/(2*np.power(c,3))*(np.linalg.norm(velocity_j,axis=0)**2 + np.power(np.sum(velocity_j*Dij,axis=0)/magDij,2) -np.sum(s_c_accelerations(position_j,semi_major,orbital_freq)*Dij,axis=0))
        third_term = magDij/(2*np.power(c,3))*(np.sqrt(velocity_j[0]**2+velocity_j[1]**2+velocity_j[2]**2)**2 + np.power(np.sum(velocity_j*Dij,axis=0)/magDij,2) -np.sum(s_c_accelerations(position_j,semi_major,orbital_freq)*Dij,axis=0))

        delay_in_time[i] = magDij/c + second_term + third_term +  shapiro(position_i,position_j)/c
        if settings.is_tcb==False:
            mprs[i] = delay_in_time[i]+ delta_tau(psi_i,m_init1,eccentricity,orbital_freq,semi_major,z,settings.tcb_times[z-1]) - delta_tau(psi_j,m_init1,eccentricity,orbital_freq,semi_major,k,settings.tcb_times[z-1] - delay_in_time[i])


    if settings.is_tcb==True:
        return delay_in_time
    else:
        return mprs
