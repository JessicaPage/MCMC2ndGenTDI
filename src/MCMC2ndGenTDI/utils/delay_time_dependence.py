import numpy as np
from .. import settings

def orbital_parameters(semi_major,inclination):


    orbital_freq=np.sqrt(settings.GM_SUN/semi_major**3)

    cos_inclination = np.cos(inclination)
    sin_inclination = np.sin(inclination)

    return orbital_freq,cos_inclination,sin_inclination


def s_c_positions(psi_here,eccentricity,cos_inclination,sin_inclination,semi_major,orbital_freq,omega,arg_per,k):

	lambda_k = omega  + arg_per
	zeta_t = semi_major*(np.cos(psi_here) - eccentricity)
	eta_t = semi_major*np.sqrt(1.0-eccentricity**2)*np.sin(psi_here)
	positions = np.empty((3,settings.length))
	
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
	velocities = np.empty((3,settings.length))

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

    return 2.0*settings.GM_SUN/(settings.c**2)*np.log((mag_pos_j + mag_pos_i + mag_diff)/(mag_pos_j+mag_pos_i-mag_diff))



def psi(m_init1,eccentricity,orbital_freq,k,t):
    m = m_init1 + orbital_freq*(t-settings.t_init)

    psi_return = m + (eccentricity-np.power(eccentricity,3)/8.0)*np.sin(m) + 0.5*eccentricity**2*np.sin(2.0*m)  + 3.0/8*np.power(eccentricity,3)*np.sin(3.0*m)
    for i in np.arange(2):
        error =psi_return - eccentricity * np.sin(psi_return) - m
        psi_return -= error / (1.0 - eccentricity * np.cos(psi_return)) 

    return psi_return

def theta(k):
    return 2.0*np.pi*(k-1)/3.0

def delta_tau(psi_here,m_init1,eccentricity,orbital_freq,semi_major,k,t):
    psi_here_init = psi(m_init1,eccentricity,orbital_freq,k,t_init)

    return -3.0/2.0*(orbital_freq*semi_major/settings.c)**2*(t-settings.t_init) - 2.0*(orbital_freq*semi_major/settings.c)**2*eccentricity/orbital_freq*(np.sin(psi_here)-np.sin(psi_here_init))



def time_dependence(m_init1,semi_major,eccentricity,inclination,omega_init,arg_per):


	delay_in_time = np.empty((6,settings.length))
	if settings.is_tcb==False:
		mprs = np.empty((6,settings.length))

	orbital_freq,cos_inclination,sin_inclination = orbital_parameters(semi_major,inclination)

	for i,k,z in zip(np.arange(6),np.array([2,3,3,1,1,2]),np.array([3,2,1,3,2,1])):
		psi_i = psi(m_init1[z-1],eccentricity[z-1],orbital_freq[z-1],z,settings.tcb_times[z-1])
		psi_j = psi(m_init1[k-1],eccentricity[k-1],orbital_freq[k-1],k,settings.tcb_times[z-1])
		
		position_i = s_c_positions(psi_i,eccentricity[z-1],cos_inclination[z-1],sin_inclination[z-1],semi_major[z-1],orbital_freq[z-1],omega_init[z-1],arg_per[z-1],z)
		position_j = s_c_positions(psi_j,eccentricity[k-1],cos_inclination[k-1],sin_inclination[k-1],semi_major[k-1],orbital_freq[k-1],omega_init[k-1],arg_per[k-1],k)

		Dij = position_i-position_j


		magDij = np.sqrt(Dij[0]**2+Dij[1]**2+Dij[2]**2)


		velocity_j = s_c_velocities(psi_j,eccentricity[k-1],cos_inclination[k-1],sin_inclination[k-1],semi_major[k-1],orbital_freq[k-1],omega_init[k-1],arg_per[k-1],k)

		second_term = np.sum(Dij*velocity_j,axis=0)/(settings.c**2)

		mag_v_j = np.sqrt(velocity_j[0]**2+velocity_j[1]**2+velocity_j[2]**2)
		third_term = magDij/(2.0*np.power(settings.c,3))*(mag_v_j**2 + np.power(np.sum(velocity_j*Dij,axis=0)/magDij,2) -np.sum(s_c_accelerations(position_j,semi_major[k-1],orbital_freq[k-1])*Dij,axis=0))

		delay_in_time[i] = magDij/settings.c + second_term + third_term +  shapiro(position_i,position_j)/settings.c

		if settings.is_tcb==False:

			mprs[i] = delay_in_time[i]+ delta_tau(psi_i,m_init1[z-1],eccentricity[z-1],orbital_freq[z-1],semi_major[z-1],z,tcb_times[z-1]) - delta_tau(psi_j,m_init1[k-1],eccentricity[k-1],orbital_freq[k-1],semi_major[k-1],k,settings.tcb_times[z-1] - delay_in_time[i])
			
	if settings.is_tcb==True:
		return delay_in_time
	else:
		return mprs
