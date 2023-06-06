from __future__ import absolute_import, division, print_function
import sys
#from mpmath import binomial,log,pi,ceil
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import kaiser,kaiser_beta
#from mpmath import *
import time
from scipy.stats import norm
#from astropy import constants as const
import numpy as np
from scipy.stats import linregress


#from scipy.signal import butter,filtfilt
#from scipy.special import gamma,binom,comb
#import h5py   
#import nexusformat.nexus as nx
from lisaconstants import GM_SUN,c,ASTRONOMICAL_YEAR,ASTRONOMICAL_UNIT,SUN_SCHWARZSCHILD_RADIUS,OBLIQUITY
#from orbital import KeplerianElements


#cut_off = int(1e3)    
cut_off = 0 
'''
positions_1 = np.genfromtxt('s_c_positions_1.dat')
positions_2 = np.genfromtxt('s_c_positions_2.dat')
positions_3 = np.genfromtxt('s_c_positions_3.dat')
'''
positions_1 = np.genfromtxt('s_c_positions_hour_1.dat')
positions_2 = np.genfromtxt('s_c_positions_hour_2.dat')
positions_3 = np.genfromtxt('s_c_positions_hour_3.dat')

'''
positions_1 = np.genfromtxt('s_c_positions_9hour_1.dat')
positions_2 = np.genfromtxt('s_c_positions_9hour_2.dat')
positions_3 = np.genfromtxt('s_c_positions_9hour_3.dat')
'''
print('positions_1') 
print(positions_1)

'''
positions = np.array([positions_1,positions_2,positions_3]) 


#obliquity = 84381.406 / (60 * 60) * (2 * np.pi / 360)
obliquity = OBLIQUITY
rotation_matrix = np.array([
	[1, 0, 0],
	[0, np.cos(obliquity), -np.sin(obliquity)],
	[0, np.sin(obliquity), np.cos(obliquity)]
])
ecliptic_positions = np.einsum('ijk,kl->ijl', positions, rotation_matrix)

positions_1 = ecliptic_positions[0]
positions_2 = ecliptic_positions[1]
positions_3 = ecliptic_positions[2]
'''


'''
positions_1=positions_1.T
positions_2=positions_2.T
positions_3=positions_3.T
'''
print('positions_1')    
print(positions_1)   
'''
velocity_1 = np.genfromtxt('s_c_velocity_1.dat').T
velocity_2 = np.genfromtxt('s_c_velocity_2.dat').T
velocity_3 = np.genfromtxt('s_c_velocity_3.dat').T
'''
'''
velocity_1 = np.genfromtxt('s_c_velocity_1.dat')
velocity_2 = np.genfromtxt('s_c_velocity_2.dat')
velocity_3 = np.genfromtxt('s_c_velocity_3.dat')
'''
velocity_1 = np.genfromtxt('s_c_velocity_hour_1.dat')
velocity_2 = np.genfromtxt('s_c_velocity_hour_2.dat')
velocity_3 = np.genfromtxt('s_c_velocity_hour_3.dat')
'''
velocity_1 = np.genfromtxt('s_c_velocity_9hour_1.dat')
velocity_2 = np.genfromtxt('s_c_velocity_9hour_2.dat')
velocity_3 = np.genfromtxt('s_c_velocity_9hour_3.dat')
'''
print('velocity_1')
print(velocity_1)
'''


velocity_1 = np.gradient(positions_1,1/4.0,axis=0)
velocity_2 = np.gradient(positions_2,1/4.0,axis=0)
velocity_3 = np.gradient(positions_3,1/4.0,axis=0)
'''
print('velocity_1')
print(velocity_1)
'''
mag_pos_1 = np.sqrt(positions_1[0]**2+positions_1[1]**2+positions_1[2]**2)     
mag_pos_2 = np.sqrt(positions_2[0]**2+positions_2[1]**2+positions_2[2]**2)     
mag_pos_3 = np.sqrt(positions_3[0]**2+positions_3[1]**2+positions_3[2]**2)     

mag_vel_1 = np.sqrt(velocity_1[0]**2+velocity_1[1]**2+velocity_1[2]**2)     
mag_vel_2 = np.sqrt(velocity_2[0]**2+velocity_2[1]**2+velocity_2[2]**2)     
mag_vel_3 = np.sqrt(velocity_3[0]**2+velocity_3[1]**2+velocity_3[2]**2) 
'''
mag_pos_1 = np.sqrt(positions_1.T[0]**2+positions_1.T[1]**2+positions_1.T[2]**2)     
mag_pos_2 = np.sqrt(positions_2.T[0]**2+positions_2.T[1]**2+positions_2.T[2]**2)     
mag_pos_3 = np.sqrt(positions_3.T[0]**2+positions_3.T[1]**2+positions_3.T[2]**2)     

mag_vel_1 = np.sqrt(velocity_1.T[0]**2+velocity_1.T[1]**2+velocity_1.T[2]**2)     
mag_vel_2 = np.sqrt(velocity_2.T[0]**2+velocity_2.T[1]**2+velocity_2.T[2]**2)     
mag_vel_3 = np.sqrt(velocity_3.T[0]**2+velocity_3.T[1]**2+velocity_3.T[2]**2)        

print('mag_pos_1')
print(mag_pos_1)
#keplerian_elements = orbital.elements.KeplerianElements.from_state_vector(positions_1,velocity_1,ref_epoch=13100.0)
'''
ang_momen_1 = np.cross(positions_1,velocity_1,axisa = 0,axisb=0).T
ang_momen_2 = np.cross(positions_2,velocity_2,axisa = 0,axisb=0).T
ang_momen_3 = np.cross(positions_3,velocity_3,axisa = 0,axisb=0).T
'''
ang_momen_1 = np.cross(positions_1,velocity_1)
ang_momen_2 = np.cross(positions_2,velocity_2)
ang_momen_3 = np.cross(positions_3,velocity_3)

print('ang_momen_1')
print(ang_momen_1)
print('ang_momen_1')
print(ang_momen_1.T)
print('velocity_1')
print(velocity_1)
'''
mag_ang_momen_1 = np.sqrt(ang_momen_1[0]**2+ang_momen_1[1]**2+ang_momen_1[2]**2) 
mag_ang_momen_2 = np.sqrt(ang_momen_2[0]**2+ang_momen_2[1]**2+ang_momen_2[2]**2) 
mag_ang_momen_3 = np.sqrt(ang_momen_3[0]**2+ang_momen_3[1]**2+ang_momen_3[2]**2) 
'''
mag_ang_momen_1 = np.sqrt(ang_momen_1.T[0]**2+ang_momen_1.T[1]**2+ang_momen_1.T[2]**2) 
mag_ang_momen_2 = np.sqrt(ang_momen_2.T[0]**2+ang_momen_2.T[1]**2+ang_momen_2.T[2]**2) 
mag_ang_momen_3 = np.sqrt(ang_momen_3.T[0]**2+ang_momen_3.T[1]**2+ang_momen_3.T[2]**2) 

print('mag_ang_momen_1')
print(mag_ang_momen_1)
'''
ecc_1 = np.cross(velocity_1,ang_momen_1,axisa = 0,axisb=0).T/GM_SUN - positions_1/mag_pos_1
ecc_2 = np.cross(velocity_2,ang_momen_2,axisa = 0,axisb=0).T/GM_SUN - positions_2/mag_pos_2
ecc_3 = np.cross(velocity_3,ang_momen_3,axisa = 0,axisb=0).T/GM_SUN - positions_3/mag_pos_3
'''

print('np.cross(velocity_1,ang_momen_1)/GM_SUN')
print(np.cross(velocity_1,ang_momen_1)/GM_SUN)
print('np.cross(velocity_1,ang_momen_1).T/GM_SUN')
print(np.cross(velocity_1,ang_momen_1).T/GM_SUN)

print('positions_1.T/mag_pos_1')
print(positions_1.T/mag_pos_1)
ecc_1 = np.cross(velocity_1,ang_momen_1).T/GM_SUN - positions_1.T/mag_pos_1
ecc_2 = np.cross(velocity_2,ang_momen_2).T/GM_SUN - positions_2.T/mag_pos_2
ecc_3 = np.cross(velocity_3,ang_momen_3).T/GM_SUN - positions_3.T/mag_pos_3

mag_ecc_1 = np.sqrt(ecc_1[0]**2+ecc_1[1]**2+ecc_1[2]**2)     
mag_ecc_2 = np.sqrt(ecc_2[0]**2+ecc_2[1]**2+ecc_2[2]**2)     
mag_ecc_3 = np.sqrt(ecc_3[0]**2+ecc_3[1]**2+ecc_3[2]**2)     

print('ecc_1')
print(ecc_1)
print('mag_ecc_1)')
print(mag_ecc_1)
print('mean mag_ecc_1)')
print(np.mean(mag_ecc_1))
'''
n_1 = np.array([-1*ang_momen_1[1],ang_momen_1[0],np.zeros(len(ang_momen_1[0]))])
n_2 = np.array([-1*ang_momen_2[1],ang_momen_2[0],np.zeros(len(ang_momen_2[0]))])
n_3 = np.array([-1*ang_momen_3[1],ang_momen_3[0],np.zeros(len(ang_momen_3[0]))])
'''
n_1 = np.array([-1*ang_momen_1.T[1],ang_momen_1.T[0],np.zeros(len(ang_momen_1.T[0]))])
n_2 = np.array([-1*ang_momen_2.T[1],ang_momen_2.T[0],np.zeros(len(ang_momen_2.T[0]))])
n_3 = np.array([-1*ang_momen_3.T[1],ang_momen_3.T[0],np.zeros(len(ang_momen_3.T[0]))])
print('n_1')
print(n_1)
print('n_1')
print(n_1.T)
mag_n_1 = np.sqrt(n_1[0]**2+n_1[1]**2+n_1[2]**2) 
mag_n_2 = np.sqrt(n_2[0]**2+n_2[1]**2+n_2[2]**2)     
mag_n_3 = np.sqrt(n_3[0]**2+n_3[1]**2+n_3[2]**2)     
    
print('positions_1*velocity_1')
print(positions_1*velocity_1)
print('np.sum(positions_1*velocity_1,axis=1)')
print(np.sum(positions_1*velocity_1,axis=1))
neg_1 = np.argwhere(np.sum(positions_1*velocity_1,axis=1)>=0)
print('neg_1')
print(neg_1)
if all(np.sum(positions_1*velocity_1,axis=1)>=0.0):
	true_anomaly_1 = np.arccos(np.sum(ecc_1*positions_1.T,axis=0)/(mag_ecc_1*mag_pos_1))
	print('greater')

else:

	true_anomaly_1 = 2.0*np.pi - np.arccos(np.sum(ecc_1*positions_1.T,axis=0)/(mag_ecc_1*mag_pos_1))
print('np.sum(positions_2*velocity_2,axis=1)')
print(np.sum(positions_2*velocity_2,axis=1))
if all(np.sum(positions_2*velocity_2,axis=1)>=0.0):
	print('greater')
	true_anomaly_2 = np.arccos(np.sum(ecc_2*positions_2.T,axis=0)/(mag_ecc_2*mag_pos_2))
else:

	true_anomaly_2 = 2.0*np.pi - np.arccos(np.sum(ecc_2*positions_2.T,axis=0)/(mag_ecc_2*mag_pos_2))	
print('np.sum(positions_3*velocity_3,axis=1)')
print(np.sum(positions_3*velocity_3,axis=1))
if all(np.sum(positions_3*velocity_3,axis=1)>=0.0):
	print('greater')

	true_anomaly_3 = np.arccos(np.sum(ecc_3*positions_3.T,axis=0)/(mag_ecc_3*mag_pos_3))
else:
	true_anomaly_3 = 2.0*np.pi - np.arccos(np.sum(ecc_3*positions_3.T,axis=0)/(mag_ecc_3*mag_pos_3))

	
inclination_1 = np.arccos(ang_momen_1.T[2]/mag_ang_momen_1)
inclination_2 = np.arccos(ang_momen_2.T[2]/mag_ang_momen_2)
inclination_3 = np.arccos(ang_momen_3.T[2]/mag_ang_momen_3)

ecc_anomaly_1 = 2.0*np.arctan(np.tan(true_anomaly_1/2.0)/(np.sqrt((1.0+mag_ecc_1)/(1.0-mag_ecc_1))))
ecc_anomaly_2 = 2.0*np.arctan(np.tan(true_anomaly_2/2.0)/(np.sqrt((1.0+mag_ecc_2)/(1.0-mag_ecc_2))))
ecc_anomaly_3 = 2.0*np.arctan(np.tan(true_anomaly_3/2.0)/(np.sqrt((1.0+mag_ecc_3)/(1.0-mag_ecc_3))))


print('ecc_anomaly_1')
print(ecc_anomaly_1[cut_off::])
#sys.exit()
print('n_1[1]')
print(n_1[1])
if all(n_1[1] >=0.0):
	raan_1 = np.arccos(n_1[0]/mag_n_1)
else:
	raan_1 = 2.0*np.pi - np.arccos(n_1[0]/mag_n_1)
	
if all(n_2[1] >=0.0):
	raan_2 = np.arccos(n_2[0]/mag_n_2)
else:
	raan_2 = 2.0*np.pi - np.arccos(n_2[0]/mag_n_2)
	
if all(n_3[1] >=0.0):
	raan_3 = np.arccos(n_3[0]/mag_n_3)
else:
	raan_3 = 2.0*np.pi - np.arccos(n_3[0]/mag_n_3)
	
print('ecc_1[2]')	
print(ecc_1)
print(ecc_1[2])
#sys.exit()
if all(ecc_1[2]>=0.0):
	arg_per_1 = np.arccos(np.sum(n_1*ecc_1,axis=0)/(mag_n_1*mag_ecc_1))
else:

	arg_per_1 = 2.0*np.pi - np.arccos(np.sum(n_1*ecc_1,axis=0)/(mag_n_1*mag_ecc_1))
if all(ecc_2[2]>=0.0):
	arg_per_2 = np.arccos(np.sum(n_2*ecc_2,axis=0)/(mag_n_2*mag_ecc_2))
else:

	arg_per_2 = 2.0*np.pi - np.arccos(np.sum(n_2*ecc_2,axis=0)/(mag_n_2*mag_ecc_2))
if all(ecc_3[2]>=0.0):
	arg_per_3 = np.arccos(np.sum(n_3*ecc_3,axis=0)/(mag_n_3*mag_ecc_3))
else:

	arg_per_3 = 2.0*np.pi - np.arccos(np.sum(n_3*ecc_3,axis=0)/(mag_n_3*mag_ecc_3))
	
mean_anomaly_1 = ecc_anomaly_1 - mag_ecc_1*np.sin(ecc_anomaly_1)
mean_anomaly_2 = ecc_anomaly_2 - mag_ecc_2*np.sin(ecc_anomaly_2)
mean_anomaly_3 = ecc_anomaly_3 - mag_ecc_3*np.sin(ecc_anomaly_3)


semi_major_1 = 1.0/(2.0/mag_pos_1 - mag_vel_1**2/GM_SUN)
semi_major_2 = 1.0/(2.0/mag_pos_2 - mag_vel_2**2/GM_SUN)
semi_major_3 = 1.0/(2.0/mag_pos_3 - mag_vel_3**2/GM_SUN)

#np.savetxt('truth_vals_ESA.dat',)

print('mag_ecc_1)')
print(mag_ecc_1)
print('mean mag_ecc_1)')
print(np.mean(mag_ecc_1))

print('inclination_1')
print(inclination_1)
print('inclination_2')
print(inclination_2)
print('inclination_3')
print(inclination_3)
print('arg_per_1')
print(arg_per_1)
print('arg_per_2')
print(arg_per_2)
print('arg_per_3')
print(arg_per_3)
print('mean_anomaly_1')
print(mean_anomaly_1)
print('mean_anomaly_2')
print(mean_anomaly_2)
print('mean_anomaly_3')
print(mean_anomaly_3)
print('semi_major_1')
print(semi_major_1)
print('semi_major_2')
print(semi_major_2)
print('semi_major_3')
print(semi_major_3)
print('raan_1')
print(raan_1)
print('raan_2')
print(raan_2)
print('raan_3')
print(raan_3)
print('mag_ecc_1)')
print(mag_ecc_1)
print('mag_ecc_2)')
print(mag_ecc_2)
print('mag_ecc_3)')
print(mag_ecc_3)



data = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_ESA_orbits_tcb_ltt_orbits_mprs_and_dpprs_to_file_1_hour_4_Hz_NO_AA_filter_NEW.dat',names=True)
times = data['time']
'''
plt.plot(times,semi_major_1)
plt.title(r'$a_{1}$')
plt.show()

plt.plot(times,mag_ecc_1)
plt.title(r'$e_{1}$')
plt.show()

plt.plot(times,inclination_1)
plt.title(r'$i_{1}$')
plt.show()

plt.plot(times,mean_anomaly_1)
plt.title(r'$m_{1}$')
plt.show()

plt.plot(times,raan_1)
plt.title(r'$\Omega_{1}$')
plt.show()

plt.plot(times,arg_per_1)
plt.title(r'$\omega_{1}$')
plt.show()
'''
header = 'semi_major_1 semi_major_2 semi_major_3 mag_ecc_1 mag_ecc_2 mag_ecc_3 inclination_1 inclination_2 inclination_3 mean_anomaly_1 mean_anomaly_2 mean_anomaly_3 raan_1 raan_2 raan_3 arg_per_1 arg_per_2 arg_per_3'
#elements = np.array([semi_major_1[0],semi_major_2[0],semi_major_3[0],mag_ecc_1[0],mag_ecc_2[0],mag_ecc_3[0],inclination_1[0],inclination_2[0],inclination_3[0],mean_anomaly_1[0],mean_anomaly_2[0],mean_anomaly_3[0],raan_1[0],raan_2[0],raan_3[0],arg_per_1[0],arg_per_2[0],arg_per_3[0],2.4764904587217673,2.5892099624633613,2.6093372953421508])
#elements_a = np.array([semi_major_1[cut_off]+2.4764904587217673*np.arange(13100.0,13100.0+len(mean_anomaly_1))[cut_off::]/4.0,semi_major_2[cut_off]+2.5892099624633613*np.arange(13100.0,13100.0+len(mean_anomaly_1))[cut_off::]/4.0,semi_major_3[cut_off]+2.4764904587217673*np.arange(13100.0,13100.0+len(mean_anomaly_1))[cut_off::]/4.0])
#elements = np.array([semi_major_1[cut_off],semi_major_2[cut_off],semi_major_3[cut_off],mag_ecc_1[cut_off],mag_ecc_2[cut_off],mag_ecc_3[cut_off],inclination_1[cut_off],inclination_2[cut_off],inclination_3[cut_off],mean_anomaly_1[0],mean_anomaly_2[0],mean_anomaly_3[0],raan_1[cut_off],raan_2[cut_off],raan_3[cut_off],arg_per_1[cut_off],arg_per_2[cut_off],arg_per_3[cut_off],2.4764904587217673,2.5892099624633613,2.6093372953421508])
#elements = np.array([np.mean(semi_major_1),np.mean(semi_major_2),np.mean(semi_major_3),np.mean(mag_ecc_1),np.mean(mag_ecc_2),np.mean(mag_ecc_3),np.mean(inclination_1),np.mean(inclination_2),np.mean(inclination_3),mean_anomaly_1[cut_off],mean_anomaly_2[cut_off],mean_anomaly_3[cut_off],np.mean(raan_1),np.mean(raan_2),np.mean(raan_3),np.mean(arg_per_1),np.mean(arg_per_2),np.mean(arg_per_3)])
elements = np.array([semi_major_1[cut_off],semi_major_2[cut_off],semi_major_3[cut_off],mag_ecc_1[cut_off],mag_ecc_2[cut_off],mag_ecc_3[cut_off],inclination_1[cut_off],inclination_2[cut_off],inclination_3[cut_off],mean_anomaly_1[cut_off],mean_anomaly_2[cut_off],mean_anomaly_3[cut_off],raan_1[cut_off],raan_2[cut_off],raan_3[cut_off],arg_per_1[cut_off],arg_per_2[cut_off],arg_per_3[cut_off]])
print('elements')
print(elements.T)
#plt.plot(semi_major_1)
#plt.show()


print('times')
print(times)
print(linregress(times, semi_major_2))
print(linregress(times, semi_major_1))
print(linregress(times, semi_major_1).slope)
print(linregress(times, semi_major_3))
print(linregress(times, mag_ecc_1))
print(linregress(times, mag_ecc_2))
print(linregress(times, mag_ecc_3))
np.savetxt('elements_from_Cartesian_hour.dat',elements.T,header = header)

np.savetxt('semi_major.dat',np.array([semi_major_1[cut_off::],semi_major_2[cut_off::],semi_major_3[cut_off::]]).T)
#print('keplerian_elements')
#print(keplerian_elements)