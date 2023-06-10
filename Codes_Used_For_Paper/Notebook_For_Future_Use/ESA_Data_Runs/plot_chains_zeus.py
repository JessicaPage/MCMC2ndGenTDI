import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import corner 
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import matplotlib.colors as color 
from chainconsumer import ChainConsumer
#from matplotlib import rcParams
from matplotlib import rc
import h5py
from lisaconstants import GM_SUN,c,ASTRONOMICAL_YEAR,ASTRONOMICAL_UNIT
import zeus
import matplotlib as mpl

# # CREATING FILTERS
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


removed =0


dir = './zeus_no_light_mode/'
with h5py.File('saved_chains_zeus_light_mode_false.h5', "r") as hf:
    samples = np.copy(hf['samples'])
    logprob_samples = np.copy(hf['logprob'])
'''    
dir = './zeus/'
with h5py.File('saved_chains_zeus.h5', "r") as hf:
    samples = np.copy(hf['samples'])
    logprob_samples = np.copy(hf['logprob'])
'''   
hf.close()

print('samples')
print(samples)
print(samples.shape)

reshaped_samples = samples.reshape(-1, samples.shape[-1])
print('samples flattened')
print(reshaped_samples)
print(reshaped_samples.shape)


tau = zeus.AutoCorrTime(samples)




print('tau')
print(tau)

'''
print('samples')
print(samples.T)
print(samples[:,:,0].flatten())
'''
print('logprob')
#print(logprob_samples)
print(logprob_samples[-1])
test_name=' '


#sys.exit()
semi_major_1 = samples[:,:,0].flatten()/1.0e6


length = len(semi_major_1)
removed =int(0.5*length)
thin = 1



semi_major_1 = samples[:,:,0].flatten()[removed::thin]/1.0e6
semi_major_2 = samples[:,:,1].flatten()[removed::thin]/1.0e6
semi_major_3 = samples[:,:,2].flatten()[removed::thin]/1.0e6

eccentricity_1 = samples[:,:,3].flatten()[removed::thin]
eccentricity_2 = samples[:,:,4].flatten()[removed::thin]
eccentricity_3 = samples[:,:,5].flatten()[removed::thin]

inclination_1 = samples[:,:,6].flatten()[removed::thin]
inclination_2 = samples[:,:,7].flatten()[removed::thin]
inclination_3 = samples[:,:,8].flatten()[removed::thin]

minit1 =  samples[:,:,9].flatten()[removed::thin]
minit2 =  samples[:,:,10].flatten()[removed::thin]
minit3 =  samples[:,:,11].flatten()[removed::thin]


omega_1 = samples[:,:,12].flatten()[removed::thin]
omega_2 = samples[:,:,13].flatten()[removed::thin]
omega_3 = samples[:,:,14].flatten()[removed::thin]

arg_per_1 = samples[:,:,15].flatten()[removed::thin]
arg_per_2 = samples[:,:,16].flatten()[removed::thin]
arg_per_3 = samples[:,:,17].flatten()[removed::thin]
'''

semi_major_1 = samples[:,37,0].flatten()[removed::]/1.0e6
semi_major_2 = samples[:,37,1].flatten()[removed::]/1.0e6
semi_major_3 = samples[:,37,2].flatten()[removed::]/1.0e6

eccentricity_1 = samples[:,37,3].flatten()[removed::]
eccentricity_2 = samples[:,37,4].flatten()[removed::]
eccentricity_3 = samples[:,37,5].flatten()[removed::]

inclination_1 = samples[:,37,6].flatten()[removed::]
inclination_2 = samples[:,37,7].flatten()[removed::]
inclination_3 = samples[:,37,8].flatten()[removed::]

minit1 =  samples[:,37,9].flatten()[removed::]
minit2 =  samples[:,37,10].flatten()[removed::]
minit3 =  samples[:,37,11].flatten()[removed::]


omega_1 = samples[:,37,12].flatten()[removed::]
omega_2 = samples[:,37,13].flatten()[removed::]
omega_3 = samples[:,37,14].flatten()[removed::]

arg_per_1 = samples[:,37,15].flatten()[removed::]
arg_per_2 = samples[:,37,16].flatten()[removed::]
arg_per_3 = samples[:,37,17].flatten()[removed::]

#sys.exit()
#likelihood = data['likelihood']
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
#removed = int(0.25*len(likelihood))
#removed=0
#likelihood = data['likelihood'][removed::]
minit1 = data[9]
minit2 = data[10]
minit3 = data[11]


semi_major_1 = data[0]
semi_major_2 = data[1]
semi_major_3 = data[2]

omega_1 = data[12]
omega_2 = data[13]
omega_3 = data[14]

arg_per_1 = data[15]
arg_per_2 = data[16]
arg_per_3 = data[17]

eccentricity_1 = data[3]
eccentricity_2 = data[4]
eccentricity_3 = data[5]

inclination_1 = data[6]
inclination_2 = data[7]
inclination_3 = data[8]
'''

#chi_2 = data['chi_2_here'][removed::]



#max_indice = np.argmax(likelihood)


#........................truth values................................
#........................truth values................................
#........................truth values................................

elements_data = np.genfromtxt('elements_from_Cartesian_hour.dat')

semi_major_0=np.array([elements_data[0],elements_data[1],elements_data[2]])
eccentricity_0 = np.array([elements_data[3],elements_data[4],elements_data[5]])
inclination_0 = np.array([elements_data[6],elements_data[7],elements_data[8]])
m_init1_0 =np.array([elements_data[9],elements_data[10],elements_data[11]])
omega_init_0 = np.array([elements_data[12],elements_data[13],elements_data[14]])
arg_per_0 = np.array([elements_data[15],elements_data[16],elements_data[17]])


length = len(semi_major_1)

print('length')
print(length)


cut_off = int(1e3)

minit1_truth = m_init1_0[0]
minit2_truth = m_init1_0[1]
minit3_truth = m_init1_0[2]

semi_major_1_truth = semi_major_0[0]/1.0e6
semi_major_2_truth = semi_major_0[1]/1.0e6
semi_major_3_truth = semi_major_0[2]/1.0e6

eccentricity_1_truth = eccentricity_0[0]
eccentricity_2_truth = eccentricity_0[1]
eccentricity_3_truth = eccentricity_0[2]

inclination_1_truth = inclination_0[0]
inclination_2_truth = inclination_0[1]
inclination_3_truth = inclination_0[2]


omega_1_truth = omega_init_0[0]
omega_2_truth = omega_init_0[1]
omega_3_truth = omega_init_0[2]

#eccentricity_truth = 0.004815434522687179
arg_per_1_truth = arg_per_0[0]
arg_per_2_truth = arg_per_0[1]
arg_per_3_truth = arg_per_0[2]

#fig, axes = zeus.cornerplot(reshaped_samples[removed::thin], size=(16,16))


rc('font', **{'family': 'serif'})
rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (8.6,8.6)
plt.rcParams['font.size'] = 15
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{lmodern}'    # latin modern, recommended to replace computer modern sans serif
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath']  # <- tricky! -- gotta actually tell tex to use! 

#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------

data_plot = np.array([minit1,semi_major_1,omega_1,arg_per_1,eccentricity_1,inclination_1,minit2,semi_major_2,omega_2,arg_per_2,eccentricity_2,inclination_2]).T
c = ChainConsumer()
#parameters=[ r"$L_{3}$", r"$L_{2}$",r"$L_{1}$", r"$L^{'}_{3}$",r"$L^{'}_{2}$",r"$L^{'}_{1}$",r"$\dot{L_{3}}$",r"$\dot{L_{2}}$",r"$\dot{L_{1}}$"]
#parameters=[r"$m_{\mathrm{init}}_{1}}$", r"$\lambda_{1}$",r"$e$"]
#parameters=["m", "lambda"]
#parameters=["m", "lambda","eccentricity"]
#parameters=[r"$m_{1_{0}}$",r"$a_{1}$",r"$\Omega_{1}$",r"$\omega_{1}$","e1","i1"]
parameters=[r"$M_{1_{0}} (\mathrm{ rad})$",r"$a_{1}(\mathrm{ km})$",r"$\Omega_{1}(\mathrm{ rad})$",r"$\omega_{1}(\mathrm{ rad})$",r"$e_{1}$",r"$\iota_{1}(\mathrm{ rad})$",r"$M_{2_{0}} (\mathrm{ rad})$",r"$a_{2}(\mathrm{ km})$",r"$\Omega_{2}(\mathrm{ rad})$",r"$\omega_{2}(\mathrm{ rad})$",r"$e_{2}$",r"$\iota_{2}(\mathrm{ rad})$"]

c.add_chain(data_plot, parameters=parameters,color=color.to_hex('orange'),name=r'Spacecraft \# 1,\# 2')
c.configure(spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=0.5,label_font_size=4,tick_font_size=4,summary=False,legend_kwargs={'fontsize':10})
c.configure_truth(color='k')
c.plotter.plot(chains=[r'Spacecraft \# 1,\# 2'],figsize=(8.6,8.6),filename=dir+'corner_plot_sc_1_2.png', truth=[minit1_truth,semi_major_1_truth,omega_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth,minit2_truth,semi_major_2_truth,omega_2_truth,arg_per_2_truth,eccentricity_2_truth,inclination_2_truth],legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_sc_1.png', truth=[minit1_truth,semi_major_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()

data_plot = np.array([minit1,semi_major_1,omega_1,arg_per_1,eccentricity_1,inclination_1,minit3,semi_major_3,omega_3,arg_per_3,eccentricity_3,inclination_3]).T
c = ChainConsumer()
#parameters=[ r"$L_{3}$", r"$L_{2}$",r"$L_{1}$", r"$L^{'}_{3}$",r"$L^{'}_{2}$",r"$L^{'}_{1}$",r"$\dot{L_{3}}$",r"$\dot{L_{2}}$",r"$\dot{L_{1}}$"]
#parameters=[r"$m_{\mathrm{init}}_{1}}$", r"$\lambda_{1}$",r"$e$"]
#parameters=["m", "lambda"]
#parameters=["m", "lambda","eccentricity"]
#parameters=[r"$m_{1_{0}}$",r"$a_{1}$",r"$\Omega_{1}$",r"$\omega_{1}$","e1","i1"]
parameters=[r"$M_{1_{0}} (\mathrm{ rad})$",r"$a_{1}(\mathrm{ km})$",r"$\Omega_{1}(\mathrm{ rad})$",r"$\omega_{1}(\mathrm{ rad})$",r"$e_{1}$",r"$\iota_{1}(\mathrm{ rad})$",r"$M_{3_{0}}$",r"$a_{3}$",r"$\Omega_{3}$",r"$\omega_{3}$",r"$e_{3}$",r"$\iota_{3}$"]

c.add_chain(data_plot, parameters=parameters,color=color.to_hex('red'),name=r'Spacecraft \# 1,\# 3')
c.configure(spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=0.5,label_font_size=4,tick_font_size=4,summary=False,legend_kwargs={'fontsize':10})
c.configure_truth(color='k')
c.plotter.plot(chains=[r'Spacecraft \# 1,\# 3'],figsize=(8.6,8.6),filename=dir+'corner_plot_sc_1_3.png', truth=[minit1_truth,semi_major_1_truth,omega_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth,minit3_truth,semi_major_3_truth,omega_3_truth,arg_per_3_truth,eccentricity_3_truth,inclination_3_truth],legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_sc_1.png', truth=[minit1_truth,semi_major_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()

data_plot = np.array([minit2,semi_major_2,omega_2,arg_per_2,eccentricity_2,inclination_2,minit3,semi_major_3,omega_3,arg_per_3,eccentricity_3,inclination_3]).T
c = ChainConsumer()
#parameters=[ r"$L_{3}$", r"$L_{2}$",r"$L_{1}$", r"$L^{'}_{3}$",r"$L^{'}_{2}$",r"$L^{'}_{1}$",r"$\dot{L_{3}}$",r"$\dot{L_{2}}$",r"$\dot{L_{1}}$"]
#parameters=[r"$m_{\mathrm{init}}_{1}}$", r"$\lambda_{1}$",r"$e$"]
#parameters=["m", "lambda"]
#parameters=["m", "lambda","eccentricity"]
#parameters=[r"$m_{1_{0}}$",r"$a_{1}$",r"$\Omega_{1}$",r"$\omega_{1}$","e1","i1"]
parameters=[r"$M_{2_{0}} (\mathrm{ rad})$",r"$a_{2}(\mathrm{ km})$",r"$\Omega_{2}(\mathrm{ rad})$",r"$\omega_{2}(\mathrm{ rad})$",r"$e_{2}$",r"$\iota_{2}(\mathrm{ rad})$",r"$M_{3_{0}}$",r"$a_{3}$",r"$\Omega_{3}$",r"$\omega_{3}$",r"$e_{3}$",r"$\iota_{3}$"]

c.add_chain(data_plot, parameters=parameters,color=color.to_hex('purple'),name=r'Spacecraft \# 2,\# 3')
c.configure(spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=0.5,label_font_size=4,tick_font_size=4,summary=False,legend_kwargs={'fontsize':10})
c.configure_truth(color='k')
c.plotter.plot(chains=[r'Spacecraft \# 2,\# 3'],figsize=(8.6,8.6),filename=dir+'corner_plot_sc_2_3.png', truth=[minit2_truth,semi_major_2_truth,omega_2_truth,arg_per_2_truth,eccentricity_2_truth,inclination_2_truth,minit3_truth,semi_major_3_truth,omega_3_truth,arg_per_3_truth,eccentricity_3_truth,inclination_3_truth],legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_sc_1.png', truth=[minit1_truth,semi_major_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()

#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------


data_plot = np.array([minit1,semi_major_1,omega_1,arg_per_1,eccentricity_1,inclination_1]).T
c = ChainConsumer()
#parameters=[ r"$L_{3}$", r"$L_{2}$",r"$L_{1}$", r"$L^{'}_{3}$",r"$L^{'}_{2}$",r"$L^{'}_{1}$",r"$\dot{L_{3}}$",r"$\dot{L_{2}}$",r"$\dot{L_{1}}$"]
#parameters=[r"$m_{\mathrm{init}}_{1}}$", r"$\lambda_{1}$",r"$e$"]
#parameters=["m", "lambda"]
#parameters=["m", "lambda","eccentricity"]
#parameters=[r"$m_{1_{0}}$",r"$a_{1}$",r"$\Omega_{1}$",r"$\omega_{1}$","e1","i1"]
parameters=[r"$M_{1_{0}} (\mathrm{rad})$",r"$a_{1}(\mathrm{Mm})$",r"$\Omega_{1} (\mathrm{rad})$",r"$\omega_{1} (\mathrm{rad})$",r"$e_{1}$",r"$\iota_{1}(\mathrm{rad})$"]

c.add_chain(data_plot, parameters=parameters,color=color.to_hex('orange'),name=r'Spacecraft \# 1')
#c.configure(summary_area=0.95,spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure(spacing=0.0,kde=False,smooth=3,cloud=True,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure_truth(color='k')
c.plotter.plot(chains=[r'Spacecraft \# 1'],figsize=(8.6,8.6),filename=dir+'corner_plot_sc_1.png', truth=[minit1_truth,semi_major_1_truth,omega_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth],legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_sc_1.png', truth=[minit1_truth,semi_major_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
c.plotter.plot_walks(filename=dir+'corner_plot_sc_1_WALKS.png')
plt.close()




data_plot = np.array([minit2,semi_major_2,omega_2,arg_per_2,eccentricity_2,inclination_2]).T

#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------

c = ChainConsumer()
#parameters=[ r"$L_{3}$", r"$L_{2}$",r"$L_{1}$", r"$L^{'}_{3}$",r"$L^{'}_{2}$",r"$L^{'}_{1}$",r"$\dot{L_{3}}$",r"$\dot{L_{2}}$",r"$\dot{L_{1}}$"]
#parameters=[r"$m_{\mathrm{init}}_{1}}$", r"$\lambda_{1}$",r"$e$"]
#parameters=["m", "lambda"]
#parameters=["m", "lambda","eccentricity"]
#parameters=[r"$m_{1_{0}}$",r"$a_{1}$",r"$\Omega_{1}$",r"$\omega_{1}$","e1","i1"]
parameters=[r"$M_{2_{0}} (\mathrm{rad})$",r"$a_{2}(\mathrm{Mm})$",r"$\Omega_{2} (\mathrm{rad})$",r"$\omega_{2} (\mathrm{rad})$",r"$e_{2}$",r"$\iota_{2}(\mathrm{rad})$"]

c.add_chain(data_plot, parameters=parameters,color=color.to_hex('purple'),name=r'Spacecraft \# 2')
c.configure(spacing=0.0,kde=False,smooth=3,cloud=True,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure_truth(color='k')
c.plotter.plot(chains=[r'Spacecraft \# 2'],figsize=(8.6,8.6),filename=dir+'corner_plot_sc_2.png', truth=[minit2_truth,semi_major_2_truth,omega_2_truth,arg_per_2_truth,eccentricity_2_truth,inclination_2_truth],legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_sc_1.png', truth=[minit1_truth,semi_major_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
c.plotter.plot_walks(filename=dir+'corner_plot_sc_2_WALKS.png')
plt.close()


data_plot = np.array([minit3,semi_major_3,omega_3,arg_per_3,eccentricity_3,inclination_3]).T


#data_plot_31 = np.array([L_3_here_31,L_2_here_31,L_1_here_31,L_3_p_here_31,L_2_p_here_31,L_1_p_here_31]).T



#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------
c = ChainConsumer()
#parameters=[ r"$L_{3}$", r"$L_{2}$",r"$L_{1}$", r"$L^{'}_{3}$",r"$L^{'}_{2}$",r"$L^{'}_{1}$",r"$\dot{L_{3}}$",r"$\dot{L_{2}}$",r"$\dot{L_{1}}$"]
#parameters=[r"$m_{\mathrm{init}}_{1}}$", r"$\lambda_{1}$",r"$e$"]
#parameters=["m", "lambda"]
#parameters=["m", "lambda","eccentricity"]
#parameters=[r"$m_{1_{0}}$",r"$a_{1}$",r"$\Omega_{1}$",r"$\omega_{1}$","e1","i1"]
parameters=[r"$M_{3_{0}} (\mathrm{rad})$",r"$a_{3}(\mathrm{Mm})$",r"$\Omega_{3} (\mathrm{rad})$",r"$\omega_{3} (\mathrm{rad})$",r"$e_{3}$",r"$\iota_{3}(\mathrm{rad})$"]

c.add_chain(data_plot, parameters=parameters,color=color.to_hex('green'),name=r'Spacecraft \# 3')
c.configure(spacing=0.0,kde=False,smooth=3,cloud=True,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure_truth(color='k')
c.plotter.plot(chains=[r'Spacecraft \# 3'],figsize=(8.6,8.6),filename=dir+'corner_plot_sc_3.png', truth=[minit3_truth,semi_major_3_truth,omega_3_truth,arg_per_3_truth,eccentricity_3_truth,inclination_3_truth],legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_sc_1.png', truth=[minit1_truth,semi_major_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
c.plotter.plot_walks(filename=dir+'corner_plot_sc_3_WALKS.png')
plt.close()




'''

data_plot =  np.array([minit1,minit2,minit3]).T

c = ChainConsumer()
parameters=[r"$m_{1_{0}}$",r"$m_{2_{0}}$",r"$m_{3_{0}}$"]
c.add_chain(data_plot, parameters=parameters,color=color.to_hex('blue'),name=test_name)
c.configure(summary_area=0.95,spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure_truth(color='k')
c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_minit.png', truth=[minit1_truth,minit2_truth,minit3_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()

data_plot =  np.array([semi_major_1,semi_major_2,semi_major_3]).T

c = ChainConsumer()
parameters=[r"$a_{1}$",r"$a_{2}$",r"$a_{3}$"]
c.add_chain(data_plot, parameters=parameters,color=color.to_hex('blue'),name=test_name)
c.configure(summary_area=0.95,spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure_truth(color='k')
c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_semi_major.png', truth=[semi_major_1_truth,semi_major_2_truth,semi_major_3_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()

data_plot =  np.array([eccentricity_1,eccentricity_2,eccentricity_3]).T

c = ChainConsumer()
parameters=[r"$e_{1}$",r"$e_{2}$",r"$e_{3}$"]
c.add_chain(data_plot, parameters=parameters,color=color.to_hex('blue'),name=test_name)
#c.configure(sigmas=[0,1.645],spacing=1.0,kde=False,smooth=1,shade=False)
c.configure(summary_area=0.95,spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})

c.configure_truth(color='k')
c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_eccentricity.png', truth=[eccentricity_1_truth,eccentricity_2_truth,eccentricity_3_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()


data_plot =  np.array([inclination_1,inclination_2,inclination_3]).T

c = ChainConsumer()
parameters=[r"$i_{1}$",r"$i_{2}$",r"$i_{3}$"]
c.add_chain(data_plot, parameters=parameters,color=color.to_hex('blue'),name=test_name)
#c.configure(sigmas=[0,1.645],spacing=1.0,kde=False,smooth=1,shade=False)
c.configure(summary_area=0.95,spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure_truth(color='k')
c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_inclination.png', truth=[inclination_1_truth,inclination_2_truth,inclination_3_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()

data_plot =  np.array([omega_1,omega_2,omega_3]).T

c = ChainConsumer()
parameters=[r"$\Omega_{1}$",r"$\Omega_{2}$",r"$\Omega_{3}$"]
c.add_chain(data_plot, parameters=parameters,color=color.to_hex('blue'),name=test_name)
#c.configure(sigmas=[0,1.645],spacing=1.0,kde=False,smooth=1,shade=False)
c.configure(summary_area=0.95,spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure_truth(color='k')
c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_omega.png', truth=[omega_1_truth,omega_2_truth,omega_3_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()

data_plot =  np.array([arg_per_1,arg_per_2,arg_per_3]).T

c = ChainConsumer()
parameters=[r"$\omega_{1}$",r"$\omega_{2}$",r"$\omega_{3}$"]
c.add_chain(data_plot, parameters=parameters,color=color.to_hex('blue'),name=test_name)
#c.configure(sigmas=[0,1.645],spacing=1.0,kde=False,smooth=1,shade=False)
c.configure(summary_area=0.95,spacing=0.0,cloud=True,kde=False,smooth=1,shade=True,usetex=True,linewidths=1.5,label_font_size=8,tick_font_size=8,summary=True,legend_kwargs={'fontsize':15})
c.configure_truth(color='k')
c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_arg_per.png', truth=[arg_per_1_truth,arg_per_2_truth,arg_per_3_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()

'''


'''
plt.plot(semi_major_1)
plt.ylabel('a1')
plt.xlabel('iteration')
plt.show()



plt.plot(minit1)
plt.ylabel('m1')
plt.xlabel('iteration')
plt.show()

plt.plot(eccentricity_1)
plt.ylabel('e1')
plt.xlabel('iteration')
plt.show()

plt.plot(arg_per_1)
plt.ylabel('w1')
plt.xlabel('iteration')
plt.show()

plt.plot(omega_1)
plt.ylabel('Omega1')
plt.xlabel('iteration')
plt.show()


plt.plot(inclination_1)
plt.ylabel('i3')
plt.xlabel('iteration')
plt.show()
'''


#----------------------------------------------------------------------------
#chi^2 histograms
#----------------------------------------------------------------------------
#sixteen_old = np.quantile(chi_2_old,0.16)
sixteen_new = np.quantile(chi_2_new,0.16)

#eightyfour_old =  np.quantile(chi_2_old,0.84)
eightyfour_new =  np.quantile(chi_2_new,0.84)

'''
plt.hist(chi_2_new,label='over-sampled data generation')
plt.hist(chi_2_old,label='interpolation-only data generation')
plt.axvline(sixteen_old, color='k', linestyle='dashed')
plt.axvline(sixteen_new, color='k', linestyle='dashed')
plt.axvline(eightyfour_old, color='k', linestyle='dashed')
plt.axvline(eightyfour_new, color='k', linestyle='dashed')
plt.legend()
plt.title(r'$\Sigma \chi^{2}$')
plt.show()
'''



'''
plt.plot(semi_major_2)
plt.ylabel('a2')
plt.xlabel('iteration')
plt.show()


plt.plot(semi_major_3)
plt.ylabel('a3')
plt.xlabel('iteration')
plt.show()
'''

'''
gs = fig.axes[9].get_gridspec()
#gs = fig.axes.get_gridspec()


# remove the underlying axes
for ax in fig.axes[10:12]:
#for ax in fig.axes[1:, -1]:
    ax.remove()
axbig = fig.add_subplot(gs[10:12])
#axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5), xycoords='axes fraction', va='center')

#plt.sca(axbig)
'''




'''
plt.hist(chi_2_new,label='Oversampled Data Generation',color='green',histtype='step',density=True)
plt.hist(chi_2_old,label='Interpolated-only data',color='blue',histtype='step',density=True)
plt.axvline(sixteen_old, color='blue', linestyle='dashed')
plt.axvline(sixteen_new, color='green', linestyle='dashed')
plt.axvline(eightyfour_old, color='blue', linestyle='dashed')
plt.axvline(eightyfour_new, color='green', linestyle='dashed')
#plt.xlabel()
#plt.legend()
plt.xlabel(r'$\sum\limits^{f_{\mathrm{max}}}_{i=f_{\mathrm{min}}} \chi^{2}_{i}$',fontsize=12)
plt.xticks(fontsize=10)
#plt.yticks(fontsize=10)
plt.savefig('data_generation_combined.png')
plt.show()
///

'''

'''
plt.hist(likelihood_new,label='over-sampled data generation')
plt.hist(likelihood_old,label='interpolation-only data generation')
plt.legend()
plt.title(r'$\log{\mathcal{L}}$')
plt.show()
'''
'''

#90% credible interval
fig = corner.corner(data_plot,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='green')
corner.corner(data_plot_31,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.05, 0.95],show_titles=False, truths=[0.0,0.0,0.0,0.0,0.0,0.0],truth_color='k',title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='blue',fig=fig)
plt.legend(handles=[blue_line,green_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4)
plt.savefig(dir+'corner_plot_compare_data_generation_90.png')
plt.show()
'''

'''
corner.corner(data_plot_true,labels=[ r"$L_{3}-L_{3_{True}}$", 
		r"$L_{2}-L_{2_{True}}$",r"$L_{1}-L_{1_{True}}$", r"$L^{'}_{3}-L^{'}_{3_{True}}$",r"$L^{'}_{2}-L^{'}_{2_{True}}$",r"$L^{'}_{1}-L^{'}_{1_{True}}$"],
				label_kwargs=dict(fontsize=12),quantiles=[0.16, 0.84],show_titles=False, title_fmt='.2e',title_kwargs={"fontsize": 12},
						use_math_text=True,color='purple',fig=fig)
'''


'''
plt.hist(L_3_here,bins=20,density=True,label='X Channel')
plt.hist(L_3_here_31,bins=20,density=True,label='XYZ Channels',alpha=0.5)
plt.xlabel(r"$L_{3}-L_{3_{True}}$")
#plt.title('X Channel Only')
plt.legend()
plt.show()

plt.hist(L_2_here,bins=20,density=True,label='X Channel')
plt.hist(L_2_here_31,bins=20,density=True,label='XYZ Channels',alpha=0.5)
plt.xlabel(r"$L_{2}-L_{2_{True}}$")
#plt.title('X Channel Only')
plt.legend()
plt.show()
'''



'''
data_estimation = np.array([L_3,L_2,L_3_p,L_2_p])
print('likelihood')
print(likelihood)
#data_plot = np.vstack([likelihood,L_3,L_2,L_3_p,L_2_p])
print('data_plot')
print(data_plot)
print('data_estimation')
print(data_estimation)
cc_L_3_L_3_p = np.corrcoef(L_3,L_3_p)
cc_L_3_L_2_p = np.corrcoef(L_3,L_2_p)
cc_L_3_L_2 = np.corrcoef(L_3,L_2)
cc_L_3_p_L_2 = np.corrcoef(L_3_p,L_2)
cc_L_3_p_L_2_p = np.corrcoef(L_3_p,L_2_p)
cc_L_2_p_L_2 = np.corrcoef(L_2_p,L_2)
print('cc_L_3_L_3_p')
print(cc_L_3_L_3_p)
print('cc_L_3_L_2_p')
print(cc_L_3_L_2_p)
print('cc_L_3_L_2')
print(cc_L_3_L_2)
print('cc_L_3_p_L_2')
print(cc_L_3_p_L_2)
print('cc_L_3_p_L_2_p')
print(cc_L_3_p_L_2_p)
print('cc_L_2_p_L_2')
print(cc_L_2_p_L_2)

cov = np.cov(data_estimation)
print('covariance matrix')
print(cov)
corr = np.corrcoef(data_estimation)
print('normalized covariance matrix')
print(corr)



diff_L_3_L_2 = L_3-L_2
plt.hist(diff_L_3_L_2,bins = 50)
plt.title('actual_diff = {0}'.format(L_3_real-L_2_real))
plt.axvline(np.median(diff_L_3_L_2),label = 'median={0}'.format(np.median(diff_L_3_L_2)))
plt.legend()
plt.xlabel(r"$L_{3}-L_{2}$")
plt.savefig(dir+'L_3-L_2.png')
plt.show()

diff_L_3_L_2_p = L_3-L_2_p
plt.hist(diff_L_3_L_2_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_3_real-L_2_p_real))
plt.axvline(np.median(diff_L_3_L_2_p),label = 'median={0}'.format(np.median(diff_L_3_L_2_p)))
plt.legend()
plt.xlabel(r"$L_{3}-L^{'}_{2}$")
plt.savefig(dir+'L_3-L_2_p.png')
plt.show()

diff_L_3_L_3_p = L_3-L_3_p
plt.hist(diff_L_3_L_3_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_3_real-L_3_p_real))
plt.axvline(np.median(diff_L_3_L_3_p),label = 'median={0}'.format(np.median(diff_L_3_L_3_p)))
plt.legend()
plt.xlabel(r"$L_{3}-L^{'}_{3}$")
plt.savefig(dir+'L_3-L_3_p.png')
plt.show()

diff_L_2_L_2_p = L_2-L_2_p
plt.hist(diff_L_2_L_2_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_2_real-L_2_p_real))
plt.axvline(np.median(diff_L_2_L_2_p),label = 'median={0}'.format(np.median(diff_L_2_L_2_p)))
plt.legend()
plt.xlabel(r"$L_{2}-L^{'}_{2}$")
plt.savefig(dir+'L_2-L_2_p.png')
plt.show()

diff_L_2_L_3_p = L_2-L_3_p
plt.hist(diff_L_2_L_3_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_2_real-L_3_p_real))
plt.axvline(np.median(diff_L_2_L_3_p),label = 'median={0}'.format(np.median(diff_L_2_L_3_p)))
plt.legend()
plt.xlabel(r"$L_{2}-L^{'}_{3}$")
plt.savefig(dir+'L_2-L_3_p.png')
plt.show()

diff_L_3_p_L_2_p = L_3_p-L_2_p
plt.hist(diff_L_3_p_L_2_p,bins = 50)
plt.title('actual_diff = {0}'.format(L_3_p_real-L_2_p_real))
plt.axvline(np.median(diff_L_3_p_L_2_p),label = 'median={0}'.format(np.median(diff_L_3_p_L_2_p)))
plt.legend()
plt.xlabel(r"$L^{'}_{3}-L^{'}_{2}$")
plt.savefig(dir+'L_3_p-L_2_p.png')
plt.show()
'''


'''


# Set up the parameters of the problem.
ndim, nsamples = 3, 50000

# Generate some fake data.
np.random.seed(42)
data1 = np.random.randn(ndim * 4 * nsamples // 5).reshape([4 * nsamples // 5, ndim])
data2 = (4*np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples // 5).reshape([nsamples // 5, ndim]))
data = np.vstack([data1, data2])

print('data1')
print(data1)
print('data2')
print(data2)
print('data')
print(data)
# Plot it.
figure = corner.corner(data, labels=[r"$x$", r"$y$", r"$\log \alpha$", r"$\Gamma \, [\mathrm{parsec}]$"],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
plt.show()
'''
'''
removed = int(0.25*len(L_2))
#difference = L_1-L_2


index = np.argmax(likelihood)

print('L_3 max likelihood')
print(L_3[index])
print('L_2 max likelihood')
print(L_2[index])

plt.plot(likelihood)
plt.xlabel('iteration #')
plt.ylabel(r'$\log{\mathcal{L}}$')
plt.savefig(dir+'likelihood.png')
plt.show()

plt.plot(L_3-L_3_real)
plt.title('L_3 chain')
plt.ylabel(r'$L_{3}-L_{3_{True}}$')
plt.savefig(dir+'L_3_diff.png')
plt.show()

#plt.plot(L_2[int(len(L_1)/2)::])
plt.plot(L_2-L_2_real)
#plt.legend(['L_1','L_2'],loc='best')
plt.title('L_2 chain')
plt.ylabel(r'$L_{2}-L_{2_{True}}$')
plt.savefig(dir+'L_2_diff.png')
plt.show()

plt.plot(L_3_p-L_3_p_real)
plt.title('L_3_p chain')
plt.ylabel(r'$L_{3p}-L_{3p_{True}}$')
plt.savefig(dir+'L_3_p_diff.png')
plt.show()

#plt.plot(L_2[int(len(L_1)/2)::])
plt.plot(L_2_p-L_2_p_real)
#plt.legend(['L_1','L_2'],loc='best')
plt.title('L_2_p chain')
plt.ylabel(r'$L_{2p}-L_{2p_{True}}$')
plt.savefig(dir+'L_2_p_diff.png')
plt.show()
'''

'''

plt.plot(L_1-L_2-true_diff)
plt.ylabel(r'$\Delta L - \Delta L_{True}$')
plt.savefig(dir+'L_1_L_2_diff.png')
plt.show()

array_to_hist = difference-true_diff

weights = np.empty_like(array_to_hist)
bin =30
weights.fill(bin / (array_to_hist.max()-array_to_hist.min()) / array_to_hist.size)
plt.hist(array_to_hist, bins=bin, weights=weights)
#n,bins,patches=plt.hist(array_to_hist,density=True)
plt.xlabel(r'$\Delta L-\Delta L_{True}$ [s]')
plt.ylabel('POSTERIOR')
plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
#plt.xlabel(r'$L_{2}-L2_{True}$')
plt.savefig(dir+'delta_L_pdf.png')
plt.show()
'''


'''
cm = plt.cm.get_cmap('RdYlBu')
#sc = plt.scatter(L_1[removed:-1:1], L_2[removed:-1:1], c=likelihood[removed:-1:1], s=5, cmap=cm,label='likelihood')
sc = plt.scatter(L_3_p-L_3_p_real, L_2-L_2_real, c=likelihood, s=5, cmap=cm,label='likelihood')

#plt.plot(L_1,L_2)
plt.colorbar(sc)
plt.xlabel(r'$L_{3p}-L_{3p_{True}}$')
plt.ylabel(r'$L_{2}-L_{2_{True}}$')

#plt.xlim(-2e-5,2.5e-5)
#plt.ylim(-2e-5,2e-5)

#plt.ticklabel_format(axis='both', style='sci',scilimits=(-6,6))
plt.savefig(dir+'Two_D_Likelihood_L_3_p_L_2.png')
plt.show()
'''

'''
cm = plt.cm.get_cmap('RdYlBu')
#sc = plt.scatter(L_1[removed:-1:1], L_2[removed:-1:1], c=likelihood[removed:-1:1], s=5, cmap=cm,label='likelihood')
sc = plt.scatter(L_3-L_3_real, difference-true_diff, c=likelihood, s=5, cmap=cm,label='likelihood')
#plt.plot(L_1,L_2)
cbar = plt.colorbar(sc)
cbar.set_label('$log{\mathcal{L}}$')
plt.xlabel(r'$L_{1}-L_{1_{True}}$ [s]')
plt.ylabel(r'$\Delta L-\Delta L_{True}$ [s]')
plt.ticklabel_format(axis='both', style='sci',scilimits=(-6,6))
plt.savefig(dir+'Two_D_Difference_Likelihood.png')
plt.show()
'''

'''
print('L 1 median')
print(median_1)

print('L 2 median')
print(median_2)

print('L 1 lower')
print(p_l_1)
print('L 2 lower')
print(p_l_2)
print('L 1 upper')
print(p_u_1)
print('L 2 upper')
print(p_u_2)

print('max likelihood index')
print(index)
print('L_1 max likelihood')
print(L_1[index]+L_1_real)
print('L_2 max likelihood')
print(L_2[index]+L_2_real)
'''



'''
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot3D(L_1,L_2,likelihood)
#ax2.scatter3D(0,0,np.max(likelihood),color='black')
ax2.view_init(azim=50, elev=5)
ax2.set_xlabel(r'x where $x = 8.4 \pm \frac{x}{c}$')
ax2.set_ylabel(r'x where $x = -10.4 \pm \frac{x}{c}$')
ax2.set_zlabel('likelihood')
plt.show()
'''


'''
filename="samples_ESA_chain.h5"

f = h5py.File(filename, 'r')#ESA

print(f['mcmc'].keys())

print('accepted')
print(f['mcmc']['accepted'][:])

print('chain')
print(f['mcmc']['chain'][:])

print('logprob')
print(f['mcmc']['log_prob'][:])

print('length chain')
print(len(f['mcmc']['accepted'][0]))
'''

'''
#........................semi-major v time with 90% credible intervle shaded................................
#........................semi-major v time with 90% credible intervle shaded................................
#........................semi-major v time with 90% credible intervle shaded................................

data_LISA_Orbits = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_ESA_orbits_tcb_ltt_orbits_mprs_and_dpprs_to_file_1_hour_4_Hz_NO_AA_filter_NEW.dat',names=True)
times = data_LISA_Orbits['time']

semi_major_1_low_quantile = np.quantile(semi_major_1, 0.05)
semi_major_1_high_quantile = np.quantile(semi_major_1, 0.95)
semi_major_2_low_quantile = np.quantile(semi_major_2, 0.05)
semi_major_2_high_quantile = np.quantile(semi_major_2, 0.95)
semi_major_3_low_quantile = np.quantile(semi_major_3, 0.05)
semi_major_3_high_quantile = np.quantile(semi_major_3, 0.95)

semi_major_from_elements = np.genfromtxt('semi_major.dat').T

print('semi_major_from_elements')
print(semi_major_from_elements)

plt.plot(times,semi_major_from_elements[0])
plt.fill_between(times, semi_major_1_low_quantile,semi_major_1_high_quantile, alpha = 0.2, color = 'blue')
plt.title(r'$a_{1}$')
plt.savefig('a_1_time_dependence_w_90_percent.png')
plt.show()

plt.plot(times,semi_major_from_elements[1])
plt.fill_between(times, semi_major_2_low_quantile,semi_major_2_high_quantile, alpha = 0.2, color = 'blue')
plt.title(r'$a_{2}$')
plt.savefig('a_2_time_dependence_w_90_percent.png')
plt.show()

plt.plot(times,semi_major_from_elements[2])
plt.fill_between(times, semi_major_3_low_quantile, semi_major_3_high_quantile, alpha = 0.2, color = 'blue')

plt.title(r'$a_{3}$')
plt.savefig('a_3_time_dependence_w_90_percent.png')
plt.show()
'''

'''
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (7,7)
'''

#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------
'''
data_plot_1 = np.array([minit1,semi_major_1,omega_1,arg_per_1,eccentricity_1,inclination_1]).T
data_plot_2 = np.array([minit2,semi_major_2,omega_2,arg_per_2,eccentricity_2,inclination_2]).T
data_plot_3 = np.array([minit3,semi_major_3,omega_3,arg_per_3,eccentricity_3,inclination_3]).T

c = ChainConsumer()
#parameters=[ r"$L_{3}$", r"$L_{2}$",r"$L_{1}$", r"$L^{'}_{3}$",r"$L^{'}_{2}$",r"$L^{'}_{1}$",r"$\dot{L_{3}}$",r"$\dot{L_{2}}$",r"$\dot{L_{1}}$"]
#parameters=[r"$m_{\mathrm{init}}_{1}}$", r"$\lambda_{1}$",r"$e$"]
#parameters=["m", "lambda"]
#parameters=["m", "lambda","eccentricity"]
#parameters=[r"$m_{1_{0}}$",r"$a_{1}$",r"$\Omega_{1}$",r"$\omega_{1}$","e1","i1"]
parameters=[r"$M_{0} (\mathrm{ rad})$",r"$a(\mathrm{ km})$",r"$\Omega(\mathrm{ rad})$",r"$\omega(\mathrm{ rad})$",r"$e$",r"$\iota(\mathrm{ rad})$"]

c.add_chain(data_plot_1, parameters=parameters,color=color.to_hex('orange'),name=r'Spacecraft \# 1')
c.add_chain(data_plot_2, parameters=parameters,color=color.to_hex('purple'),name=r'Spacecraft \# 2')
c.add_chain(data_plot_3, parameters=parameters,color=color.to_hex('green'),name=r'Spacecraft \# 3')

c.configure(summary_area=0.95,spacing=0.0,cloud=False,kde=False,smooth=1,shade=False,usetex=True,linewidths=0.5,label_font_size=4,tick_font_size=4,summary=False,legend_kwargs={'fontsize':10})
c.configure_truth(color='k')
c.plotter.plot(chains=[r'Spacecraft \# 1',r'Spacecraft \# 2',r'Spacecraft \# 3'],figsize=(8.6,8.6),filename=dir+'corner_plot_sc_all_3_one_fig.png', truth=[minit1_truth,semi_major_1_truth,omega_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth,minit2_truth,semi_major_2_truth,omega_2_truth,arg_per_2_truth,eccentricity_2_truth,inclination_2_truth,minit3_truth,semi_major_3_truth,omega_3_truth,arg_per_3_truth,eccentricity_3_truth,inclination_3_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_sc_1.png', truth=[minit1_truth,semi_major_1_truth,arg_per_1_truth,eccentricity_1_truth,inclination_1_truth],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
#c.plotter.plot_walks(display=True)
plt.close()
'''