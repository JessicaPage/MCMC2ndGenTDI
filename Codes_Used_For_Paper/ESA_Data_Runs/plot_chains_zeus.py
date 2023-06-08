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
test_name='Zeus Sampled ESA Data Keplerian Parameters'


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


