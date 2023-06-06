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
import emcee
import matplotlib as mpl

# # CREATING FILTERS
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
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

test_name=' '

reader = emcee.backends.HDFBackend("samples_Keplerian_chain_omega_emcee_backend_testing.h5", read_only=True)
#dir = './Keplerian_data_1_day_Keplerian_orbital_parameters_minit_semi_major_arg_per_2Hz_N=7_fmin=5e-4_fmax=0_03_emcee_testing/'
dir = './'
'''
reader = emcee.backends.HDFBackend("samples_Keplerian_chain_omega_emcee_backend_testing_small_ball.h5", read_only=True)
dir = './Keplerian_data_1_day_Keplerian_orbital_parameters_minit_semi_major_arg_per_2Hz_N=7_fmin=5e-4_fmax=0_03_emcee_testing_small_ball/'
'''


'''
try:
	tau = reader.get_autocorr_time()
except:
	print('not accurate tau at all. Just for plotting purposes')
	tau = 5
'''

tau=500

print('tau')
print(tau)



#tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
#thin = int(0.5 * np.min(tau))
thin =1
flatchain = reader.get_chain(discard=burnin,thin=thin,flat=True)
log_prob = reader.get_log_prob(discard=burnin,thin=thin,flat=True)
#blobs = reader.get_blobs()

print('flatchain')
print(flatchain)
print('flatchain.T')

print(flatchain.T)
print('log_prob')
print(log_prob)




#sys.exit()


removed = 0
minit1 = flatchain.T[0][removed::]
semi_major = flatchain.T[1][removed::]
arg_per = flatchain.T[2][removed::]

cut_off = int(1e3)

minit1_truth = 0.0
delta_truth = 5.0/8
semi_major_truth = ASTRONOMICAL_UNIT
Omega_1_truth = np.pi/2.0
#eccentricity_truth = 0.004815434522687179
arg_per_truth = -np.pi/2.0

low_semi_major = 149597800000.0 # where LISA Constants is 149597870700.0
high_semi_major = 1.499e11 



#ndim, nsamples = 5, len(likelihood)

#data_plot = np.array([likelihood,L_3_here,L_2_here,L_3_p_here,L_2_p_here]).T

#data_plot = np.array([L_3_here,L_2_here,L_1_here,L_3_p_here,L_2_p_here,L_1_p_here]).T
#data_plot_dot = np.array([L_3_here_dot,L_2_here_dot,L_1_here_dot,L_3_p_here_dot,L_2_p_here_dot,L_1_p_here_dot]).T
#data_plot = np.array([L_3,L_2,L_1,L_3_p,L_2_p,L_1_p,L_3_dot,L_2_dot,L_1_dot]).T
data_plot = np.array([minit1,semi_major-semi_major_truth,arg_per-arg_per_truth]).T
#data_plot = np.array([minit1,lambda1]).T



#data_plot_31 = np.array([L_3_here_31,L_2_here_31,L_1_here_31,L_3_p_here_31,L_2_p_here_31,L_1_p_here_31]).T



#----------------------------------------------------------------------------
#chain consumer method
#----------------------------------------------------------------------------


c = ChainConsumer()
#parameters=[ r"$L_{3}$", r"$L_{2}$",r"$L_{1}$", r"$L^{'}_{3}$",r"$L^{'}_{2}$",r"$L^{'}_{1}$",r"$\dot{L_{3}}$",r"$\dot{L_{2}}$",r"$\dot{L_{1}}$"]
#parameters=[r"$m_{\mathrm{init}}_{1}}$", r"$\lambda_{1}$",r"$e$"]
#parameters=["m", "lambda"]
#parameters=["m", "lambda","eccentricity"]
parameters=[r"$M_{1_{0}} (\mathrm{rad})$",r"$\Delta a (\mathrm{m})$",r"$\Delta\omega (\mathrm{rad})$"]


c.add_chain(data_plot, parameters=parameters,color=color.to_hex('blue'),name=test_name)
#c.configure(sigmas=[0,1.645],spacing=1.0,kde=False,smooth=1,shade=True,usetex=True)
c.configure(summary_area=0.95,spacing=1.0,kde=False,smooth=1,shade=True,usetex=True,label_font_size=15,linewidths=1.5)

c.configure_truth(color='k')
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_day_3_Hz_N=5_fmax=0_03.png', truth=[minit1_truth,semi_major_truth,arg_per_truth],display=True,legend=True)
c.plotter.plot(chains=[test_name],figsize=(8.6,8.6),filename=dir+'corner_plot_Keplerian_Parameters_1_day_3_Hz_N=5_fmax=0_03_differences.png', truth=[minit1_truth,0.0,0.0],display=True,legend=True)
#c.plotter.plot(chains=[test_name],figsize=(7,7),filename=dir+'corner_plot_Keplerian_Parameters_1_hour.png', truth=[minit1_truth,lambda1_truth],display=True,legend=True)
c.plotter.plot_walks(display=True)

plt.close()

'''
plt.plot(semi_major,label='a posterior')
plt.axhline(semi_major_truth,label='a truth')
#plt.axhline(low_semi_major,label='a low cut off prior')
#plt.axhline(high_semi_major,label='a low cut off prior')
plt.ylabel('a')
plt.xlabel('iteration')
plt.legend()
plt.show()



plt.plot(minit1)
plt.axhline(minit1_truth)

plt.ylabel('m1')
plt.xlabel('iteration')
plt.show()

plt.plot(arg_per)
plt.axhline(arg_per_truth)
plt.ylabel(r'$\omega$')
plt.xlabel('iteration')
plt.show()
'''
'''

plt.plot(Omega_1)
plt.axhline(Omega_1_truth)
plt.ylabel('Omega1')
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
filename = 'samples_Keplerian_chain_Omega_omega_emcee_backend_copy.h5'

f = h5py.File(filename, 'r')
for key in f.keys():
    print(key) #Names of the root level object names in HDF5 file - can be groups or datasets.
    print(type(f[key])) # get the object type: usually group or dataset

#Get the HDF5 group; key needs to be a group name from above
mcmc = f['mcmc']
#print(mcmc)
#Checkout what keys are inside that group.
for key in mcmc.keys():
    print(key)

print(mcmc['log_prob'][:])
print(mcmc['chain'][:])
'''

'''
#Test #1
data = np.recfromtxt('chainfile_testing_LISA_Instrument_Data_disable_all_but_laser_lock_six_static_tps_ppr_RR_comparison.dat',names=True)
data_truths = np.genfromtxt('LISA_Instrument_RR_disable_all_but_laser_lock_six_static_orbits_tps_ppr_orbits_pyTDI_size.dat',names=True)
filename = 'measurements_disable_all_but_laser_lock_six_static_orbits_tps_ppr_pyTDI_size.h5'
dir='./static_tps_ppr_RR_comparison/'
test_number=1
test_name='Static Data, Static Functions'
'''


'''
#Test #2
#data = np.recfromtxt('chainfile_testing_LISA_Instrument_Data_disable_all_but_laser_lock_six_static_tps_ppr_TDI_2_0_Keplerian_data_1_day_Keplerian_orbital_parameters_minit_semi_major_omega_Omega1_2Hz_N=7_fmin=5e-4_fmax=0_03.dat',names=True)
#data = np.recfromtxt('chainfile_testing_LISA_Instrument_Data_disable_all_but_laser_lock_six_static_tps_ppr_TDI_2_0_Keplerian_data_1_day_Keplerian_orbital_parameters_minit_semi_major_omega_Omega1_2Hz_N=7_fmin=5e-4_fmax=0_03_partly_blocked_proposals.dat',names=True)
data = np.genfromtxt('samples_Keplerian_chain_omega_emcee.dat')
#data = np.genfromtxt('chainfile_testing_LISA_Instrument_Data_disable_all_but_laser_lock_six_static_tps_ppr_TDI_2_0_Keplerian_data_1_day_Keplerian_orbital_parameters_minit_semi_major_omega_Omega1_2Hz_N=7_fmin=5e-4_fmax=0_03_WITH_DELTA.dat',names=True)
#data_truths = np.recfromtxt('chainfile_testing_LISA_Instrument_Data_disable_all_but_laser_lock_six_static_tps_ppr_TDI_2_0_Keplerian_one_hour_linear.dat',names=True)
dir = './Keplerian_data_1_day_Keplerian_orbital_parameters_minit_semi_major_arg_per_2Hz_N=7_fmin=5e-4_fmax=0_03_emcee/'

test_number=2
test_name=' '




likelihood = data['likelihood']
likelihood_truth = likelihood[0]
#Only analyzing first 25%, (lazy way of evaluating burn-in time)
removed = int(0.25*len(likelihood))
#removed=0
likelihood = data['likelihood'][removed::]
minit1 = data['minit1'][removed::]
#delta = data['delta'][removed::]
semi_major = data['semi_major'][removed::]
Omega_1 = data['Omega_1'][removed::]
arg_per = data['arg_per'][removed::]

ar = data['Current_AR'][removed::]
chi_2 = data['chi_2_here'][removed::]

#removed = 51
#removed=50
removed =0
minit1 = data[:,1][removed::]
#delta = data['delta'][removed::]
semi_major = data[:,2][removed::]
#Omega_1 = data[:,3][removed::]
arg_per = data[:,3][removed::]
print('semi_major')
print(semi_major)
print(len(semi_major))
print('data')
print(data)

#length = len(likelihood)
'''