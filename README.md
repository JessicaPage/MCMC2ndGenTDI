# MCMC2ndGenTDI

This generates MCMC samples of parameters to estimate the time-varying LISA spacecraft separations required for laser frequency noise suppression in time-delay interferometry (TDI). See the paper published on the subject [here](https://arxiv.org/abs/2305.14186). The data to sample over is generated with [LISA Instrument and LISA Orbits](https://gitlab.in2p3.fr/lisa-simulation/instrument), so the naming and labeling conventions may look similar to their codes for clarity.

The default settings for creating a BayesTDI() class instance are for LISA data generated with Keplerian orbit model parameterization, i.e. ({M01,a,w}--> Lij(t)). 

If running Keplerian orbit model parameterization, i.e. ({M01,a,w}--> Lij(t))

1) generate_data_Keplerian.ipynb (and follow directions within)
2) example_mcmc_run.ipynb (and follow directions for Keplerian within)



Sampling for arbitrary numerical orbits: The spacecraft separations are parameterized by 6 orbital elements that differ for each of the 3 spacecraft. See [the paper](https://arxiv.org/abs/2305.14186) for more details. Successful sampling is still a work in progress, but directions for the numerical options are found in example_mcmc_run.ipynb. 


If running numerical orbit model (18-parameter)  parameterization:

1) generate_data-ESA.ipynb
2) elements_from_Cartesian.ipynb
3) example_mcmc_run.ipynb (and follow directions for numerical/esa within)

## Installation

```
pip install mcmc2ndgentdi
```

## Current version does not link to default data file that is included in package. Currently, you must download the repository zip file and put the data files in the directory you run your code from.


## Example Use

```
from mcmc2ndgentdi.run_mcmc_TDI import BayesTDI
```
### Create BayesTDI() class instance. Set all arguments here. If running data that used numerical orbit model (18 parameter model) run first example and comment line 2, if running data that used Keplerian model, use second example and comment line 1.

```
#b1 = BayesTDI('LISA_Instrument_ESA_orbits_tcb_orbits_4_Hz_3600_sec.dat', cut_off=0,f_s=4.0,t_init= 13100.00,f_min= 5.0e-4,f_max = 0.1,orbit_model='esa',orbital_elements_file='elements_from_Cartesian_4_Hz_3600_sec.dat',tcb=True,number_n=7,Nens = 37,Nburnin = 100,Nsamples = 100000)
b1 = BayesTDI('LISA_Instrument_Keplerian_orbits_ppr_orbits_4_Hz_3600_sec.dat', cut_off=0,f_s=4.0,t_init=0.0,f_min= 5.0e-4,f_max = 0.1,orbit_model='keplerian',orbital_elements_file=None,tcb=False,number_n=7,Nens = 37,Nburnin = 100,Nsamples = 100000)
```

### Initialize best einsum path for FDI filter
```
einsum_path_to_use = b1.get_einsum_path()
```

### Run either zeus sampler for numerical orbit model data (18-parameter model; still under development for converged posteriors) or the working Keplerian orbit model parameterization.

```
b1.run_emcee_Keplerian_mcmc(einsum_path_to_use)
```
