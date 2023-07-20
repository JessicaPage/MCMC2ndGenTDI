#MCMC2ndGen

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


