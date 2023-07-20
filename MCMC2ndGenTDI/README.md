This generates MCMC samples of parameters to estimate the time-varying LISA spacecraft separations required for laser frequency noise suppression in time-delay interferometry (TDI). See the paper published on the subject [here](https://arxiv.org/abs/2305.14186). The data to sample over is generated with [LISA Instrument and LISA Orbits](https://gitlab.in2p3.fr/lisa-simulation/instrument), so the naming and labeling conventions may look similar to their codes for clarity.

If running Keplerian orbit model parameterization, i.e. ({M01,a,w}--> Lij(t))

1) generate_data_Keplerian.ipynb (and follow directions within)
2) example_mcmc_run.ipynb (and follow directions for Keplerian within)


If running numerical orbit model (18-parameter)  parameterization:

1) generate_data-ESA.ipynb
2) elements_from_Cartesian.ipynb
3) example_mcmc_run.ipynb (and follow directions for numerical/esa within)


