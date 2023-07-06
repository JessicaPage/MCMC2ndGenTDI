# MCMC2ndGen

MCMC2ndGenTDI directory is the main running directory for end user and future development. It can be run with all 
arguments given to class BayesTDI() as-is in example_mcmc_run.ipynb.

    If running Keplerian orbit model parameterization ({M01,a,w}-->Lij(t))

        generate_data_Keplerian.ipynb (and follow directions within)
        example_mcmc_run.ipynb (and follow directions for Keplerian within)

    If running numerical orbit model (18-parameter) parameterization:

        generate_data-ESA.ipynb
        elements_from_Cartesian.ipynb
        example_mcmc_run.ipynb (and follow directions for numerical/esa within)



Order of codes to run in Codes_Used_For_Paper/Notebook_For_Future_Use/ESA_Data_Runs (notebook headings are self-explanatory):

    generate_data-ESA.ipynb
    elements_from_Cartesian.ipynb
    mcmc_ESA_Zeus.ipynb

mcmc_ESA_Zeus.ipynb is an example sampler for the ESA data. But it is not complete and has not resulted in converged posteriors yet. But there are differences in the L(t) calculations from the Keplerian mcmc because of differing parameterizations so nearly complete version is given here.


Order of codes to run in Codes_Used_For_Paper/Notebook_For_Future_Use/Keplerian_Data_Runs (notebook headings are self-explanatory):

    generate_data.ipynb
    mcmc_Keplerian_Data.ipynb
    plot_chains.ipynb once step 2 is done sampling

Codes_Used_For_Paper Directory are the unaltered files used in the data for paper https://arxiv.org/abs/2305.14186.
