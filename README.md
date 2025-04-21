# Bayesian Calibration of the 40K Decay Scheme
This repository contains python code and data to run the Markov Chain Monte Carlo (MCMC) algoritum for the estimation of posterior distributions for all 4 modes of potassium-40 decay and the total potassium-40 decay constant. 

This algorithm is a framework in the manuscript - Bayesian calibration of the 40K decay scheme with implications for 40K-based geochronology. Geochimica et Cosmochimica Acta (2025) - Carter, J.N., Hasler, C.E., Fuentes, A.J., Tholt, A.J., Morgan, L.E. and Renne, P.R.


This work is a combined effort from the 40Ar/39Ar lab-group at the Berkeley Geochronology Center (BGC) and discussion and collaboration with Leah Morgan (USGS). 

Why it is necessary
-------------------
The 40K decay constant underpins both the K-Ar and 40Ar/39Ar chronometers. However, currently there are a range of potential choices for calibrations that have unique values and uncertainties. With this algorithm we attempt to make the "best" estimate of the 40K decay constant using the analyses of Vesuvius and a suite of samples with combined 40Ar/39Ar, U238/Pb206, and U235/207Pb data. We have attempted to account for as many unknowns of all of these systems with unique age perturbation parameters for all samples. We then perform a detailed MCMC analysis to make inferences on the 40K total decay constant and all other model parameters. For all details see - Jack N. Carter, Caroline E.J. Hasler, Anthony J. Fuentes, Andrew J. Tholt, Leah E. Morgan, Paul R. Renne,
Bayesian calibration of the 40K decay scheme with implications for 40K-based geochronology,
Geochimica et Cosmochimica Acta,
2025,
,
ISSN 0016-7037,
https://doi.org/10.1016/j.gca.2025.03.024.
(https://www.sciencedirect.com/science/article/pii/S0016703725001620)

Paper Abstract
--------------
The K/Ar and 40Ar/39Ar geochronometers are based on the naturally occurring radionuclide 40K. Their precision and accuracy are limited by uncertainties on the 40K decay constants and, in the case of the 40Ar/39Ar geochronometer, the isotopic composition of neutron fluence monitors. To address these limitations, we introduce a Bayesian calibration of the 40K decay scheme. We formulate robust priors for all model parameters including partial 40K decay constants, 238U and 235U decay constants, and age offset parameters to account for phenomena that can perturb apparent U-Pb and 40Ar/39Ar ages. We then harness a set of complementary 40Ar/39Ar, 238U/206Pb, and 235U/207Pb data from well- characterized geological samples with ages from 1.919 ka to 2000 Ma to derive Bayesian estimates of the 40K decay constants. Posterior values for the partial 40K decay constants are λβ-= (4.9252 ± 0.0054) × 10−10 yr−1, λβ+ = (5.6658 ± 0.1543) × 10−15 yr−1, λEC∗ = (5.7404 ± 0.0053) × 10−11 yr−1, and λEC0= (4.9060 ± 0.2942) × 10−13 yr−1 (uncertainties reported at the 68 % (1σ) credible interval). These combine to a total 40K decay constant λtot= (5.5042 ± 0.0054) × 10−10 yr−1. Model estimates of the 238U and 235U decay constants are statistically indistinguishable from those reported by Jaffey et al. (1971). Posterior values of the 40K decay constants and the 40Ar*/40K isotopic composition of Fish Canyon sanidine (FCs) define a K/Ar FCs age of 28.183 ± 0.017 Ma (1σ). Significantly, Bayesian calibrated 40Ar/39Ar ages align with astronomically tuned ages throughout the Cenozoic and with 238U/206Pb and 235U/207Pb ages in the Mesozoic, Paleozoic, and Proterozoic, as well as having comparable precision to the 238U/206Pb method. Thus, Bayesian calibration of the 40K decay scheme and the K/Ar age of FCs reconciles the 40Ar/39Ar, U-Pb, and astronomical chronometers.


Input data
----------
File name - Final_Bayes_Input_Data.xlsx 
Excel spreadsheet containing all the input data used in the model. These are R-values relative to Fish Canyon sanidine (FCs), U235/Pb207, and U238/Pb206 ratios. All data has been pre-treated such that the uncertainties reported are inflated by the square root of the Mean Square Weighted Deviations (MSWDs) of the raw data if the MSWD > 1. All uncertainties are reported at the 1 sigma or 68% confidence level. 


Python Class
------------
BayesCal_MCMC.py is a python class that holds in it all the function needed for the algorithm. 

Jupyter Notebook
----------------
Running_Notebook.ipynb is a jupyter notebook that first reads in the python then allows the user to define how many chains, how many iterations, and how many batches to run the MCMC for. This notebook also lays out the handy function calls to get various diagnostic chain results (e.g., posterior plot or Gelman-Rubin statistic) or inferences of the model (e.g., Get_40K_Decay_Stuff). Comments are provided in the notebook for easier reading. 


