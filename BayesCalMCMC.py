import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import gzip
import pickle
from scipy.stats import norm, uniform
from scipy import stats
import os
import time
import random
import matplotlib.pyplot as plt
import itertools

class BayesCal_MCMC:
    """
    Class for the Bayesian Calibration of the 40K Decay Scheme
    ----------------------------------------------------------
    
    Inputs
    ------
    Data - suite of triplets - 238U/206Pb ratios, 235U/207Pb ratios, and R-values (relating a sample to Fish Canyon sanidine (FCs)
    nchains - How many chains do you want to run? - The more the merrier but watch out for computation time
    iterations - How many iterations per chain?
    Run_Name - Make a name and everything will store with this prefix so you can find it
    Start_from_Pickles - Flag (True or False) - if True you will start from the last pickled position and the chain will run from there. If False - start again.
    nbatch - Run the chains in batches for each example if you select 10000 iterations and 40 batches it will do a total of 40 * 10000 samples and store them all.
    The analysis will then compress all stored batched pickle files together for final analysis
    
    Outputs
    -------
    Estimates of all four modes of 40K decay and the total 40K decay constant.
    Another output is the K-Ar age of FCs.
    
    Not a new idea. Just an adaptation from the optimization model of Renne et al. (2010, 2011). We try to make the estimate less wrong by including like age altering phenomena that can make ages older or younger or offset and keep ages the same. We include as many parameters as we can think of which will effect the internal argon relationhsips and the internal U/Pb relationships.
    
    Notes on the Pickles
    ====================
    This model will take time (maybe a week of computation time to reach convergence). We use pickles so you can effectively store all tuple outputs and then just start them up from the last saved position. Then you just keep doing this until convergence (assess with trace plots and Gelman-Rubin statistic, also track the temperature for the annealing we targeted a temperature of 0.05 then did a large sampling batch ~ 5 million for all results.). This is just a suggestion and can probably be done better but i was quite happy with it. We tune every parameter of the model to reach an acceptance rate of 23.4% (recommend printing this out every now and then to see how it behaves).
    
    Pickles will overwrite everytime and for each run the result will only reflect that batch.
    Because of the pickles i dont assign a burn-in. So you just have to keep running and running until you reach desired temperature and acceptance rate and then do a large sampling batch from the stationary MCMC.
    
    Future
    ======
    - Add autocorrelation function
    - Use emcee (Foreman-Mackey et al. (2013)) - used in high-dimensional cosmology models a lot would be ideal.
    - More parameters or tuned parameter?
    - Adaptive priors
    """
    
    def __init__(self,
                 data,
                 nchains = 10,
                 iterations = 100000,
                 Run_Name = 'Run_Test',
                 Start_from_Pickles = False,
                 nbatch = 40):
        
        
        self.nchains = nchains
        self.N_data = data.shape[0]
        self.Run_Name = Run_Name
        self.iterations = iterations
        self.data = data
        self.Start_from_pickles = Start_from_Pickles # Can be switch on and off. If off you will start fresh and create a new batch of pickles
        self.nbatch = nbatch
        self.Grouped_Chain_Results = None # Empty tuple for later use as store all chain results
        
        # Data
        # U235/Pb207 ratios, U238/Pb206 ratios, R-values to FCs
        self.u68_ratios = self.data['U238_ratios'].values
        self.u68_ratios_err = self.data['U238_ratios_err'].values

        self.u75_ratios = self.data['U235_ratios'].values
        self.u75_ratios_err = self.data['U235_ratios_err'].values

        self.R_values   = self.data['R_values'].values
        self.R_values_err   = self.data['R_values_err'].values
        
        # Age Uncertainty of the U68 and U75 systems
        self.U68_error = self.data['U238_age_err (Ma)'].values
        self.U75_error = self.data['U235_age_err (Ma)'].values

        
        # Priors
        self.K_act_EC = 3.3173 # Mougeot and Helmer 2009
        self.K_act_EC_err = 0.0305
        
        self.K_act_Beta = 28.057 # Mougeot and Helmer 2009
        self.K_act_Beta_err = 0.14
        
        self.K_act_Positron = 1.15e-5 # Engelkiemer et al.
        self.K_act_Positron_err = 0.14e-5
        
        self.K_act_EC_ground = 0.0095 # Stukel et al. (2023) and Hairasz et al. (2023)
        self.K_act_EC_ground_err = 0.0024 # Complete uncertainty random and systematic
        
        self.FCS_kappa = 1.6407e-3 # Jourdan and Renne (2007)
        self.FCS_kappa_err = 0.0047e-3
        
        self.res_time_low = 0.0 # Ma
        self.res_time_high = 0.5 # Ma
        
        
        self.U238_decay_const = 1.551254796141587e-10 # Jaffey et al. (1971)
        self.U238_decay_const_err = 8.33205360145874e-14 * 1.5 # Mattinson Suggested Inflation
        
        self.U235_decay_const = 9.848498608430476e-10 # Jaffey et al. (1971)
        self.U235_decay_const_err = 6.716698160081027e-13 * 1.5 # Mattinson Suggested Inflation
        
        self.K40K_naumenko = 1.1668e-4 # Naumenko et al. (2013) - Result for SRM985 (This is used for the 40K/K of the decay constant material)
        self.K40K_naumenko_err = 0.0004e-4
        
        self.K40K_FCs_Morgan = 1.16590011999e-4 # Morgan et al. (2018) - Prior for FCs - measured directly by Morgan et al. (2018) using delta 41K - We use the delta 41K and make an inference as to what the 40K/K would be for FCs.
        self.K40K_FCs_Morgan_err = 0.000812e-4
        
        self.K40K_silicate = 1.16586449e-4 # From Morgan et al. (2018). Like the FCs we use the entire range from silicates reported from Morgan et al. (2018) from delta 41K and make inferences as to the range of 40K/K. Few assumptions are made in this calculation. Need to upload this code as well!
        self.K40K_silicate_err = 0.00823e-4
        
        self.Katmw = 39.0983 # Garner et al. (1975) - Maybe updated to Morgan et al. (2018)?
        self.Katmw_err = 0.00012
        
        self.Avogrado = 6.0221367e23 # Cohen and Taylor (1987)
        self.Avogrado_err = 0.0000072e23
        # Calculate Vesuvius Age using R-value
        # and Min/Kuiper et al.
        # Just using Monte Carlo
        # Age uncertainty required for likelihood
        
        self.Vesuvius_Age1_err = self.Ves_age(self.R_values[0], self.R_values_err[0])
        self.Vesuvius_Age2_err = self.Ves_age(self.R_values[1], self.R_values_err[1])
        self.Vesuvius_Age1 = 1917.5 /1e6 # Measurements (1997) to timing of Vesuvius Eruption
        self.Vesuvius_Age2 = (1917.5 + 16)/1e6 # Measurements (2013) to timing of Vesuvius Eruption

    def Ves_age(self, R, Rerr):
        age = np.zeros(30000)
        for i in range(30000):
            r_mc = np.random.normal(R, Rerr)
            lam = np.random.normal(5.463e-10, 0.054e-10)
            fcs = np.random.normal(28.201e6, 0.023e6)
            z = np.exp(lam * fcs ) - 1
            age[i] = (1/lam) * np.log(z * r_mc + 1)
        return age.std()/1e6
        
        
    def Vesuvius_prior1(self, x):
        # Prior for the age of Vesuvius relative to Renne et al. (1997)
        # I give a year "uncertainty" as a flat top prior then
        # each side is a large penalty for being older or younger
        # These are stitched priors but the point is to not favour any date with the potential window
        # Pliny the younger - August 79CE to October 79CE based on other finds (harvests?)
        uniform_density = 1 / (0.001918 - 0.001917)
        lambda_exp = uniform_density
        
        if x < 0.001917:
            exp_pdf = lambda_exp * np.exp(- lambda_exp * 10 *(0.001917 - x))
            return np.log(exp_pdf)
        
        elif 0.001917 <= x <= 0.001918:
            return np.log(uniform_density)
        
        else:
            exp_pdf = lambda_exp * np.exp(- lambda_exp * 10 *(x - 0.001918))
            return np.log(exp_pdf)
    
    
    def Vesuvius_prior2(self, x):
        # Prior for the age of the Vesuivus eruption from the
        # 2013 Measurement of Renne et al. (2013)
        uniform_density = 1 / (0.001934 - 0.001933)
        lambda_exp = uniform_density
        
        if x < 0.001933:
            exp_pdf = lambda_exp * np.exp(- lambda_exp * 10 *(0.001933 - x))
        
            return np.log(exp_pdf)
        
        elif 0.001933 <= x <= 0.001934:
            return np.log(uniform_density)
        
        else:
            exp_pdf = lambda_exp * np.exp(- lambda_exp * 10 *(x - 0.001934))
            return np.log(exp_pdf)
        
    # Needed Functions
    def FishCanyonAge(self, theta):
        """
        Read in Model parameters and calculate the age of Fish Canyon
        --------------
        Input - Theta
        Output - FCs age
        Accounts for natural variability in 40K/K (Morgan et al. 2018)
        40K/K variability between the decay constant material, the neutron fluence monitor, and the
        unknown sample
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        lambda_beta, lambda_electron_capture = self.Decay_Constant(theta)
        
        lambda_ec0 = (lambda_electron_capture * kdk_ratio)
        
        lambda_beta_plus = (beta_plus_beta_minus * lambda_beta)
    
        
        lambda_total = lambda_beta + lambda_electron_capture + lambda_ec0 + lambda_beta_plus
        
        lambda_ar = lambda_electron_capture + lambda_ec0 + lambda_beta_plus
    
        kappa_ = F_value * (K40_K_decayconst/K40_K_fcs)
        
        
        FCs_age = (1/(lambda_total))  \
        * np.log(1 + ((lambda_total)/lambda_ar)*kappa_)
    
        return FCs_age

    def Decay_Constant(self, theta):
        """
        Function to get decay constants from the activity and other physical parameters
        Input - Theta
        Output - beta decay branch and electron capture branch
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        # 3.155693e7 is the number of seconds in the mean solar year
    
        lambda_beta = Activity_beta * ((K_atmw * 3.155693e7) / (K40_K_decayconst * Avagadro))
    
        lambda_electron_capture = Activity_electron_capture * ((K_atmw * 3.155693e7) /  (K40_K_decayconst * Avagadro))
    
    
        return lambda_beta, lambda_electron_capture
        
        
    def Total_Decay_Constant(self, theta):
        """
        Function to calculate the total decay constant
        Input - Theta
        Ouput - Total decay constant
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        lambda_beta, lambda_electron_capture = self.Decay_Constant(theta)
        
        lambda_ec0 = lambda_electron_capture * kdk_ratio
        
        lambda_beta_plus = beta_plus_beta_minus * lambda_beta

        
        lambda_total = lambda_beta + lambda_electron_capture + lambda_ec0 + lambda_beta_plus

        
        return lambda_total
        
    def Partial_Decay_Constants(self, theta):
        """
        Function that determine all four 40K decay constants
        Input - Theta
        Output - all four branches
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        lambda_beta, lambda_electron_capture = self.Decay_Constant(theta)
        
        lambda_ec0 = lambda_electron_capture * kdk_ratio
        
        lambda_beta_plus = beta_plus_beta_minus * lambda_beta

        return lambda_beta, lambda_electron_capture, lambda_ec0, lambda_beta_plus
        


    def ArAr_Age_Model(self, theta):
        """
        This function returns the
        model Ar/Ar ages
        ---------------------------
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        lambda_beta, lambda_electron_capture = self.Decay_Constant(theta)
        
        lambda_ec0 = lambda_electron_capture * kdk_ratio
        
        lambda_beta_plus = beta_plus_beta_minus * lambda_beta

        lambda_total = lambda_beta + lambda_electron_capture + lambda_ec0 + lambda_beta_plus
        
        lambda_ar = lambda_electron_capture + lambda_ec0 + lambda_beta_plus

        r__ = K40_K_sample/K40_K_fcs

        kappa_ = F_value * (K40_K_decayconst/K40_K_fcs)
        
        part1 = 1/(lambda_total)
        
        part2 = np.log(((lambda_total/lambda_ar)*kappa_*R_values*r__) + 1)
        
        t_argon = (part1 * part2)
        
        return t_argon
        
        

    def U238_Age_Model(self, theta):
        """
        This function returns the
        238U/206Pb age
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        U_age = (1/U_238_lam) * np.log(1 + UPb_ratios)
        
        return U_age
    
    def U235_Age_Model(self, theta):
        """
        This function returns the
        235U/207Pb age
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        U_age = (1/U_235_lam) * np.log(1 + UPb_7_ratios)

        return U_age
        
    def ArAr_Model_Age_with_all_correction(self, theta):
        # Function to calculate "corrected" ArAr ages
        # Not really a correction but
        # incoporates the unique age perturbation parameter
        Age = self.ArAr_Age_Model(theta)/1e6
        Age_corr = Age + theta[10]
        
        return Age_corr
        
    def U238_Model_Ages_Corr(self, theta):
        # Function to calculate "corrected" U86 ages
        # Not really a correction but
        # incoporates the unique age perturbation parameter and residence time
        U238_age = self.U238_Age_Model(theta)/1e6
        
        # U238 Corrected age is substracting residence and
        # Add age perturbation parameter
        U238_age_corr = U238_age[2:] - (theta[13]) + theta[11]
        return U238_age_corr
        
    def U235_Model_Ages_Corr(self, theta):
        # Function to calculate "corrected" U75 ages
        # Not really a correction but
        # incoporates the unique age perturbation parameter and residence time
        # To use
        U235_age = self.U235_Age_Model(theta)/1e6
        U235_age_corr = U235_age[3:] - (theta[13][1:]) + theta[12]
        return U235_age_corr

    def Normal_loglike_quick(self, model, data, error):
        # Reduce form of the Normal log-likelihood
        delta = model - data
        ll = - 0.5 * np.square(delta / error)
        return ll
    
    def Make_Initial_Theta_Guess(self):
        initial_thetas = []
        
        for i in range(0, self.nchains):
            log_prior = - np.inf
            while log_prior == -np.inf:
                
                Activity_EC_Star_guess = np.random.normal(self.K_act_EC,
                                                   self.K_act_EC_err)
                                                   
                Activity_Beta_guess = np.random.normal(self.K_act_Beta,
                                                 self.K_act_Beta_err)
                                                 
                Avogrado_guess = np.random.normal(self.Avogrado,
                                            self.Avogrado_err)
                                            
                Positron_guess = np.random.normal(self.K_act_Positron,
                                                  self.K_act_Positron_err)
                
                EC_ground_guess = np.random.normal(self.K_act_EC_ground,
                                                    self.K_act_EC_ground_err)
                                                    
                kappa_value_guess = np.random.normal(self.FCS_kappa,
                                                     self.FCS_kappa_err)
                                                     
                K40K_samples_guess = np.random.normal(self.K40K_silicate,
                                                     self.K40K_silicate_err,
                                                     self.N_data)
                                                     
                K40K_FCs_guess = np.random.normal(self.K40K_FCs_Morgan,
                                                self.K40K_FCs_Morgan_err)
                                                
                K40K_decayconst_guess = np.random.normal(self.K40K_naumenko,
                                                self.K40K_naumenko_err)
                    
                Katmw_guess = np.random.normal(self.Katmw, self.Katmw_err)
                
                U235_decay_const_guess = np.random.normal(self.U235_decay_const,
                                                          self.U235_decay_const_err)
                                                          
                U238_decay_const_guess = np.random.normal(self.U238_decay_const,
                                                          self.U238_decay_const_err)
                                                
                residencetime_guess = np.random.uniform(self.res_time_low,
                self.res_time_high, self.N_data - 2)
                # Age Perturbing Phenomena
                ArAr_ext_guess = np.zeros(self.N_data)
                U238_ext_guess = np.zeros(self.N_data - 2)
                U235_ext_guess = np.zeros(self.N_data - 3)

                # Input data triplets
                R_value_guesses = np.random.normal(self.R_values, self.R_values_err)
                
                U238_ratio_guesses = np.random.normal(self.u68_ratios, self.u68_ratios_err)
                
                U235_ratio_guesses = np.random.normal(self.u75_ratios, self.u75_ratios_err)
                
                # Get full theta
                theta_initial = (Activity_Beta_guess, Activity_EC_Star_guess,
                Avogrado_guess, K40K_decayconst_guess, K40K_samples_guess, K40K_FCs_guess,
                Katmw_guess, Positron_guess, EC_ground_guess,
                kappa_value_guess, ArAr_ext_guess, U238_ext_guess, U235_ext_guess,
                residencetime_guess, R_value_guesses, U238_ratio_guesses, U238_decay_const_guess,
                U235_ratio_guesses, U235_decay_const_guess)
                
                log_prior = self.Ln_Posterior(theta_initial)
                
                if log_prior != -np.inf:
                    initial_thetas.append(theta_initial)
                
        return initial_thetas
        
    def is_negative_or_zero(self, val):
        """
        Helper function
        to check if the
        value or any value
        in an array is
        less than or equal to zero.
        """
        if np.isscalar(val):
            return val <= 0
        else:
            return (val <= 0).any()
            
    def Flat_Residencetime_Prior(self, x):
        if np.any(x < 0) or np.any(x > 0.5):
            return -np.inf
        else:
            return 0
    
    # Priors, Likelihood, and Posterior
    def Ln_Priors(self, theta):
        # Quick Check
        #if any(self.is_negative_or_zero(param) for param in theta):
        #    return - np.inf
            
        Fish = self.FishCanyonAge(theta)/1e6 # Ma
        # Think this is fair and doesnt ever really apply only in initialization
        # its possible to make FCs way too old or young so this just keeps us some
        # what constrained to reality.
        if Fish < 28.0 or Fish > 29.0:
            return -np.inf
        
        # Unpack theta
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        
        # "Corrected" Ages
        Ar_Model_Ages_Corr = self.ArAr_Model_Age_with_all_correction(theta)
        U238_Model_Ages_Corr = self.U238_Model_Ages_Corr(theta)
        U235_Model_Ages_Corr = self.U235_Model_Ages_Corr(theta)
        
        
        # "Uncorrected" Ages
        ArAr_Ages = self.ArAr_Age_Model(theta)/1e6
        U235_Ages = self.U235_Age_Model(theta)/1e6
        U238_Ages = self.U238_Age_Model(theta)/1e6
        """
        Priors
        """
        lp = 0
        
        """
        40K
        """
        lp += norm.logpdf(Activity_electron_capture,
                          self.K_act_EC,
                          self.K_act_EC_err)
                          
        lp += norm.logpdf(Activity_beta,
                         self.K_act_Beta,
                         self.K_act_Beta_err)
                         
        lp += norm.logpdf(beta_plus_beta_minus,
                         self.K_act_Positron,
                         self.K_act_Positron_err)
                         
        lp += norm.logpdf(kdk_ratio,
                         self.K_act_EC_ground,
                         self.K_act_EC_ground_err)
                         
        """
        U- decay constants
        """
        lp += norm.logpdf(U_238_lam,
                         self.U238_decay_const,
                         self.U238_decay_const_err)
                         
        lp += norm.logpdf(U_235_lam,
                         self.U235_decay_const,
                         self.U235_decay_const_err)
                         
                         
        """
        Age Constraints
        """
        lp += np.sum(uniform.logpdf(Ar_Model_Ages_Corr, 0, 4600))
        
        lp += np.sum(uniform.logpdf(U235_Model_Ages_Corr, 0, 4600))

        lp += np.sum(uniform.logpdf(U238_Model_Ages_Corr , 0, 4600))

        """
        K isotopic Composition
        """
        lp += np.sum(norm.logpdf(K40_K_sample, self.K40K_silicate,
                                    self.K40K_silicate_err))
        lp += norm.logpdf(K40_K_decayconst, self.K40K_naumenko,
                        self.K40K_naumenko_err)
        lp += norm.logpdf(K40_K_fcs, self.K40K_FCs_Morgan,
                            self.K40K_FCs_Morgan_err)
        
        """
        Katmw
        """
        lp += norm.logpdf(K_atmw, self.Katmw, self.Katmw_err)
        
        """
        Age Perturbation
        Parameters
        ------------------
        - Things we dont know but accounting for phenomena that we do know can
        perturb an age. We center them on zero because I think it is reasonable
        and a number of phenomena likely cancel out
        but these would probably better as a student t distribution.
        see code below. Would select degrees of freedom df = 3 as a compromise
        - Allow for "heavier" tails to catch more extreme phenomena than is
        probabilistically described by the normal dist.
        ----------------------------------
        from scipy.stats import t
        lp += np.sum(t.logpdf(Ar_agediff, df = 3,  0, ArAr_Ages/100))
        lp += np.sum(t.logpdf(U238_ext, df = 3,  0, U238_Ages[2:]/100))
        lp += np.sum(t.logpdf(U235_ext, df = 3,  0, U235_Ages[3:]/100))
        """
        lp += np.sum(norm.logpdf(Ar_agediff, 0, ArAr_Ages/100))
        lp += np.sum(norm.logpdf(U235_ext, 0, U235_Ages[3:]/100))
        lp += np.sum(norm.logpdf(U238_ext, 0, U238_Ages[2:]/100))
        
        """
        Residence Time
        """
        lp += self.Flat_Residencetime_Prior(residence_time)
        
        """
        kappa value for FCs
        """
        lp += norm.logpdf(F_value, self.FCS_kappa,
                            self.FCS_kappa_err)
        
        """
        DATA
        -----
        - R-values relating Fish Canyon to a sample
        - U68 ratios
        - U75 ratios
        """
        lp += np.sum(norm.logpdf(R_values, loc = self.R_values,
                                    scale = self.R_values_err))
        lp += np.sum(norm.logpdf(UPb_ratios[2:], loc = self.u68_ratios[2:],
                            scale = self.u68_ratios_err[2:]))
        lp += np.sum(norm.logpdf(UPb_7_ratios[3:], loc = self.u75_ratios[3:],
                                scale = self.u75_ratios_err[3:]))
        
        """
        Vesuvius
        """
        lp += self.Vesuvius_prior1(Ar_Model_Ages_Corr[0]) # The real control on vesuvius here! Likelihood is weak in comparison
        lp += self.Vesuvius_prior2(Ar_Model_Ages_Corr[1])
        return lp


    def Normal_loglike_quick(self, model_Ar, model_U, error):
        delta = model_U -  model_Ar
        ll = - 0.5 * np.square(delta / error)
        return ll

    def Ln_Likelihood(self, theta):
        # Unpack theta
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        Ar_Model_Ages_Corr = self.ArAr_Model_Age_with_all_correction(theta)
        U238_Model_Ages_Corr = self.U238_Model_Ages_Corr(theta)
        U235_Model_Ages_Corr = self.U235_Model_Ages_Corr(theta) # Everything in Ma
        
        # Uranium - Lead
        ll_U8 = self.Normal_loglike_quick(Ar_Model_Ages_Corr[2:],
                                          U238_Model_Ages_Corr,
                                                  self.U68_error[2:]) # UPb uncertainties in Ma
        
        
        ll_U5 = self.Normal_loglike_quick(Ar_Model_Ages_Corr[3:],
                                          U235_Model_Ages_Corr,
                                                  self.U75_error[3:]) # UPb uncertainties in Ma
                                                  
        
        # Vesuvius
        LL_V1 = self.Normal_loglike_quick(self.Vesuvius_Age1, Ar_Model_Ages_Corr[0],
                                        self.Vesuvius_Age1_err)
                                        
        LL_V2 = self.Normal_loglike_quick(self.Vesuvius_Age2,Ar_Model_Ages_Corr[1],
                                        self.Vesuvius_Age2_err)
                                        
        # Summed Likelihood of everything!
        ll_tot = np.sum(ll_U5) + np.sum(ll_U8) + LL_V1 + LL_V2
        return ll_tot
    
        
    def Ln_Posterior(self, theta):
        lp = self.Ln_Priors(theta)
        if not np.isfinite(lp):
            return - np.inf
        post = lp + self.Ln_Likelihood(theta) # Log form of reduced Bayes' Rule.
        return post
    
    
    def perturb_parameter_scalar(self, current_value, tuning_factor, temperature):
        
        step_size= np.random.normal(0, tuning_factor * temperature * abs(current_value))
        new_value = current_value + step_size
        
        return new_value
        
    def perturb_parameter_array(self, current_array, tuning_factor, index, temperature):
    
        new_array = np.copy(current_array)
        step_size= np.random.normal(0, tuning_factor * temperature * abs(current_array[index]))
        new_array[index] = current_array[index] + step_size
        return new_array
        
    """
    All MCMC Moves
    --------------
    First Scalar Moves
    Second Vector Moves
    """
    
    def U238_decay_Move(self, theta, tuning_factor, temperature):
    
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

        U_238_lam_prime = self.perturb_parameter_scalar(U_238_lam,
                          tuning_factor, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture, \
                        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
                        beta_plus_beta_minus, kdk_ratio,\
                        F_value, Ar_agediff, \
                        U238_ext, U235_ext, residence_time,
                        R_values,  UPb_ratios, U_238_lam_prime, \
                        UPb_7_ratios, U_235_lam )
                        
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
        
        if log_posterior_proposal == -np.inf:
            return U_238_lam, False
        
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return U_238_lam_prime, True
        else:
            return U_238_lam, False


    def U235_decay_Move(self, theta, tuning_factor, temperature):
    
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

        U_235_lam_prime = self.perturb_parameter_scalar(U_235_lam,
                          tuning_factor, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture, \
                        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
                        beta_plus_beta_minus, kdk_ratio,\
                        F_value, Ar_agediff, \
                        U238_ext, U235_ext, residence_time,
                        R_values,  UPb_ratios, U_238_lam, \
                        UPb_7_ratios, U_235_lam_prime )
                        
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
        
        if log_posterior_proposal == -np.inf:
            return U_235_lam, False
        
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return U_235_lam_prime, True
        else:
            return U_235_lam, False
            
            
    def K40K_decayconst_Move(self, theta, tuning_factor, temperature):
    
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

        K40_K_decayconst_prime = self.perturb_parameter_scalar(K40_K_decayconst,
                      tuning_factor, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture, \
                        Avagadro, K40_K_decayconst_prime, K40_K_sample, K40_K_fcs, K_atmw, \
                        beta_plus_beta_minus, kdk_ratio,\
                        F_value, Ar_agediff, \
                        U238_ext, U235_ext, residence_time,
                        R_values,  UPb_ratios, U_238_lam, \
                        UPb_7_ratios, U_235_lam)
                        
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
        
        if log_posterior_proposal == -np.inf:
            return K40_K_decayconst, False
        
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return K40_K_decayconst_prime, True
        else:
            return K40_K_decayconst, False



    def K40K_FCs_Move(self, theta, tuning_factor, temperature):
    
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

        K40_K_fcs_prime = self.perturb_parameter_scalar(K40_K_fcs,
                      tuning_factor, temperature)
        
        theta_prime = (    Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs_prime, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time,
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)

        log_posterior_proposal = self.Ln_Posterior(theta_prime)
            
        if log_posterior_proposal == -np.inf:
            return K40_K_fcs, False
        
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return K40_K_fcs_prime, True
    
        else:
            return K40_K_fcs, False
            

    def K_atmw_move(self, theta, tuning_factor, temperature):

        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

        K_atmw_prime = self.perturb_parameter_scalar(K_atmw,
                          tuning_factor, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw_prime, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,\
        U238_ext, U235_ext, residence_time,
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
    
        if log_posterior_proposal == -np.inf:
            return K_atmw, False

        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return K_atmw_prime, True
    
        else:
            return K_atmw, False
            
    
    def Activity_Beta_move(self, theta, tuning_factor, temperature):
    
 
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
    
        
        log_posterior_current = self.Ln_Posterior(theta)
    
        Activity_beta_prime = self.perturb_parameter_scalar(Activity_beta,
                        tuning_factor, temperature)
        
        theta_prime = (Activity_beta_prime, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time,
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
    
        if log_posterior_proposal == -np.inf:
            return Activity_beta, False


        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return Activity_beta_prime, True
    
        else:
            return Activity_beta, False
            
            
    def Activity_EC_move(self, theta, tuning_factor, temperature):
 
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)
    
        Activity_electron_capture_prime = self.perturb_parameter_scalar(Activity_electron_capture,
                        tuning_factor, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture_prime, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,\
        U238_ext, U235_ext, residence_time,
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
    
        if log_posterior_proposal == -np.inf:
            return Activity_electron_capture, False
    
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return Activity_electron_capture_prime, True
    
        else:
            return Activity_electron_capture, False


    def Avagadro_move(self, theta, tuning_factor, temperature):

        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

        Avagadro_prime = self.perturb_parameter_scalar(Avagadro,
                          tuning_factor, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture, \
        Avagadro_prime, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return Avagadro, False

        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return Avagadro_prime, True
    
        else:
            return Avagadro, False


    def beta_plus_beta_minus_move(self, theta, tuning_factor, temperature):
 
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)
    
    
        beta_plus_beta_minus_prime = self.perturb_parameter_scalar(beta_plus_beta_minus,
                        tuning_factor, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus_prime, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time,
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
    
        if log_posterior_proposal == -np.inf:
            return beta_plus_beta_minus, False
    
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return beta_plus_beta_minus_prime, True
    
        else:
            return beta_plus_beta_minus, False

    def kdk_ratio_move(self, theta, tuning_factor, temperature):
 
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)


        kdk_ratio_prime = self.perturb_parameter_scalar(kdk_ratio,
                          tuning_factor, temperature)
        
        theta_prime = ( Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio_prime,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return kdk_ratio, False

        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return kdk_ratio_prime, True
    
        else:
            return kdk_ratio, False


    def FCs_move(self, theta, tuning_factor, temperature):
 
        # Current state
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)
    
    
        F_value_prime = self.perturb_parameter_scalar(F_value,
                        tuning_factor, temperature)
        
        theta_prime = ( Activity_beta, Activity_electron_capture, \
                        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
                        beta_plus_beta_minus, kdk_ratio,\
                        F_value_prime, Ar_agediff, \
                        U238_ext, U235_ext, residence_time,
                        R_values,  UPb_ratios, U_238_lam, \
                        UPb_7_ratios, U_235_lam)
            
                
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
    
        if log_posterior_proposal == -np.inf:
            return F_value, False

        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return F_value_prime, True
    
        else:
            return F_value, False
            
            
    """
    Vector Moves
    =============
    """
    
    def U235_Pb207_Move(self, theta, tuning_factor, index, temperature):
 
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)
    
    
        UPb_7_ratios_prime = self.perturb_parameter_array(UPb_7_ratios,
                        tuning_factor, index, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture, \
                        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
                        beta_plus_beta_minus, kdk_ratio,\
                        F_value, Ar_agediff,  \
                        U238_ext, U235_ext, residence_time,
                        R_values,  UPb_ratios, U_238_lam, \
                        UPb_7_ratios_prime, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)
    
        
        if log_posterior_proposal == -np.inf:
            return UPb_7_ratios, False
            
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return UPb_7_ratios_prime, True
    
        else:
            return UPb_7_ratios, False
            
            
    def U238_Pb206_Move(self, theta, tuning_factor, index, temperature):
 
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

        UPb_ratios_prime = self.perturb_parameter_array(UPb_ratios,
                      tuning_factor, index, temperature)
    
        theta_prime = ( Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time,
        R_values,  UPb_ratios_prime, U_238_lam, \
        UPb_7_ratios, U_235_lam)
        
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return UPb_ratios, False
            
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return UPb_ratios_prime, True
    
        else:
            return UPb_ratios, False
            
    def K40_K_sample_move(self, theta, tuning_factor, index, temperature):

        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)


        K40_K_sample_prime = self.perturb_parameter_array(K40_K_sample,
                          tuning_factor, index, temperature)
        
        theta_prime = (    Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample_prime, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time,
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)

        
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return K40_K_sample, False
            
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return K40_K_sample_prime, True
    
        else:
            return K40_K_sample, False
            
            
            
    def U238_ext_move(self, theta, tuning_factor, index, temperature):

        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

        U238_ext_prime = np.copy(U238_ext)

        
        U238_ext_prime[index] += np.random.normal(0,
                                                    temperature * tuning_factor)

        theta_prime = (Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext_prime, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return U238_ext, False
            
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return U238_ext_prime, True
    
        else:
            return U238_ext, False
            
    def U235_ext_move(self, theta, tuning_factor, index, temperature):

        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)


        U235_ext_prime = np.copy(U235_ext)
    
        
        U235_ext_prime[index] += np.random.normal(0,
                                                    temperature * tuning_factor)

        theta_prime = (Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,\
        U238_ext, U235_ext_prime, residence_time,
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return U235_ext, False
            
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return U235_ext_prime, True
    
        else:
            return U235_ext, False
            

    def Ar_ext_move(self, theta, tuning_factor, index, temperature):

        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
            
        log_posterior_current = self.Ln_Posterior(theta)

        Ar_agediff_prime = np.copy(Ar_agediff)
    
        
        Ar_agediff_prime[index] += np.random.normal(0,
                                                    temperature * tuning_factor)

        theta_prime = (Activity_beta, Activity_electron_capture, \
                        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
                        beta_plus_beta_minus, kdk_ratio,\
                        F_value, Ar_agediff_prime, \
                        U238_ext, U235_ext, residence_time, \
                        R_values,  UPb_ratios, U_238_lam, \
                        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return Ar_agediff, False
            
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return Ar_agediff_prime, True
    
        else:
            return Ar_agediff, False
            
    def residence_time_move(self, theta, tuning_factor, index, temperature):
    
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
    
        log_posterior_current = self.Ln_Posterior(theta)


        residence_time_prime = np.copy(residence_time)

        
        residence_time_prime[index] += np.random.normal(0,
                                                    temperature * tuning_factor)
    
        theta_prime = (Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time_prime, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return residence_time, False
    
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return residence_time_prime, True
    
        else:
            return residence_time, False


    def R_value_move(self, theta, tuning_factor, index, temperature):
    

        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        log_posterior_current = self.Ln_Posterior(theta)

    
        R_values_prime = self.perturb_parameter_array(R_values,
                        tuning_factor, index, temperature)
        
        theta_prime = (Activity_beta, Activity_electron_capture, \
                    Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
                    beta_plus_beta_minus, kdk_ratio,\
                    F_value, Ar_agediff, \
                    U238_ext, U235_ext, residence_time, \
                    R_values_prime,  UPb_ratios, U_238_lam, \
                    UPb_7_ratios, U_235_lam)
            
        log_posterior_proposal = self.Ln_Posterior(theta_prime)

        if log_posterior_proposal == -np.inf:
            return R_values, False
            
        acceptance = (log_posterior_proposal - log_posterior_current) / temperature
        
        u = np.random.rand()
        if np.log(u) < acceptance:
            return R_values_prime, True
    
        else:
            return R_values, False
    

    def save_tuning_parameters(self, tuning_factors, chain_id):
        file_name = f'tuning_parameters_{chain_id}.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(tuning_factors, file)
        print(f"Tuning parameters saved to {file_name}")

    def update_scalar_parameter(self, theta, new_value, move_name):
        """
        Update the theta tuple based on the move made.
        """
        # Unpack the existing theta
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff, \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta

        # Update scalar parameters
        if move_name == 'Activity_beta_Z':
            Activity_beta = new_value
        elif move_name == 'Activity_electron_capture_Z':
            Activity_electron_capture = new_value
        elif move_name == 'Avagadro_Z':
            Avagadro = new_value
        elif move_name == 'K40_K_decayconst_Z':
            K40_K_decayconst = new_value
        elif move_name == 'K40_K_fcs_Z':
            K40_K_fcs = new_value
        elif move_name == 'K_atmw_Z':
            K_atmw = new_value
        elif move_name == 'beta_plus_beta_minus_Z':
            beta_plus_beta_minus = new_value
        elif move_name == 'kdk_ratio_Z':
            kdk_ratio = new_value
        elif move_name == 'F_value_Z':
            F_value = new_value
        elif move_name == 'U_238_lam_Z':
            U_238_lam = new_value
        elif move_name == 'U_235_lam_Z':
            U_235_lam = new_value

        # Return the updated theta
        return (Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)


    def update_vector_parameters(self, theta,
                                new_value,
                                move_name,
                                index):
        """
        Update the theta tuple based on the move made.
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta

            
        if move_name == 'U238_ext_Z':
            U238_ext[index] = new_value

        elif move_name == 'U235_ext_Z':
            U235_ext[index] = new_value
            
        elif move_name == 'residence_time_Z':
            residence_time[index] = new_value

        elif move_name == 'Ar_agediff_Z':
            Ar_agediff[index] = new_value

        elif move_name == 'K40_K_sample_Z':
            K40_K_sample[index] = new_value

        elif move_name == 'R_values_Z':
            R_values[index] = new_value
            
        elif move_name == 'UPb_ratios_Z':
            UPb_ratios[index] = new_value
            
        elif move_name == 'UPb_7_ratios_Z':
            UPb_7_ratios[index] = new_value
        
        return (Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam)
        
        
    def Print_Current_Model_Place(self):
        """
        Helper function to give a suite of information as to where the current chain is
         - Temperature
         - Total Iteration
         - Posterior of all Chains
         - This is before running your next batch
        """
        temperature_file  = f'{self.Run_Name}_Temp_BayesCal_2025_0.pkl'
        # Select a temp file
        if os.path.exists(temperature_file) and self.Start_from_pickles:
            with open(temperature_file, 'rb') as f:
                temp_state = pickle.load(f)
        else:
            temp_state = {"temperature": 1}

        iters_file = f'{self.Run_Name}_Iters_BayesCal_2025_0.pkl'
        # Select an iters file
        if os.path.exists(iters_file) and self.Start_from_pickles: # If the file exists and you want to Start from Pickles (e.g., start from the latest sample you have so you can continue
            with open(iters_file, 'rb') as f:
                iters_state = pickle.load(f)
        else:
            iters_state = {"iters" :1}

        total_iters = iters_state["iters"]
        
        thetas_latest = self.check_starting_parameters()
        for i in range(self.nchains):
            print("Log Posterior : ", self.Ln_Posterior(thetas_latest[i]))
        print("Temperature: ", temp_state["temperature"])
        print("Total Iterations so far: ", total_iters)


    def MCMC(self, theta, iters, chain_id):
        """
        Markov Chain Monte Carlo with Adaptive Tuning
        """
        # Time
        start_time = time.time()
        
        """
        Theta
        """
        Activity_beta, Activity_electron_capture, \
        Avagadro, K40_K_decayconst, K40_K_sample, K40_K_fcs, K_atmw, \
        beta_plus_beta_minus, kdk_ratio,\
        F_value, Ar_agediff,  \
        U238_ext, U235_ext, residence_time, \
        R_values,  UPb_ratios, U_238_lam, \
        UPb_7_ratios, U_235_lam = theta
        
        # Path to the tuning parameters file
        tuning_factors_file = f'{self.Run_Name}_tuning_factors_BayesCal_2025_{chain_id}.pkl'
        #posterior_filename = generate_filename('posterior_bayes', chain_id)


        # Check if tuning parameters file exists and load it
        if os.path.exists(tuning_factors_file) and self.Start_from_pickles:
            with open(tuning_factors_file, 'rb') as f:
                tuning_factors = pickle.load(f)
        else:
            # Initialize tuning factors for each parameter - just a guess here these get adapted throughout the MCMC
            tuning_factors = {
                'Activity_beta_Z': 0.01,
                'Activity_electron_capture_Z': 0.01,
                'K40_K_decayconst_Z': 0.0001,
                 'K40_K_fcs_Z': 0.0001,
                 'K_atmw_Z': 0.00001,
                  'Avagadro_Z': 1e-5,
                'beta_plus_beta_minus_Z': 0.01,
                'kdk_ratio_Z': 0.01,
                'F_value_Z': 0.0001,
                'U_238_lam_Z': 0.0001,
                'U_235_lam_Z': 0.001}
            
            for i in range(self.N_data):
                tuning_factors[f'UPb_7_ratios_Z_{i}'] = 0.0001
                tuning_factors[f'UPb_ratios_Z_{i}'] = 0.0001
                tuning_factors[f'R_values_Z_{i}'] = 0.0001
                tuning_factors[f'K40_K_sample_Z_{i}'] = 0.0001
                tuning_factors[f'Ar_agediff_Z_{i}'] = 0.0001
                
            for i in range(self.N_data - 2):
                tuning_factors[f'U235_ext_Z_{i}'] = 0.0001
                tuning_factors[f'U238_ext_Z_{i}'] = 0.0001
                tuning_factors[f'residence_time_Z_{i}'] = 0.01
        
        
        model_ages = self.ArAr_Age_Model(theta)
        """
        Storage
        -------
        A bit of overkill but I want to save everything
        """
        niters = iters
        Ages_store = np.zeros((niters, self.N_data))
        Ages_store_corr = np.zeros((niters, self.N_data))
        Ages_238_store = np.zeros((niters, self.N_data - 2))
        Ages_235_store = np.zeros((niters, self.N_data - 3))
        Ages_238_store_corr = np.zeros((niters, self.N_data - 2))
        Ages_235_store_corr = np.zeros((niters, self.N_data - 3))
        lambda_ec_store = np.zeros(niters)
        lambda_ec0_store = np.zeros(niters)
        lambda_ca_store = np.zeros(niters)
        lambda_bplus_store = np.zeros(niters)
        Avagadro_store = np.zeros(niters)
        U_238_lam_store = np.zeros(niters)
        U_235_lam_store = np.zeros(niters)
        FCs_store = np.zeros(niters)
        FCs_age = np.zeros(niters)
        lambda_total_store = np.zeros(niters)
        posterior_store = np.zeros(niters)
        K40_K_decayconst_store = np.zeros(niters)
        K40_K_sample_store = np.zeros((niters, self.N_data))
        K40_K_fcs_store = np.zeros(niters)
        K_atmw_store = np.zeros(niters)
        R_values_store = np.zeros((niters, self.N_data))
        UPb_ratios_store = np.zeros((niters, self.N_data - 2))
        UPb_7_ratios_store = np.zeros((niters, self.N_data - 3))

        """
        Age differences Stores
        """
        residence_time_store = np.zeros((niters, self.N_data - 2))
        Ar_agediff_store = np.zeros((niters, self.N_data))
        U8_ext_store = np.zeros((niters, self.N_data - 2))
        U5_ext_store = np.zeros((niters, self.N_data - 3))

        """
        Initial Theta Values
        """
        Ages_store[0] = self.ArAr_Age_Model(theta)/1e6 # Going to store in Ma
        Ages_store_corr[0] = self.ArAr_Age_Model(theta)/1e6 + Ar_agediff # Age corrected for perturbation factor
        Ages_238_store[0] = self.U238_Age_Model(theta)[2:]/1e6
        Ages_235_store[0] = self.U235_Age_Model(theta)[3:]/1e6
        Ages_238_store_corr[0] = self.U238_Age_Model(theta)[2:]/1e6 - residence_time  + U238_ext
        Ages_235_store_corr[0] =self.U235_Age_Model(theta)[3:]/1e6 -  residence_time[1:] + U235_ext
        Avagadro_store[0] = Avagadro
        lambda_ca_store[0], lambda_ec_store[0], lambda_ec0_store[0], lambda_bplus_store[0] = self.Partial_Decay_Constants(theta)
        FCs_store[0] = F_value
        FCs_age[0] = self.FishCanyonAge(theta)
        lambda_total_store[0] = self.Total_Decay_Constant(theta)
        posterior_store[0] = self.Ln_Posterior(theta)
        K40_K_decayconst_store[0] = K40_K_decayconst
        K40_K_sample_store[0] = K40_K_sample
        K40_K_fcs_store[0] = K40_K_fcs
        K_atmw_store[0] = K_atmw
        R_values_store[0] = R_values
        U_238_lam_store[0] = U_238_lam
        U_235_lam_store[0] = U_235_lam
        UPb_ratios_store[0] = UPb_ratios[2:]
        UPb_7_ratios_store[0] = UPb_7_ratios[3:]
        Ar_agediff_store[0]  = Ar_agediff
        residence_time_store[0]  = residence_time
        U8_ext_store[0] = U238_ext
        U5_ext_store[0] = U235_ext

        """
        Adaptation Rate
        --------------
        We try to get an acceptance rate of 23.4% (Rosenthal, 2011)
        Most likely to be the "best" estimate in this type of model approach
        """
        target_accept_rate = 0.234 # Rosenthal, 2011
        # We also need counters for each param
        proposal_counts = {p: 0 for p in tuning_factors}
        accept_counts   = {p: 0 for p in tuning_factors}
        # For an EMA of acceptance, store a running average as well:
        ema_accept_rate = {p: 0.0 for p in tuning_factors}   # or start at 0.0
        ema_alpha = 0.1

            
        temperature_file  = f'{self.Run_Name}_Temp_BayesCal_2025_{chain_id}.pkl'
        if os.path.exists(temperature_file) and self.Start_from_pickles:
            with open(temperature_file, 'rb') as f:
                temp_state = pickle.load(f)
        else:
            temp_state = {"temperature": 1}

        iters_file = f'{self.Run_Name}_Iters_BayesCal_2025_{chain_id}.pkl'
        if os.path.exists(iters_file) and self.Start_from_pickles: # If the file exists and you want to Start from Pickles (e.g., start from the latest sample you have so you can continue
            with open(iters_file, 'rb') as f:
                iters_state = pickle.load(f)
        else:
            iters_state = {"iters" :1}

        total_iters = iters_state["iters"] + niters # Total number of iterations is needed to anneal - a bit of a fudge here but I think its not too bad. Just need to keep track!
        with open(iters_file, 'wb') as f:
            iters_state = {"iters": total_iters}
            pickle.dump(iters_state, f) # dump these here! If Chain crashes chance of over estimating the number of iterations but, should be reasonable.
        
        # Define all move functions in a list
        move_functions = [
            ('Activity_beta_Z', self.Activity_Beta_move),
            ('Activity_electron_capture_Z', self.Activity_EC_move),
            ('beta_plus_beta_minus_Z', self.beta_plus_beta_minus_move),
            ('kdk_ratio_Z', self.kdk_ratio_move),
            ('Avagadro_Z', self.Avagadro_move),
            ('F_value_Z', self.FCs_move),
            ('K40_K_decayconst_Z', self.K40K_decayconst_Move),
            ('K40_K_fcs_Z', self.K40K_FCs_Move),
            ('K40_K_sample_Z', self.K40_K_sample_move),
            ('K_atmw_Z',  self.K_atmw_move),
            ('UPb_ratios_Z', self.U238_Pb206_Move),
            ('UPb_7_ratios_Z', self.U235_Pb207_Move),
            ('R_values_Z',  self.R_value_move),
            ('U_238_lam_Z', self.U238_decay_Move),
            ('U_235_lam_Z', self.U235_decay_Move),
            ('U238_ext_Z', self.U238_ext_move),
            ('U235_ext_Z', self.U235_ext_move),
            ('residence_time_Z', self.residence_time_move),
            ('Ar_agediff_Z', self.Ar_ext_move)
        ]


        # Open posterior file for appending outside the loop
        tau = 1e10 # My annealing parameter (It takes a long time!!! but I had time so I wanted to be careful to explore all the parameter space well)
        Temp_initial = temp_state['temperature'] # Initial temp (if you have the pickle file it will start here otherwise will be 1)
         
        # Current total number of iterations is this
        # We save with niters added so we start from the right place
        # So subract here and then begin the thermal annealing here
        # Its a little fudgy but seems to work so no fixing.
        current_total_iters = total_iters - niters
        current_total_iters = np.maximum(current_total_iters, 0)
        for i in range(1, niters):
            temperature = Temp_initial * np.exp( -(i + current_total_iters)/ tau) # Function for annealing the Chains - We peal off more and more and more over time (relative to the total number of iterations. So we peal off to identify the maximum posterior location. Helps with the identification of a unique solution.
            if temperature < 0.05: # We set a minimum value for this so we can stop the annealing here and
                temperature = 0.05 # This is chosen from testing - I think this is the place to stop otherwise we move into territory where its probably overly precise (Have to keep in mind the review here - We dont want to overlook unaccounted for random or systematic uncertainties that are not account for or under represented in the model)
                
            move_name, move_func = random.choice(move_functions)
        
            if move_name in ['R_values_Z',
                            'K40_K_sample_Z',
                            'Ar_agediff_Z']:

                index = np.random.randint(0, self.N_data)
                specific_counter_name = f'{move_name}_{index}'
                proposal_counts[specific_counter_name] += 1
                new_value, accepted = move_func(theta,
                                    tuning_factors[specific_counter_name],
                                        index, temperature)
        
                if accepted:
                    accept_counts[specific_counter_name] +=1
                    new_theta = self.update_vector_parameters(theta,
                                             new_value[index],  # Pass only the single updated value
                                             move_name,  # move_name
                                             index)
                    theta = new_theta
                    
            elif move_name in ['UPb_ratios_Z',
                             'UPb_7_ratios_Z']:

                index = np.random.randint(0, self.N_data)
                
                
                specific_counter_name = f'{move_name}_{index}'
                proposal_counts[specific_counter_name] += 1
                new_value, accepted = move_func(theta,
                                    tuning_factors[specific_counter_name],
                                        index, temperature)
        
                if accepted:
                    accept_counts[specific_counter_name] +=1
                    new_theta = self.update_vector_parameters(theta,
                                             new_value[index],  # Pass only the single updated value
                                             move_name,  # move_name
                                             index)
                    theta = new_theta
                    
            elif move_name in ['U238_ext_Z',
                               'residence_time_Z']:
                index = np.random.randint(0, len(residence_time))
                
                specific_counter_name = f'{move_name}_{index}'
                proposal_counts[specific_counter_name] += 1
                new_value, accepted = move_func(theta,
                                    tuning_factors[specific_counter_name],
                                        index, temperature)
        
                if accepted:
                    accept_counts[specific_counter_name] +=1
                    new_theta = self.update_vector_parameters(theta,
                                             new_value[index],  # Pass only the single updated value
                                             move_name,  # move_name
                                             index)
                    theta = new_theta
                    

            elif move_name in ['U235_ext_Z']:
                
                index = np.random.randint(0, len(U235_ext))
                
                specific_counter_name = f'{move_name}_{index}'
                proposal_counts[specific_counter_name] += 1
                new_value, accepted = move_func(theta,
                                    tuning_factors[specific_counter_name],
                                        index, temperature)
        
                if accepted:
                    accept_counts[specific_counter_name] +=1
                    new_theta = self.update_vector_parameters(theta,
                                             new_value[index],  # Pass only the single updated value
                                             move_name,  # move_name
                                             index)
                    theta = new_theta
            
            else:
                specific_counter_name = f'{move_name}'
                proposal_counts[specific_counter_name] += 1
                new_value, accepted = move_func(theta,
                                    tuning_factors[specific_counter_name],
                                               temperature)
        
                if accepted:
                    accept_counts[specific_counter_name] +=1
                    new_theta = self.update_scalar_parameter(theta,
                                                    new_value,
                                                    move_name)
                    theta = new_theta

            Ages_store[i,:] = self.ArAr_Age_Model(theta)/1e6
            Ages_store_corr[i,:] = self.ArAr_Age_Model(theta)/1e6 - theta[10]
            Ages_238_store[i,:] = self.U238_Age_Model(theta)[2:]/1e6
            Ages_235_store[i,:] = self.U235_Age_Model(theta)[3:]/1e6
            Ages_238_store_corr[i,:] = self.U238_Age_Model(theta)[2:]/1e6 - theta[13]   + theta[11]
            Ages_235_store_corr[i,:] = self.U235_Age_Model(theta)[3:]/1e6 - theta[13][1:] + theta[12]
            Ar_agediff_store[i,:] = theta[10]
            R_values_store[i,:] = theta[14]
            UPb_ratios_store[i,:] = theta[15][2:]
            UPb_7_ratios_store[i,:] = theta[17][3:]
            lambda_ca_store[i], lambda_ec_store[i], lambda_ec0_store[i], lambda_bplus_store[i] = self.Partial_Decay_Constants(theta)
            U_238_lam_store[i] = theta[16]
            U_235_lam_store[i] = theta[18]
            FCs_store[i] =theta[9]
            K40_K_decayconst_store[i] = theta[3]
            K40_K_sample_store[i, :] = theta[4]
            K40_K_fcs_store[i] = theta[5]
            K_atmw_store[i] = theta[6]
            lambda_total_store[i] = self.Total_Decay_Constant(theta)
            FCs_age[i] = self.FishCanyonAge(theta)
            posterior_store[i] = self.Ln_Posterior(theta)
            Avagadro_store[i] = theta[2]
            residence_time_store[i,:] = theta[13]
            U8_ext_store[i,:] = theta[11]
            U5_ext_store[i,:] = theta[12]


            # ---------------------------------------------------------
            #  3) Periodic adaptation of the tuning factor
            # ---------------------------------------------------------
            # E.g. adapt every some number iterations after some burn-in
            # Vary this number to change the rate of adaptibility
            # Probably a fancier way to do this so that its tunable
            # maybe ~ based on how rapid the changes are and maybe directionality of the
            # changes.
            # Keep tuning till we get to the temperature threshold then keep going and once we reach a reasonable place sample from the stationary distribution
            if (i > 50) and (i % 15000 == 0) and temperature > 0.05 and total_iters > 100e7:
                # Note for future idea
                """
                Once we reach the desired temperature we want to get all parameters to the wanted 23.4% so maybe run until this point? with a few percent tolerence. Look at the block_accept_rate, if all are within 20 - 30% stop tuning and then take 5 million samples from the stationary Markov chain.
                """
                for param in tuning_factors.keys():
                    # If not proposed at all in the last block, skip
                    if proposal_counts[param] == 0:
                        continue
            
                    # 3a) Compute acceptance rate for this block
                    # Doing a block_accept_rate check might be hange
                    block_accept_rate = accept_counts[param] / proposal_counts[param]
                    
                    if block_accept_rate < 0.234:
                    # Simple updating tuning factor step for every parameter in the model
                    # Then could be down in a dampening fashion - proportional to the total iters or something like that but, I think fine as is
                        tuning_factors[param] *= 0.95
                    else:
                        tuning_factors[param] *= 1.05
                    
                    
                    # 3e) Reset counters for next block
                    accept_counts[param] = 0
                    proposal_counts[param] = 0


            """
            Pickle Every selected batch of sample
            """
            if (i + 1) % 1000 == 0:
                #with open(posterior_filename, 'ab') as posterior_file:
                    #pickle.dump(posterior_store[i], posterior_file)
                with open(f'{self.Run_Name}_THETA_BayesCal_2025_{chain_id}.pkl', 'wb') as f:
                    pickle.dump(theta, f)
                with open(tuning_factors_file, 'wb') as f:
                    pickle.dump(tuning_factors, f)
                with open(temperature_file, 'wb') as f:
                    temp_state = {"temperature": temperature}
                    pickle.dump(temp_state, f)

                            
            
        return (Ages_store, lambda_ec_store, lambda_ec0_store, lambda_ca_store, lambda_bplus_store, lambda_total_store, FCs_store, FCs_age, R_values_store, posterior_store, Ar_agediff_store, K40_K_decayconst_store, K_atmw_store, UPb_ratios_store, U_238_lam_store, UPb_7_ratios_store, U_235_lam_store, Ages_238_store, Ages_235_store,Ages_238_store_corr, Ages_235_store_corr, Avagadro_store, theta, K40_K_sample_store, K40_K_fcs_store, Ages_store_corr, residence_time_store, U8_ext_store, U5_ext_store, Ages_238_store_corr, Ages_235_store_corr)


    def check_starting_parameters(self):
        thetas = []
        for chain_id in range(self.nchains):
            theta_pickle_file = f'{self.Run_Name}_THETA_BayesCal_2025_{chain_id}.pkl'
            if os.path.exists(theta_pickle_file) and self.Start_from_pickles:
                print(f'Pickles file exists and Start_from_pickles = {self.Start_from_pickles}')
                with open(theta_pickle_file, 'rb') as f:
                    theta_p = pickle.load(f)
                thetas.append(theta_p)
            else:
                # If no initial file exists, use initial guesses
                if not thetas or self.Start_from_pickles is False:  # Ensure it's only done once if needed
                    thetas = self.Make_Initial_Theta_Guess()
                    
        return thetas

    def Run_MCMC(self):
        iterations = self.iterations
        chain_ids = range(self.nchains)
        all_thetas = self.check_starting_parameters()
        

        def run_chain(theta, chain_id):
            return self.MCMC(theta, self.iterations, chain_id)

        results = Parallel(n_jobs = -1)(delayed(run_chain)(theta,
                                                           chain_id) for
                                        theta, chain_id in zip(all_thetas[:self.nchains],
                                                               chain_ids))

        self.Chain_Results = results

        return self.Chain_Results
        
        
    def save_results_to_pickle(self, res, filename):
        with gzip.open(filename, 'wb')  as f:
            pickle.dump(res, f)
    
    def Run_Batch(self):
        for q in range(self.nbatch):
            all_thetas = self.check_starting_parameters()
            chain_ids = np.arange(self.nchains)
            if (q + 1) % 2 == 1:
                print(f'Pickles_{-1}',
                self.Ln_Posterior(all_thetas[-1]),
                'iteration', q)
            
            def run_chain(theta, chain_id):
                return self.MCMC(theta, self.iterations, chain_id)

            results = Parallel(n_jobs = 1)(delayed(run_chain)(theta,
                                                           chain_id) for
                                        theta, chain_id in zip(all_thetas[:self.nchains],
                                                               chain_ids))

            self.save_results_to_pickle(results, f'{self.Run_Name}_results_after_{q}_iterations.pkl')

        
        
    def Chain_Stats(self, chain_list):
         # Helper function to combine all results from all chains
         # Might not be ideal but I think maybe reasonable
        
        # Calculate means and variances for each chain
        chain_means = np.array([np.mean(chain) for chain in chain_list])
        chain_variances = np.array([np.var(chain, ddof=1) for chain in chain_list])

        # Calculate weights
        weights = 1 / chain_variances

        # Calculate the weighted mean
        weighted_mean = np.average(chain_means, weights=weights)

        # Calculate the between-chain variance
        between_chain_variance = np.var(chain_means, ddof=1)

        # Combine the within-chain, between-chain variances
        total_variance = np.average(chain_variances, weights=weights) + between_chain_variance

        # Calculate the weighted standard deviation
        weighted_std = np.sqrt(total_variance)

        return weighted_mean, weighted_std
        
    
    """
    Get Grouped Results Section
    ----------------------------
    - In this code segment I will set out how to go in
     and find all the stored results and extract them for full
    analysis
    """
    
    def Get_Grouped_Param(self, z_name):
        grouped_param = [[] for _ in range(self.nchains)]
        n_group = self.nbatch

        for i in range(n_group):
            with gzip.open(f'{self.Run_Name}_results_after_{i}_iterations.pkl', 'rb') as f:
                res_all = pickle.load(f)

            z_vars = [f"z{i+1}" for i in range(31)]

            for chain_id in range(1, self.nchains + 1):
                chain_res = res_all[chain_id - 1]
                idx = z_vars.index(z_name)
                grouped_param[chain_id - 1].extend(chain_res[idx])

        return grouped_param
    
    def Get_Grouped_Chain_Results(self):
        """
        Reads all MCMC result files and stitches them together into one list of dictionaries,
        where each dictionary contains all parameters for a single chain.
        """
        result_dicts = [{} for _ in range(self.nchains)]  # Initialize one dict per chain
    
        z_vars = [f"z{i+1}" for i in range(31)]  # All variable names
    
        for i in range(self.nbatch):
            with gzip.open(f'{self.Run_Name}_results_after_{i}_iterations.pkl', 'rb') as f:
                res_all = pickle.load(f)  # res_all[chain_id - 1] = list of 31 arrays
    
            for chain_id in range(1, self.nchains + 1):
                chain_res = res_all[chain_id - 1]
    
                for var_idx, var_name in enumerate(z_vars):
                    key = f"{var_name}_{chain_id}"
    
                    if key not in result_dicts[chain_id - 1]:
                        result_dicts[chain_id - 1][key] = []
    
                    result_dicts[chain_id - 1][key].extend(chain_res[var_idx])
    
        self.Grouped_Chain_Results = result_dicts
        return self.Grouped_Chain_Results
        
    def Ensure_Chain_Results(self):
        # Little helper function to call and make sure you have the results ready to use
        if self.Grouped_Chain_Results is None:
            self.Get_Grouped_Chain_Results()
    
    def Get_Grouped_Logp(self):
        self.Ensure_Chain_Results()
        grouped_logp = [[] for _ in range(self.nchains)]
        result_dicts = self.Get_Grouped_Chain_Results()
        
        # Extract log_p values for each chain and append them to the corresponding list in grouped_logp
        for chain_id in range(1, self.nchains + 1):  # Loop through each chain
            chain_dict = result_dicts[chain_id - 1]  # Get the dictionary for this chain
            log_p_chain = chain_dict[f"z10_{chain_id}"]  # Extract log_p (assuming it is z10)
    
            # Append the log_p values for this chain
            grouped_logp[chain_id - 1].extend(log_p_chain)
    
        return grouped_logp
        
        
    def Get_Grouped_FCs(self):
        grouped_param = [[] for _ in range(self.nchains)]
        result_dicts = self.Get_Grouped_Chain_Results()
        
        # Extract log_p values for each chain and append them to the corresponding list in grouped_logp
        for chain_id in range(1, self.nchains + 1):  # Loop through each chain
            chain_dict = result_dicts[chain_id - 1]  # Get the dictionary for this chain
            param_chain = chain_dict[f"z8_{chain_id}"]  # Extract log_p (assuming it is z10)
    
            # Append the log_p values for this chain
            grouped_param[chain_id - 1].extend(param_chain)
    
        return grouped_param
        
    def Plot_FCs(self):
        self.Ensure_Chain_Results()
        grouped_fcs = self.Get_Grouped_FCs()
        fig, ax = plt.subplots(1,1, figsize = (5,5))

        for i in range(self.nchains):
            ax.hist(np.array(grouped_fcs[i])/1e6,
               label = f'Chain {i + 1}',
               density = True, bins = 25)
    
        ax.set_xlabel('FCs Age (Ma)')
        ax.set_ylabel('Density')
        ax.legend(frameon = True, loc = 4, fontsize = 10, ncol = 2)
        
    def Plot_Posterior(self):
        grouped_logp = self.Get_Grouped_Logp()
        
        """
        Plot of log posterior probability
        """
        fig, ax = plt.subplots(1,1,
                            figsize = (5, 4.6))
        
        for i in range(self.nchains):
            ax.plot(grouped_logp[i],
                label = f'Chain {i + 1}')
            
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log Posterior')
        ax.legend(frameon = False, loc = 4, fontsize = 8, ncol = 2)
            


    def Get_Grouped_KDK_Stats(self):
        # Ensure we have access to grouped results
        self.Ensure_Chain_Results()
        result_dicts = self.Get_Grouped_Chain_Results()
    
        # Prepare containers
        grouped_lam_tot = [[] for _ in range(self.nchains)]
        grouped_lam_ecstar = [[] for _ in range(self.nchains)]
        grouped_lam_ec0 = [[] for _ in range(self.nchains)]
        grouped_lam_beta = [[] for _ in range(self.nchains)]
        grouped_lam_betaplus = [[] for _ in range(self.nchains)]
        grouped_lam_ar = [[] for _ in range(self.nchains)]
        grouped_br = [[] for _ in range(self.nchains)]
    
        for chain_id in range(1, self.nchains + 1):
            chain_dict = result_dicts[chain_id - 1]
    
            # Convert everything to numpy arrays
            ec_star_chain = np.array(chain_dict[f"z2_{chain_id}"])
            ec_0_chain = np.array(chain_dict[f"z3_{chain_id}"])
            beta_chain = np.array(chain_dict[f"z4_{chain_id}"])
            betaplus_chain = np.array(chain_dict[f"z5_{chain_id}"])
            lamtot_chain = np.array(chain_dict[f"z6_{chain_id}"])
    
            # Store arrays
            grouped_lam_tot[chain_id - 1].append(lamtot_chain)
            grouped_lam_ecstar[chain_id - 1].append(ec_star_chain)
            grouped_lam_ec0[chain_id - 1].append(ec_0_chain)
            grouped_lam_beta[chain_id - 1].append(beta_chain)
            grouped_lam_betaplus[chain_id - 1].append(betaplus_chain)
    
            lam_ar = betaplus_chain + ec_0_chain + ec_star_chain
            br = beta_chain / lam_ar
    
            grouped_lam_ar[chain_id - 1].append(lam_ar)
            grouped_br[chain_id - 1].append(br)
    
        # Compute weighted means and errors
        lam_tot_wm, lam_tot_wm_err = self.Chain_Stats(grouped_lam_tot)
        lam_ar_wm, lam_ar_wm_err = self.Chain_Stats(grouped_lam_ar)
        lam_ecstar_wm, lam_ecstar_wm_err = self.Chain_Stats(grouped_lam_ecstar)
        lam_ec0_wm, lam_ec0_wm_err = self.Chain_Stats(grouped_lam_ec0)
        lam_beta_wm, lam_beta_wm_err = self.Chain_Stats(grouped_lam_beta)
        lam_betaplus_wm, lam_betaplus_wm_err = self.Chain_Stats(grouped_lam_betaplus)
    
        # Flatten all arrays for correlation
        lam_tot_all = np.concatenate([np.array(x[0]) for x in grouped_lam_tot])
        lam_ar_all = np.concatenate([np.array(x[0]) for x in grouped_lam_ar])
        lam_beta_all = np.concatenate([np.array(x[0]) for x in grouped_lam_beta])
        lam_betaplus_all = np.concatenate([np.array(x[0]) for x in grouped_lam_betaplus])
        lam_ec_all = np.concatenate([np.array(x[0]) for x in grouped_lam_ecstar])
        lam_ec0_all = np.concatenate([np.array(x[0]) for x in grouped_lam_ec0])
        lam_br_all = np.concatenate([np.array(x[0]) for x in grouped_br])
    
        # Build param dictionary for correlation calculation
        param_dict = {
            'lam_tot': lam_tot_all,
            'lam_ar': lam_ar_all,
            'lam_beta': lam_beta_all,
            'lam_betaplus': lam_betaplus_all,
            'lam_ec': lam_ec_all,
            'lam_ec0': lam_ec0_all,
            'lam_br': lam_br_all
        }
    
        # Compute correlation coefficients
        import itertools
        rho_dict = {}
        for key1, key2 in itertools.combinations(param_dict.keys(), 2):
            corr = np.corrcoef(param_dict[key1], param_dict[key2])[0, 1]
            rho_dict[f"rho_{key1}_{key2}"] = corr
    
        # Convert weighted means to DataFrame
        data = {
            'lam_tot_wm': lam_tot_wm,
            'lam_tot_wm_err': lam_tot_wm_err,
            'lam_ar_wm': lam_ar_wm,
            'lam_ar_wm_err': lam_ar_wm_err,
            'lam_ecstar_wm': lam_ecstar_wm,
            'lam_ecstar_wm_err': lam_ecstar_wm_err,
            'lam_ec0_wm': lam_ec0_wm,
            'lam_ec0_wm_err': lam_ec0_wm_err,
            'lam_beta_wm': lam_beta_wm,
            'lam_beta_wm_err': lam_beta_wm_err,
            'lam_betaplus_wm': lam_betaplus_wm,
            'lam_betaplus_wm_err': lam_betaplus_wm_err
        }
    
        df = pd.DataFrame(data, index=[0])
        rho_df = pd.DataFrame(rho_dict, index=[0])
        final_df = pd.concat([df, rho_df], axis=1)
    
        return final_df
        
    def Get_FCS_kappa_Value(self):
        result_dicts = self.Get_Grouped_Chain_Results()
        chain_id = np.arange(self.nchains)
        grouped_kappa = [[] for _ in range(self.nchains)]
        # Extract log_p values for each chain and append them to the corresponding list in grouped_logp
        for chain_id in range(1, self.nchains + 1):  # Loop through each chain
            chain_dict = result_dicts[chain_id - 1]  # Get the dictionary for this chain
            kappa_chain = chain_dict[f"z7_{chain_id}"]  # Extract log_p (assuming it is z10)

            # Append the log_p values for this chain
            grouped_kappa[chain_id - 1].append(kappa_chain)
            
        return grouped_kappa
    
    def Get_U_decay_constants(self):
        """
        Get_Grouped Uranium Decay Constants
        """
        grouped_u238 = [[] for _ in range(self.nchains)]
        grouped_u235 = [[] for _ in range(self.nchains)]
        result_dicts = self.Get_Grouped_Chain_Results()

        for chain_id in range(1, self.nchains + 1):  # Loop through each chain
            chain_dict = result_dicts[chain_id - 1]  # Get the dictionary for this chain
            u238_chain = chain_dict[f"z15_{chain_id}"]  # Extract log_p (assuming it is z10)
            u235_chain = chain_dict[f"z17_{chain_id}"]  # Extract log_p (assuming it is z10)


            # Append the log_p values for this chain
            grouped_u238[chain_id - 1].extend(u238_chain)
            grouped_u235[chain_id - 1].extend(u235_chain)
            
        u238_wm, U238_wm_err = self.Chain_Stats(grouped_u238)
        u235_wm, U235_wm_err = self.Chain_Stats(grouped_u235)
        
        df = pd.DataFrame({"U238": u238_wm,
                           "U238_err": U238_wm_err,
                           "U235": u235_wm,
                           "U235_err": U235_wm_err,
                          },
                          index= [0])
                          
        return df
        
    def robust_weighted_stats_vector_with_ci(self, result_dicts, parameter_name):
        """
        Calculate weighted mean, standard deviation, and non-negative constrained confidence intervals for a vector-valued parameter
        from multiple MCMC chains, considering both within-chain and between-chain statistics.
        -Vector Parameters
        
        """
        chain_means = []
        chain_variances = []
    
        # Calculate mean and variance for each chain for each vector element
        for i in range(self.nchains):
            chain_data = np.array(result_dicts[i][f"{parameter_name}_{i+1}"])
            mean = np.mean(chain_data, axis=0)
            variance = np.var(chain_data, axis=0, ddof=1)
            chain_means.append(mean)
            chain_variances.append(variance)
    
        chain_means = np.array(chain_means)
        chain_variances = np.array(chain_variances)
    
        # Calculate weights as the reciprocals of the variances
        weights = np.where(chain_variances > 0, 1 / chain_variances, 0)
    
        # Calculate the weighted mean of the chain means
        weighted_means = np.average(chain_means, axis=0, weights=weights)
    
        # Calculate the between-chain variance for each element
        between_chain_variance = np.var(chain_means, axis=0, ddof=1)
    
        # Combine the within-chain and between-chain variances for each element
        total_variance = np.average(chain_variances, axis=0, weights=weights) + between_chain_variance
    
        # Calculate the weighted standard deviation for each element
        weighted_std = np.sqrt(total_variance)
    
        # Calculate the confidence intervals based on the normal distribution
        ci_68_lower = weighted_means - stats.norm.ppf(0.84) * weighted_std
        ci_68_upper = weighted_means + stats.norm.ppf(0.84) * weighted_std
        ci_95_lower = weighted_means - stats.norm.ppf(0.975) * weighted_std
        ci_95_upper = weighted_means + stats.norm.ppf(0.975) * weighted_std
        # Apply non-negative constraint to lower bounds of confidence intervals
    
        ci_68 = (ci_68_lower, ci_68_upper)
        ci_95 = (ci_95_lower, ci_95_upper)
    
    
        return weighted_means, ci_68_lower, ci_68_upper
        
    def Get_U235_Corrected_Ages(self):
        grouped_Age_corr = []
        grouped_Age_corr_low68 = []
        grouped_Age_corr_high68 = []

        # Loop through each group and collect summary stats
        for i in range(self.nbatch):
            with gzip.open(f'{self.Run_Name}_results_after_{i}_iterations.pkl', 'rb') as f:
                res_ = pickle.load(f)
        
            z_vars = [f"z{j+1}" for j in range(31)]
            result_dicts = []
        
            for chain_id, chain in enumerate(res_, start=1):
                result_dict = {}
                for z_idx, z_var in enumerate(z_vars):
                    result_dict[f"{z_var}_{chain_id}"] = chain[z_idx]
                result_dicts.append(result_dict)
        
            # Call your method to compute weighted stats for z21 (U235 age corrected)
            wm, low68, high68 = self.robust_weighted_stats_vector_with_ci(result_dicts, 'z21')
        
            grouped_Age_corr.append(wm)
            grouped_Age_corr_low68.append(low68)
            grouped_Age_corr_high68.append(high68)
            
        return grouped_Age_corr, grouped_Age_corr_low68, grouped_Age_corr_high68
    
    def Get_U238_Corrected_Ages(self):
        grouped_Age_corr = []
        grouped_Age_corr_low68 = []
        grouped_Age_corr_high68 = []

        # Loop through each group and collect summary stats
        for i in range(self.nbatch):
            with gzip.open(f'{self.Run_Name}_results_after_{i}_iterations.pkl', 'rb') as f:
                res_ = pickle.load(f)
        
            z_vars = [f"z{j+1}" for j in range(31)]
            result_dicts = []
        
            for chain_id, chain in enumerate(res_, start=1):
                result_dict = {}
                for z_idx, z_var in enumerate(z_vars):
                    result_dict[f"{z_var}_{chain_id}"] = chain[z_idx]
                result_dicts.append(result_dict)
        
            # Call your method to compute weighted stats for z21 (U235 age corrected)
            wm, low68, high68 = self.robust_weighted_stats_vector_with_ci(result_dicts, 'z20')
        
            grouped_Age_corr.append(wm)
            grouped_Age_corr_low68.append(low68)
            grouped_Age_corr_high68.append(high68)
            
        return grouped_Age_corr, grouped_Age_corr_low68, grouped_Age_corr_high68
        
    def Get_ArAr_Corrected_Ages(self):
        grouped_Age_corr = []
        grouped_Age_corr_low68 = []
        grouped_Age_corr_high68 = []

        # Loop through each group and collect summary stats
        for i in range(self.nbatch):
            with gzip.open(f'{self.Run_Name}_results_after_{i}_iterations.pkl', 'rb') as f:
                res_ = pickle.load(f)
        
            z_vars = [f"z{j+1}" for j in range(31)]
            result_dicts = []
        
            for chain_id, chain in enumerate(res_, start=1):
                result_dict = {}
                for z_idx, z_var in enumerate(z_vars):
                    result_dict[f"{z_var}_{chain_id}"] = chain[z_idx]
                result_dicts.append(result_dict)
        
            # Call your method to compute weighted stats for z21 (U235 age corrected)
            wm, low68, high68 = self.robust_weighted_stats_vector_with_ci(result_dicts, 'z26')
        
            grouped_Age_corr.append(wm)
            grouped_Age_corr_low68.append(low68)
            grouped_Age_corr_high68.append(high68)
            
        return grouped_Age_corr, grouped_Age_corr_low68, grouped_Age_corr_high68

    def Get_Delta_Ages(self):
        result_dicts = self.Get_Grouped_Chain_Results()
        
        grouped_u235, grouped_u235_low, grouped_u235_high = self.Get_U235_Corrected_Ages()
        grouped_u238, grouped_u238_low, grouped_u238_high = self.Get_U238_Corrected_Ages()
        grouped_arar, grouped_arar_low, grouped_arar_high = self.Get_ArAr_Corrected_Ages()
        
        u238_corr, u238_corr_err = self.combine_weighted_means(grouped_u238, grouped_u238_low, grouped_u238_high)
        u235_corr, u235_corr_err = self.combine_weighted_means(grouped_u235, grouped_u235_low, grouped_u235_high)
        arar_corr, arar_corr_err = self.combine_weighted_means(grouped_arar, grouped_arar_low, grouped_arar_high)
        
        delta_238, delta_238_err = self.Delta_Ages(u238_corr, u238_corr_err, arar_corr[2:], arar_corr_err[2:])
        delta_235, delta_235_err = self.Delta_Ages(u235_corr, u235_corr_err, arar_corr[3:], arar_corr_err[3:])
                                        
        df = pd.DataFrame({"Delta_Age_238": delta_238,
                   "Delta_Age_238_Err": delta_238_err,
                   "Delta_Age_235": np.insert(delta_235, 0, 0),
                   "Delta_Age_235_Err": np.insert(delta_235_err,0,0),
                   "Ar_age": arar_corr[2:],
                  "Ar_age_err": arar_corr[2:]})

        return df
        
    def Delta_Ages(self, U_ages, U_ages_err, Ar_ages, Ar_ages_err):
        """
        Monte Carlo for Delta Age Figure (Fig 1.)
        """
        N_ = len(U_ages)
        n = 10000
        delta_age = np.zeros((n, N_))
        for i in range(n):
            U_ages_mc = np.random.normal(U_ages, U_ages_err)
            Ar_ages_mc = np.random.normal(Ar_ages, Ar_ages_err)
            delta_age[i, :] = ((U_ages_mc/Ar_ages_mc) - 1) * 100
    
        return delta_age.mean(axis = 0), delta_age.std(axis = 0)
        
        
    def combine_weighted_means(self, arrays, low, high):
        # Number of elements in each array (they should all be of the same length)
        n_elements = len(arrays[0])
    
        # Initialize arrays for storing the weighted mean and combined uncertainty
        weighted_means = np.zeros(n_elements)
        combined_uncertainties = np.zeros(n_elements)
    
        for i in range(n_elements):
            means_at_i = [array[i] for array in arrays]  # Extract all values at index i from each array
            low_unc_at_i = [low_unc[i] for low_unc in low]  # Extract lower uncertainties at index i
            high_unc_at_i = [high_unc[i] for high_unc in high]  # Extract upper uncertainties at index i
    
            # Calculate the unweighted mean (simple arithmetic mean)
            weighted_means[i] = np.mean(means_at_i)
    
            # Calculate combined uncertainty as the mean of the uncertainties (average of low and high uncertainties)
            avg_uncertainties_at_i = [(low_ + high_) / 2 for low_, high_ in zip(low_unc_at_i, high_unc_at_i)]
            combined_uncertainties[i] = np.mean(avg_uncertainties_at_i)
    
        return weighted_means, combined_uncertainties
        

    def combine_unweighted_means_lows_highs(self, arrays, low, high):
        # Number of elements in each array (they should all be of the same length)
        n_elements = len(arrays[0])
    
        # Initialize arrays for storing the unweighted mean, low, and high
        unweighted_means = np.zeros(n_elements)
        unweighted_lows = np.zeros(n_elements)
        unweighted_highs = np.zeros(n_elements)
    
        for i in range(n_elements):
            means_at_i = [array[i] for array in arrays]  # Extract all values at index i from each array
            low_unc_at_i = [low_unc[i] for low_unc in low]  # Extract lower uncertainties at index i
            high_unc_at_i = [high_unc[i] for high_unc in high]  # Extract upper uncertainties at index i
    
            # Calculate the unweighted mean (simple arithmetic mean)
            unweighted_means[i] = np.mean(means_at_i)
    
            # Calculate the unweighted low (average of the lower uncertainties)
            unweighted_lows[i] = np.mean(low_unc_at_i)
    
            # Calculate the unweighted high (average of the upper uncertainties)
            unweighted_highs[i] = np.mean(high_unc_at_i)
    
        return unweighted_means, unweighted_lows, unweighted_highs
        

    def Get_Residence_Time(self):
        grouped_residence = []
        grouped_residence_low68 = []
        grouped_residence_high68 = []
        
        for i in range(self.nbatch):
            with gzip.open(f'{self.Run_Name}_results_after_{i}_iterations.pkl', 'rb') as f:
                res_ = pickle.load(f)
        
            z_vars = [f"z{i+1}" for i in range(31)]
            for chain_id in range(1, self.nchains + 1):
                for z_var in z_vars:
                    vars()[f"{z_var}_{chain_id}"] = res_[chain_id - 1][z_vars.index(z_var)]
        
            """
            Make a Results Dictionary for plotting
            """
            result_dicts = []
            
            for chain_id, res_ in enumerate(res_, start=1):
                result_dict = {}
                for var_name, value in zip(z_vars, res_):
                    result_dict[f"{var_name}_{chain_id}"] = value
                result_dicts.append(result_dict)
        
            x_wm, x_low, x_high  = self.robust_weighted_stats_vector_with_ci(result_dicts, 'z27')
            x_low = np.maximum(x_low, 0) # Just a bit of forcing to have min of 0, shouldnt be necessary but saves any mishaps
            x_high = np.minimum(x_high, 0.5)
            grouped_residence.append(x_wm)
            grouped_residence_low68.append(x_low)
            grouped_residence_high68.append(x_high)
            
            rest, rest_low, rest_high = self.combine_unweighted_means_lows_highs(grouped_residence, grouped_residence_low68, grouped_residence_high68)
            
            df = pd.DataFrame({"Residence Time": rest,
                              "Residence Time low": rest_low,
                             "Residence Time high": rest_high,
                              })
        return df
        

    def Get_Age_Perturbation_Parameters(self):
        grouped_xi_ar = []
        grouped_xi_ar_low68 = []
        grouped_xi_ar_high68 = []

        grouped_xi_u235 = []
        grouped_xi_u235_low68 = []
        grouped_xi_u235_high68 = []


        grouped_xi_u238 = []
        grouped_xi_u238_low68 = []
        grouped_xi_u238_high68 = []

        for i in range(self.nbatch):
            with gzip.open(f'{self.Run_Name}_results_after_{i}_iterations.pkl', 'rb') as f:
                res_ = pickle.load(f)

            z_vars = [f"z{i+1}" for i in range(31)]
            for chain_id in range(1, self.nchains + 1):
                for z_var in z_vars:
                    vars()[f"{z_var}_{chain_id}"] = res_[chain_id - 1][z_vars.index(z_var)]

            """
            Make a Results Dictionary for plotting
            """
            result_dicts = []
            
            for chain_id, res_ in enumerate(res_, start=1):
                result_dict = {}
                for var_name, value in zip(z_vars, res_):
                    result_dict[f"{var_name}_{chain_id}"] = value
                result_dicts.append(result_dict)

            x_wm, x_low, x_high  = self.robust_weighted_stats_vector_with_ci(result_dicts, 'z11')
            y_wm, y_low, y_high  = self.robust_weighted_stats_vector_with_ci(result_dicts, 'z29')
            z_wm, z_low, z_high  = self.robust_weighted_stats_vector_with_ci(result_dicts, 'z28')

            grouped_xi_ar.append(x_wm)
            grouped_xi_ar_low68.append(x_low)
            grouped_xi_ar_high68.append(x_high)
            grouped_xi_u235.append(y_wm)
            grouped_xi_u235_low68.append(y_low)
            grouped_xi_u235_high68.append(y_high)
            grouped_xi_u238.append(z_wm)
            grouped_xi_u238_low68.append(z_low)
            grouped_xi_u238_high68.append(z_high)
            
            Ar_xi, Ar_xi_low, Ar_xi_high = self.combine_unweighted_means_lows_highs(grouped_xi_ar,
                                                                  grouped_xi_ar_low68,
                                                                  grouped_xi_ar_high68)

            u235_xi, u235_xi_low, u235_xi_high =self.combine_unweighted_means_lows_highs(grouped_xi_u235,
                                                                  grouped_xi_u235_low68,
                                                                  grouped_xi_u235_high68)
                                    
            u238_xi, u238_xi_low, u238_xi_high =self.combine_unweighted_means_lows_highs(grouped_xi_u238,
                                                                  grouped_xi_u238_low68,
                                                                  grouped_xi_u238_high68)
                                                                  
            """
            DataFrames
            """
            df_ar = pd.DataFrame({"Ar_xi": Ar_xi,
                              "Ar_xi_low": Ar_xi_low,
                              "Ar_xi_high": Ar_xi_high})


            df_u238 = pd.DataFrame({"u238_xi": u238_xi,
                              "u238_xi_low": u238_xi_low,
                              "u238_xi_high": u238_xi_high})


            df_u235 = pd.DataFrame({"u235_xi": u235_xi,
                              "u235_xi_low": u235_xi_low,
                  "u235_xi_high": u235_xi_high})
                  
                  
        return df_ar, df_u238, df_u235
        
        
    """
    Chain Statistics
    ----------------
    Rhat ~ 1
    """
    def Gelman_Rubin(self, chain_list):
        """
        Calculates the Gelman-Rubin statistic for each parameter across chains.

        Args:
        chain_list (list of np.ndarray): A list where each element is an n x p matrix of samples from one MCMC chain

        Returns:
        Gelman-Rubin Statistic
        """
        n_ch = len(chain_list)
        n = chain_list[0].shape[0]
        # Stack chains to simplify calculations, shape is (m, n, p)
        stacked_chains = np.stack(chain_list, axis=0)
        # Calculate means across samples within each chain (m x p)
        chain_means = np.mean(stacked_chains, axis=1)
        # Overall mean across chains for each parameter (p)
        grand_mean = np.mean(chain_means, axis=0)
        # Between-chain variance for each parameter (p)
        B = (n / (n_ch - 1)) * np.sum((chain_means - grand_mean)**2, axis=0)
        # Within-chain variances for each parameter (p)
        W = np.mean(np.var(stacked_chains, axis=1, ddof=1), axis=0)
        # Estimate the variance (p)
        var_plus = ((n - 1) / n) * W + (1 / n) * B
        # Gelman-Rubin statistic (p)
        R_hat = np.sqrt(var_plus / W)
        return R_hat
        

        
        
    def Get_40K_Decay_Stuff(self):
        """
        Get some information on the four decay modes and
        the total decay constant - Some chain statistics using the Gelman-Rubin statistic approach
        """
        if self.Grouped_Chain_Results is None:
            self.Ensure_Chain_Results()

        EC_star = []
        EC_zero = []
        Beta = []
        Beta_plus = []
        Lam_tot = []

        for i in range(1, self.nchains + 1):
            chain_dict = self.Grouped_Chain_Results[i - 1]
            EC_star.append(np.array(chain_dict[f"z2_{i}"]))
            EC_zero.append(np.array(chain_dict[f"z3_{i}"]))
            Beta.append(np.array(chain_dict[f"z4_{i}"]))
            Beta_plus.append(np.array(chain_dict[f"z5_{i}"]))
            Lam_tot.append(np.array(chain_dict[f"z6_{i}"]))

        chain_list = [EC_star, EC_zero, Beta, Beta_plus, Lam_tot]

        R_hat = np.zeros(len(chain_list))
        for i in range(len(chain_list)):
            R_hat[i] = self.Gelman_Rubin(chain_list[i])

        df = pd.DataFrame({
            "EC_star": [R_hat[0]],
            "EC_zero": [R_hat[1]],
            "Beta-": [R_hat[2]],
            "Beta+": [R_hat[3]],
            "40K_lamtot": [R_hat[4]]
        })

        print("Dataframe of Gelman Rubin statistic (R-hat). An R-hat value of ~1 indicates good chain mixing and convergence.")
        return df



