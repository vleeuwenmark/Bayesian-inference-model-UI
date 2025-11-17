import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np
import scipy.io as spio
from scipy.stats import gamma
from scipy.optimize import minimize, minimize_scalar
from os.path import dirname, realpath, join as pjoin
import os
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from datetime import date
import xarray as xr

class TestData:
    def __init__(self, input_file, trial_inburn = [0, 0]):

        dir_path = dirname(realpath(__file__))
        mat_fname = pjoin(dir_path, input_file)
        self.data_mat = spio.loadmat(mat_fname)

        self.n_subjects = max(map(len,self.data_mat['aimingError']))
        self.n_steps = len(self.data_mat['aimingError'])

        self.y = self.data_mat['aimingError'][trial_inburn[0]:self.n_steps-trial_inburn[1],:self.n_subjects]
        self.p = self.data_mat['rotation'][trial_inburn[0]:self.n_steps-trial_inburn[1],:self.n_subjects]
        self.v = self.data_mat['showCursor'][trial_inburn[0]:self.n_steps-trial_inburn[1],:self.n_subjects]

        self.n_subjects = max(map(len,self.y))
        self.n_steps = len(self.y)

        self.p = pt.as_tensor_variable(self.p)
        self.v = pt.as_tensor_variable(self.v)

        self.subject_id = []

        for i in range(self.n_subjects):
            data_id = [i] * sum(~np.isnan(self.y[:,i]))
            self.subject_id.extend(data_id)



class StateSpace:
    def __init__(self,data, AB_method = 'Informed',noise_method = 'Gamma', post_predictive=False):
        
        self.y=data.y
        self.p=data.p
        self.v=data.v
        self.subject_id=data.subject_id

        if np.any(np.isnan(self.y)):
                    self.y_mask = np.isnan(self.y)
                    self.y_mean = np.nanmean(self.y)
                    self.y_std = np.nanstd(self.y)

        self.n_steps = data.n_steps
        self.n_subjects = data.n_subjects

        self.noise_method = noise_method
        self.AB_method = AB_method
        self.post_predictive = post_predictive

    def hdi_gamma(self, a, b, p):
        if a <= 1:
            # Case when shape parameter a <= 1
            lower = 0
            upper = gamma.ppf(p, a, scale=b)
        else:
            # Function to minimize
            def zerofn(x):
                return (gamma.pdf(gamma.ppf(p + gamma.cdf(x, a, scale=b), a, scale=b), a, scale=b) - gamma.pdf(x, a, scale=b)) ** 2
                    
        # Upper limit for the optimization
        maxl = gamma.ppf(1 - p, a, scale=b)

        # Optimize zerofn over the interval [0, maxl]
        res = minimize_scalar(zerofn, bounds=(0, maxl), method='bounded')

        # Get the lower and upper bounds of the HDI
        lower = res.x
        upper = gamma.ppf(p + gamma.cdf(lower, a, scale=b), a, scale=b)

        return lower, upper
        
    def gamma_params_setHDI(self, HDI_min, HDI_max, p, initial_guess = (2,1)):                    
        def objective(params, L, U, p):
            a, b = params
            lower, upper = self.hdi_gamma(a, b, p)
            return (lower - L)**2 + (upper - U)**2  # Minimize difference between HDI bounds
            
        result = minimize(objective, initial_guess, args=(HDI_min, HDI_max, p), bounds=((1e-3, None), (1e-3, None)))
        return result.x
        
    def gamma_params(self, mode, var):
        beta = (mode + pm.math.sqrt(mode*mode + 4*var))/(2 * var)
        alpha = 1 + mode * beta
        return alpha,beta

    def OneRateNonHierarchical(self, output_file = f'NoiseNonHierachical_{date.today():%d-%m-%Y}.nc', draws=1000, tune=1000):
        coords = {'trial':np.arange(self.n_steps),
                'subjects':np.arange(self.n_subjects)}
        
        with pm.Model(coords=coords) as model:
            if self.AB_method == 'Informed':
                A1mu = pm.Normal('A1mu', mu=4, sigma=0.25)
                A1std = pm.Gamma('A1std', mu=0.5, sigma=0.25)
                B1mu = pm.Normal('B1mu', mu=-2, sigma=0.25)
                B1std = pm.Gamma('B1std', mu=0.5, sigma=0.25)

                A = pm.LogitNormal('A', mu=A1mu, sigma=A1std, dims='subjects')
                B = pm.LogitNormal('B', mu=B1mu, sigma=B1std, dims='subjects')

            elif self.AB_method == 'Wide':
                A1mu = pm.Normal('A1mu', mu=2, sigma=1)
                A1std = pm.Gamma('A1std', mu=1, sigma=np.sqrt(0.5))
                B1mu = pm.Normal('B1mu', mu=-2, sigma=1)
                B1std = pm.Gamma('B1std', mu=1, sigma=np.sqrt(0.5))

                A = pm.LogitNormal('A', mu=A1mu, sigma=A1std, dims='subjects')
                B = pm.LogitNormal('B', mu=B1mu, sigma=B1std, dims='subjects')

            elif self.AB_method == 'Non Hierarchical':
                A = pm.LogitNormal('A', mu=2, sigma=0.5, dims='subjects')
                B = pm.LogitNormal('B', mu=-2, sigma=0.5, dims='subjects')


            if self.noise_method == 'Gamma':
                sigma_eta = pm.Gamma('sigma_eta', mu=1, sigma=0.5, dims='subjects')
                sigma_epsilon = pm.Gamma('sigma_epsilon', mu=4, sigma=1, dims='subjects')
                sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(pm.math.sqr(sigma_eta)+pm.math.sqr(sigma_epsilon)), dims='subjects')

            elif self.noise_method == 'Ratio':
                var_total = pm.Gamma('var_total', mu=16, sigma=1, dims='subjects')
                sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(var_total), dims='subjects')

                p1 = pm.Normal('p1', mu=-2, sigma=1, dims='subjects')
                p = pm.Deterministic('p', pm.math.invlogit(p1), dims='subjects')

                sigma_eta = pm.Deterministic('sigma_eta' ,pm.math.sqrt(p*var_total), dims='subjects')
                sigma_epsilon = pm.Deterministic('sigma_epsilon',pm.math.sqrt((1-p)*var_total), dims='subjects')

            x_init = pm.Normal('x_init', mu=0, sigma=1, dims='subjects')

            eta = pm.Normal('eta', mu=0, sigma=sigma_eta, dims=['trial', 'subjects'])
            epsilon = pm.Normal('epsilon', mu=0, sigma=sigma_epsilon, dims=['trial', 'subjects'])
            
            def grw_step(eta_t, epsilon_t, p_t, v_t, x_t, A, B):
                x_tp1 = A * (x_t+eta_t) - v_t * B * (p_t + x_t + eta_t + epsilon_t)
                return x_tp1
            
            x, updates = pytensor.scan(fn=grw_step,
                sequences=[eta, epsilon, self.p, self.v],
                outputs_info=[{"initial": x_init}],
                non_sequences=[A,B],
                name='statespace',
                strict=True)
            
            x = pm.Deterministic('x',pt.concatenate([[x_init], x[:-1,]], axis=0), dims=['trial', 'subjects'])
            
            y_hat = pm.Normal('y_hat',mu=x,sigma=sigma_total,observed=self.y,dims=['trial', 'subjects'])
            
            self.idata = pm.sample(cores=4,chains=4,draws=draws,tune=tune,init='adapt_diag',idata_kwargs={"log_likelihood": True})

            self.idata = pm.sample_posterior_predictive(self.idata_import,extend_inferencedata=True)
            idata_prior = pm.sample_prior_predictive(draws=500)

            self.idata.extend(idata_prior)

        self.idata.to_netcdf(output_file)
        self.save_to_xlsx(output_file.replace('.nc','.xlsx'))

    def OneRateNonHierarchical_reiterate_posterior(self, prior_data="", output_file="", n_iter = 1):
        if hasattr(self, 'idata') and not prior_data:
            self.idata_prior = self.idata.posterior.copy()
        else:
            self.idata_prior = az.from_netcdf(prior_data)
            self.idata_prior= self.idata_prior.posterior.copy()

        coords = {'trial':np.arange(self.n_steps),
                'subjects':np.arange(self.n_subjects)}
        
        for i in range(n_iter):
            with pm.Model(coords=coords) as model:
                A1mu = pm.Normal('A1mu', mu=self.idata_prior.A1mu.mean().values, sigma=self.idata_prior.A1mu.std().values)
                A1std = pm.Gamma('A1std', mu=self.idata_prior.A1std.mean().values, sigma=self.idata_prior.A1std.std().values)
                B1mu = pm.Normal('B1mu', mu=self.idata_prior.B1mu.mean().values, sigma=self.idata_prior.B1mu.std().values)
                B1std = pm.Gamma('B1std', mu=self.idata_prior.B1std.mean().values, sigma=self.idata_prior.B1std.std().values)

                A1 = pm.Normal('A1', mu=A1mu, sigma=A1std, dims='subjects')
                B1 = pm.Normal('B1', mu=B1mu, sigma=B1std, dims='subjects')
                A = pm.Deterministic('A', pm.math.invlogit(A1), dims='subjects')
                B = pm.Deterministic('B', pm.math.invlogit(B1), dims='subjects')

                sigma_eta = pm.Gamma('sigma_eta', mu=self.idata_prior.sigma_eta.mean().values, sigma=self.idata_prior.sigma_eta.std().values, dims='subjects')
                sigma_epsilon = pm.Gamma('sigma_epsilon',mu=self.idata_prior.sigma_epsilon.mean().values, sigma=self.idata_prior.sigma_epsilon.std().values, dims='subjects')
                sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(pm.math.sqr(sigma_eta)+pm.math.sqr(sigma_epsilon)), dims='subjects')

                eta = pm.Normal('eta', mu=0, sigma=sigma_eta, dims=['trial', 'subjects'])
                epsilon = pm.Normal('epsilon', mu=0, sigma=sigma_epsilon, dims=['trial', 'subjects'])

                x_init = pm.Normal('x_init', mu=0, sigma=1, dims='subjects')
                
                def grw_step(eta_t, epsilon_t, p_t, v_t, x_t, A, B):
                    x_tp1 = A * (x_t+eta_t) - v_t * B * (p_t + x_t + eta_t + epsilon_t)
                    return x_tp1
                
                x, updates = pytensor.scan(fn=grw_step,
                    sequences=[eta, epsilon, self.p, self.v],
                    outputs_info=[{"initial": x_init}],
                    non_sequences=[A,B],
                    name='statespace',
                    strict=True)
                
                x = pm.Deterministic('x',pt.concatenate([[x_init], x[:-1,]], axis=0), dims=['trial', 'subjects'])
                y_hat = pm.Normal('y_hat',mu=x,sigma=sigma_total,observed=self.y,dims=['trial', 'subjects'])
                self.idata = pm.sample(cores=4,chains=4,draws=1000,tune=1000,init='adapt_diag')
                idata_prior_predictive = pm.sample_prior_predictive(draws=1000,var_names=['A1mu','A1std','B1mu','B1std','A','B','sigma_eta','sigma_epsilon','y_hat'])
                self.idata.extend(idata_prior_predictive)
                self.idata = pm.sample_posterior_predictive(self.idata,extend_inferencedata=True)
                self.idata = pm.compute_log_likelihood(self.idata,extend_inferencedata=False)
                if n_iter>1: self.idata_prior = self.idata.posterior.copy()
        self.idata.to_netcdf(output_file)
        print(f'{n_iter} iteration(s) completed. Inference data saved to {output_file}')
        self.save_to_xlsx(xlsx_dir=output_file.replace('.nc','.xlsx'))

    def OneRateHierarchical(self, draw = 2000, tune = 4000, output_file = f'posteriorData-Hierarchical_{date.today():%d-%m-%Y}.nc'):
        if not output_file: output_file=f"{self.noise_method}_{date.today():%d %m %Y}.nc"
        coords = {'trial':np.arange(self.n_steps),
                'subjects':np.arange(self.n_subjects)}
        
        with pm.Model(coords=coords) as model:
            A1mu = pm.Normal('A1mu', mu=4, sigma=0.25)
            A1std = pm.Gamma('A1std', mu=0.5, sigma=0.25)
            B1mu = pm.Normal('B1mu', mu=-2, sigma=0.25)
            B1std = pm.Gamma('B1std', mu=0.5, sigma=0.25)

            A = pm.LogitNormal('A', mu=A1mu, sigma=A1std, dims='subjects')
            B = pm.LogitNormal('B', mu=B1mu, sigma=B1std, dims='subjects')

            if self.noise_method == 'Gamma':
                etamode = pm.Gamma('etamode', mu=1, sigma=0.5)
                etavar = pm.Gamma('etavar', mu=0.5, sigma=0.25)
                eta_alpha, eta_beta = self.gamma_params(etamode,etavar)

                epsilonmode = pm.Gamma('epsilonmode', mu=4, sigma=0.5)
                epsilonvar = pm.Gamma('epsilonvar', mu=0.5, sigma=0.25)
                epsilon_alpha, epsilon_beta = self.gamma_params(epsilonmode,epsilonvar)

                sigma_eta = pm.Gamma('sigma_eta', alpha=eta_alpha, beta=eta_beta, dims='subjects')
                sigma_epsilon = pm.Gamma('sigma_epsilon', alpha=epsilon_alpha, beta=epsilon_beta, dims='subjects')
                sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(pm.math.sqr(sigma_eta)+pm.math.sqr(sigma_epsilon)), dims='subjects')

            elif self.noise_method == 'NonCentered':
                gammaparams = pm.find_constrained_prior(pm.Gamma, lower=0.1, upper=5, mass = 0.99, init_guess= {'alpha':4,"beta":0.5})
                eta_mode = pm.Gamma('eta_mode', alpha=3, beta=2)
                var_norm = pm.Gamma('var_norm', **gammaparams)
                eta_var = pm.Deterministic('eta_var',eta_mode*var_norm)
                eta_alpha,eta_beta = self.gamma_params(eta_mode, eta_var)
                sigma_eta = pm.Gamma('sigma_eta', alpha=eta_alpha, beta=eta_beta, dims='subjects')

                epsilon_mode = pm.Gamma('epsilon_mode', mu=4, sigma=0.25)
                epsilon_var = pm.Gamma('epsilon_var', mu=0.25, sigma=0.05)
                epsilon_alpha, epsilon_beta = self.gamma_params(epsilon_mode,epsilon_var)

                sigma_epsilon = pm.Gamma('sigma_epsilon', alpha=epsilon_alpha, beta=epsilon_beta, dims='subjects')
                sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(pm.math.sqr(sigma_eta)+pm.math.sqr(sigma_epsilon)), dims='subjects')
            
            elif self.noise_method == 'HalfNormal':
                eta_mode = pm.HalfNormal('eta_mode', sigma=1.25) #mu=1
                sigma_eta = pm.HalfNormal('sigma_eta', sigma=(eta_mode*np.sqrt(np.pi))/np.sqrt(2), dims='subjects')

                epsilon_mode = pm.Gamma('epsilon_mode', mu=4, sigma=0.25)
                epsilon_var = pm.Gamma('epsilon_var', mu=0.25, sigma=0.05)
                epsilon_alpha, epsilon_beta = self.gamma_params(epsilon_mode,epsilon_var)

                sigma_epsilon = pm.Gamma('sigma_epsilon', alpha=epsilon_alpha, beta=epsilon_beta, dims='subjects')
                sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(pm.math.sqr(sigma_eta)+pm.math.sqr(sigma_epsilon)), dims='subjects')

            elif self.noise_method == 'InverseGamma':
                sigma_eta_mu = pm.Uniform('sigma_eta_mu', lower=0.1, upper=5)
                sigma_eta1 = pm.InverseGamma('sigma_eta1', mu=1, sigma=1)

                sigma_eta = pm.InverseGamma('sigma_eta',mu=sigma_eta_mu,sigma=sigma_eta1, dims='subjects')

                sigma_epsilon_mu = pm.Uniform('sigma_epsilon_mu', lower=0.1, upper=5)
                sigma_epsilon1 = pm.InverseGamma('sigma_epsilon1', mu=0.5, sigma=0.25)

                sigma_epsilon = pm.InverseGamma('sigma_epsilon', mu=sigma_epsilon_mu,sigma=sigma_epsilon1, dims='subjects')

                sigma_total = pm.Deterministic('sigma_total', pm.math.sqrt(pm.math.sqr(sigma_eta)+pm.math.sqr(sigma_epsilon)), dims='subjects')

            elif self.noise_method == 'Ratio':
                var_total_mu = pm.Gamma('var_total_mu', mu=10, sigma=5)
                var_total_sigma = pm.Gamma('var_total_sigma', mu=1, sigma=0.25)

                var_total = pm.Gamma('var_total', mu=var_total_mu, sigma=var_total_sigma, dims='subjects')
                sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(var_total), dims='subjects')

                p1mu = pm.Normal('p1mu',mu=-1, sigma=1)#mu=-1, sigma=0.25
                p1sigma = pm.Gamma('p1sigma',mu=1, sigma=1)#mu=0.5, sigma=0.25
                
                p1 = pm.Normal('p1', mu=p1mu, sigma=p1sigma, dims='subjects')
                p = pm.Deterministic('p', pm.math.invlogit(p1), dims='subjects') #Hier kunnen we nog een onder en bovengrens toevoegen.

                sigma_eta = pm.Deterministic('sigma_eta' ,pm.math.sqrt(p*var_total), dims='subjects')
                sigma_epsilon = pm.Deterministic('sigma_epsilon',pm.math.sqrt((1-p)*var_total), dims='subjects')

            elif self.noise_method == 'RatioTruncated':
                var_total_mu = pm.Gamma('var_total_mu', mu=4, sigma=0.25)
                var_total_sigma = pm.Gamma('var_total_sigma', mu=0.5, sigma=0.25)

                var_total = pm.Gamma('var_total', mu=var_total_mu, sigma=var_total_sigma, dims='subjects')
                sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(var_total), dims='subjects')

                p1mu = pm.Normal('p1mu',mu=-1, sigma=1)#mu=-1, sigma=0.25
                p1sigma = pm.Gamma('p1sigma',mu=1, sigma=1)#mu=0.5, sigma=0.25
                
                p1 = pm.TruncatedNormal('p1', mu=p1mu, sigma=p1sigma, lower=-8,dims='subjects')
                p = pm.Deterministic('p', pm.math.invlogit(p1), dims='subjects') #Hier kunnen we nog een onder en bovengrens toevoegen.

                sigma_eta = pm.Deterministic('sigma_eta' ,pm.math.sqrt(p*var_total), dims='subjects')
                sigma_epsilon = pm.Deterministic('sigma_epsilon',pm.math.sqrt((1-p)*var_total), dims='subjects')
            

            x_init = pm.Normal('x_init', mu=0, sigma=1, dims='subjects')

            eta = pm.Normal('eta', mu=0, sigma=sigma_eta, dims=['trial', 'subjects'])
            epsilon = pm.Normal('epsilon', mu=0, sigma=sigma_epsilon, dims=['trial', 'subjects'])
            
            def grw_step(eta_t, epsilon_t, p_t, v_t, x_t, A, B):
                x_tp1 = A * x_t - v_t * B * (p_t + x_t + epsilon_t) + eta_t
                return x_tp1
            
            x, updates = pytensor.scan(fn=grw_step,
                sequences=[eta, epsilon, self.p, self.v],
                outputs_info=[{"initial": x_init}],
                non_sequences=[A,B],
                name='statespace',
                strict=True)
            
            x = pm.Deterministic('x',pt.concatenate([[x_init], x[:-1,]], axis=0), dims=['trial', 'subjects'])
            y_hat = pm.Normal('y_hat',mu=x,sigma=sigma_total,observed=self.y,dims=['trial', 'subjects'])
            self.idata = pm.sample(cores=4,chains=4,draws=draw,tune=tune,init='adapt_diag',idata_kwargs={"log_likelihood": True})
            self.idata_prior = pm.sample_prior_predictive()
            self.idata.extend(self.idata_prior)
            self.idata = pm.sample_posterior_predictive(self.idata,extend_inferencedata=True)
            self.idata.to_netcdf(output_file)
            print(f'Inference data saved to {output_file}')
            
        self.save_to_xlsx(output_file.replace('.nc','.xlsx'))


    def OneRateHierarchical_reiterate_posterior(self, prior_data="", output_file="", n_iter = 1):
        if not output_file: output_file=f"Hierarchical{self.noise_method}_{n_iter+1}-iterations_{date.today():%d-%m-%Y}.nc"

        if hasattr(self, 'idata') and not prior_data:
            self.idata_prior = self.idata.posterior.copy()
        else:
            self.idata_prior = az.from_netcdf(prior_data)
            self.idata_prior= self.idata_prior.posterior.copy()

        coords = {'trial':np.arange(self.n_steps),
                'subjects':np.arange(self.n_subjects)}
        
        for i in range(n_iter):
            with pm.Model(coords=coords) as model:
                A1mu = pm.Normal('A1mu', mu=self.idata_prior.A1mu.mean().values, sigma=self.idata_prior.A1mu.std().values)
                A1std = pm.Gamma('A1std', mu=self.idata_prior.A1std.mean().values, sigma=self.idata_prior.A1std.std().values)
                B1mu = pm.Normal('B1mu', mu=self.idata_prior.B1mu.mean().values, sigma=self.idata_prior.B1mu.std().values)
                B1std = pm.Gamma('B1std', mu=self.idata_prior.B1std.mean().values, sigma=self.idata_prior.B1std.std().values)

                A1 = pm.Normal('A1', mu=A1mu, sigma=A1std, dims='subjects')
                B1 = pm.Normal('B1', mu=B1mu, sigma=B1std, dims='subjects')
                A = pm.Deterministic('A', pm.math.invlogit(A1), dims='subjects')
                B = pm.Deterministic('B', pm.math.invlogit(B1), dims='subjects')

                if self.noise_method == 'Ratio':
                    var_total_mu = pm.Gamma('var_total_mu', mu=self.idata_prior.var_total_mu.mean().values, sigma=self.idata_prior.var_total_mu.std().values)
                    var_total_sigma = pm.Gamma('var_total_sigma', mu=self.idata_prior.var_total_sigma.mean().values, sigma=self.idata_prior.var_total_sigma.std().values)

                    var_total = pm.Gamma('var_total', mu=var_total_mu, sigma=var_total_sigma, dims='subjects')
                    sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(var_total), dims='subjects')

                    p1mu = pm.Normal('p1mu',mu=self.idata_prior.p1mu.mean().values, sigma=self.idata_prior.p1mu.std().values)
                    p1sigma = pm.Gamma('p1sigma',mu=self.idata_prior.p1sigma.mean().values, sigma=self.idata_prior.p1sigma.std().values)

                    p1 = pm.Normal('p1', mu=p1mu, sigma=p1sigma, dims='subjects')
                    p = pm.Deterministic('p', pm.math.invlogit(p1), dims='subjects')

                    sigma_eta = pm.Deterministic('sigma_eta' ,pm.math.sqrt(p*var_total), dims='subjects')
                    sigma_epsilon = pm.Deterministic('sigma_epsilon',pm.math.sqrt((1-p)*var_total), dims='subjects')

                if (self.noise_method == 'Gamma') | (self.noise_method == 'NonCentered'): 
                    eta_alpha = pm.Gamma('eta_alpha', mu=self.idata_prior.eta_alpha.mean().values, sigma=self.idata_prior.eta_alpha.std().values)
                    eta_beta = pm.Gamma('eta_beta', mu=self.idata_prior.eta_beta.mean().values, sigma=self.idata_prior.eta_beta.std().values)

                    epsilon_alpha = pm.Gamma('epsilon_alpha', mu=self.idata_prior.epsilon_alpha.mean().values, sigma=self.idata_prior.epsilon_alpha.std().values)
                    epsilon_beta = pm.Gamma('epsilon_beta', mu=self.idata_prior.epsilon_beta.mean().values, sigma=self.idata_prior.epsilon_beta.std().values)

                    sigma_eta = pm.Gamma('sigma_eta', alpha=eta_alpha, beta=eta_beta, dims='subjects')
                    sigma_epsilon = pm.Gamma('sigma_epsilon', alpha=epsilon_alpha, beta=epsilon_beta, dims='subjects')
                    sigma_total = pm.Deterministic('sigma_total',pm.math.sqrt(pm.math.sqr(sigma_eta)+pm.math.sqr(sigma_epsilon)), dims='subjects')

                x_init = pm.Normal('x_init', mu=0, sigma=1, dims='subjects')

                eta = pm.Normal('eta', mu=0, sigma=sigma_eta, dims=['trial', 'subjects'])
                epsilon = pm.Normal('epsilon', mu=0, sigma=sigma_epsilon, dims=['trial', 'subjects'])
                
                def grw_step(eta_t, epsilon_t, p_t, v_t, x_t, A, B):
                    x_tp1 = A * (x_t+eta_t) - v_t * B * (p_t + x_t + eta_t + epsilon_t)
                    return x_tp1
                
                x, updates = pytensor.scan(fn=grw_step,
                    sequences=[eta, epsilon, self.p, self.v],
                    outputs_info=[{"initial": x_init}],
                    non_sequences=[A,B],
                    name='statespace',
                    strict=True)
                
                x = pm.Deterministic('x',pt.concatenate([[x_init], x[:-1,]], axis=0), dims=['trial', 'subjects'])
                y_hat = pm.Normal('y_hat',mu=x,sigma=sigma_total,observed=self.y,dims=['trial', 'subjects'])
                self.idata = pm.sample(cores=4,chains=4,draws=2000,tune=4000,init='adapt_diag')
                # self.idata = pm.sample_posterior_predictive(self.idata,extend_inferencedata=True)
                if i>0: self.idata_prior = self.idata.posterior.copy()
        self.idata.to_netcdf(output_file)
        print(f'{n_iter} iteration(s) completed. Inference data saved to {output_file}')
        self.save_to_xlsx(xlsx_dir=output_file.replace('.nc','.xlsx'))
            
    def save_to_xlsx(self,xlsx_dir = None, idata_import=""):
        if idata_import:
            self.idata = az.from_netcdf(idata_import)
        if not xlsx_dir:
            xlsx_dir=f"posteriorData-NoiseHierarchical_{date.today():%d-%m-%Y}.xlsx"
            
        vars = ['A','B','sigma_eta','sigma_epsilon','sigma_total']

        if os.path.isfile(xlsx_dir):
            try:
               os.rename(xlsx_dir,xlsx_dir)
            except PermissionError:
                print(f"PermissionError: Parameters not saved to .xlsx output file. Check if {xlsx_dir} is opened")
            else:
                writer = pd.ExcelWriter(xlsx_dir, engine='openpyxl')
                for var in vars:
                    params = az.summary(self.idata,var_names=var)
                    params.to_excel(writer, sheet_name=var)
                writer.close()
                print(f'Writing to {xlsx_dir} completed.')
        else:
            writer = pd.ExcelWriter(xlsx_dir, engine='openpyxl')
            for var in vars:
                params = az.summary(self.idata,var_names=var)
                params.to_excel(writer, sheet_name=var)
            writer.close()
            print(f'Writing to {xlsx_dir} completed.')
    
    def create_image(self, noise_method='', modeltype='', img_select="", output_path="", output_fname=None, idata_import = None):
        if idata_import:
            self.idata = az.from_netcdf(idata_import)


        if img_select["AB"].get() == "Trace":
            ax_AB = az.plot_trace(self.idata, var_names=['A','B'],combined=True,compact=True)
            if output_fname: 
                ax_AB.savefig(f"{output_path.get()}/{output_fname.get()}_AB_trace")
            else:
                ax_AB.savefig(f"{output_path}/AB_trace")
        elif img_select["AB"].get() == "Posterior":
            ax_AB = az.plot_posterior(self.idata, var_names=['A','B'],combine_dims={'subjects'})
            if output_fname: 
                ax_AB.savefig(f"{output_path.get()}/{output_fname.get()}_AB_posterior")
            else:
                ax_AB.savefig(f"{output_path.get()}/AB_posterior")

        if img_select["Noise"].get() == "Trace":
            ax_noise = az.plot_trace(self.idata, var_names=['sigma_eta','sigma_epsilon','sigma_total'])
            if output_fname: 
                ax_noise.savefig(f"{output_path.get()}/{output_fname.get()}_Noise_trace")
            else:
                ax_noise.savefig(f"{output_path}/Noise_trace")
        elif img_select["Noise"].get() == "Posterior":
            ax_noise = az.plot_posterior(self.idata, var_names=['sigma_eta','sigma_epsilon','sigma_total'],combine_dims={'subjects'})
            if output_fname: 
                ax_noise.savefig(f"{output_path.get()}/{output_fname.get()}_Noise_posterior")
            else:
                ax_noise.savefig(f"{output_path}/Noise_posterior")

        if modeltype["hierarchical"].get() == "Hierarchical":
            if img_select["Noise_prior_gamma"].get() == "Trace":
                ax_prior = az.plot_trace(self.idata, var_names=['eta_mode','eta_var','epsilon_mode','epsilon_var'])
                if output_fname: 
                    ax_prior.savefig(f"{output_path.get()}/{output_fname.get()}_Hyperparams_trace")
                else:
                    ax_prior.savefig(f"{output_path}/Hyperparams_trace")
            elif img_select["Noise_prior_gamma"].get() == "Posterior":
                ax_prior = az.plot_trace(self.idata, var_names=['eta_mode','eta_var','epsilon_mode','epsilon_var'])
                if output_fname: 
                    ax_prior.savefig(f"{output_path.get()}/{output_fname.get()}_Hyperparams_posterior")
                else:
                    ax_prior.savefig(f"{output_path}/Hyperparams_posterior")
            elif img_select["Noise_prior_ratio"].get() == "Trace":
                ax_prior = az.plot_trace(self.idata, var_names=['p','p1','p1mu','p1sigma'])
                if output_fname: 
                    ax_prior.savefig(f"{output_path.get()}/{output_fname.get()}_Hyperparams_trace")
                else:
                    ax_prior.savefig(f"{output_path}/Hyperparams_trace")
            elif img_select["Noise_prior_ratio"].get() == "Posterior":
                ax_prior = az.plot_posterior(self.idata, var_names=['p','p1','p1mu','p1sigma'])
                if output_fname: 
                    ax_prior.savefig(f"{output_path.get()}/{output_fname.get()}_Hyperparams_posterior")
                else:
                    ax_prior.savefig(f"{output_path}/Hyperparams_posterior")
            elif img_select["Noise_prior_NonCentered"].get() == "Trace":
                ax_prior = az.plot_trace(self.idata, var_names=['var_total_mu','var_total_sigma','var_total'])
                if output_fname: 
                    ax_prior.savefig(f"{output_path.get()}/{output_fname.get()}_Hyperparams_trace")
                else:
                    ax_prior.savefig(f"{output_path}/Hyperparams_trace")
            elif img_select["Noise_prior_NonCentered"].get() == "Posterior":
                ax_prior = az.plot_posterior(self.idata, var_names=['var_total_mu','var_total_sigma','var_total'])
                if output_fname: 
                    ax_prior.savefig(f"{output_path.get()}/{output_fname.get()}_Hyperparams_posterior")
                else:
                    ax_prior.savefig(f"{output_path}/Hyperparams_posterior")

   
        plt.show()
        
