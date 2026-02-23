import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from cosmology4 import Cosmology




class Likelihood:
    def __init__(self, filename = "pantheon_data.txt"):
        """
        File is loaded in and generated from text, accounting for 
        first line not being data.

        Parameters z, mu_obs and mu_err are columns stored as numpy arrays.
        """
    
        data = np.genfromtxt(filename, names = True, dtype = float)

        # Data columns attribute each column to varible needed for later calculations.
        self.z = data["z"]
        self.mu_obs = data["mu_obs"]
        self.sigma = data["mu_err"]

    def model_m(self, z, theta, M = -19.3, n = 5000):
        """
        Computes model apparent magnitudes m(z) for given cosmological parameters.

        'M' is absolute magnitude of supernovae, set as keyword for all calls.
        'n' is set to have convergence <<<1 but with sensible runtime.
        """

        # Cosmological parameters assigned to theta vector
        Omega_m, Omega_lambda, H0 = theta

        # Create instance of Cosmology from cosmology4.py and compute distance moduli
        cosmo = Cosmology(H0, Omega_m, Omega_lambda, np.max(z))
        mu = cosmo.distance_moduli_mu(n=n, z_values=z)

        # Adjust data for supernovae magnitude to align with observational data.
        mag = mu + M
        
        return mag

    def __call__(self, theta, M = -19.3, n = 5000, model = "lambda_cdm"):
        """
        __call__ turns instances into callable objects like regular functions

        If, elif, else loop used to define parameter vectors depending on the chosen
        model.
        """
        if model == "lambda_cdm":
            Omega_m, Omega_lambda, H0 = theta

        elif model == "no_lambda":
            Omega_m, H0 = theta
            Omega_lambda = 0.0

        else:
            raise ValueError("Unknown model type")
        
        # Turn cosmological parameters into an array.
        theta_3vector = np.array([Omega_m, Omega_lambda, H0])


        residuals = self.mu_obs - self.model_m(self.z, theta_3vector, M=M, n=n)
        chi2 = np.sum((residuals / self.sigma) ** 2)

        logL = -0.5 * chi2
        return logL
    
    def convergence_plot(self, theta, M=-19.3):
        """
        M is set as the absolute magnitude of a supernova. 

        Method evaluates log-likelihood for each N value, 
        comparing to a reference value at large N.
        """

        # Range of N values placed in array for plotting.
        N_values = np.array([50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 6000, 7000, 9000, 10000])

        # Reference value set at best estimate.
        N_ref = 8000

        # self() uses __call__ to obtain the log likelihood for reference N value.
        likeref = self(theta, M=M, n=N_ref)

        #Log likelihood for array of tested N values including absolute value.
        logL_vals = np.array([self(theta, M=M, n=N) for N in N_values])
        delta_logL = np.abs(logL_vals - likeref)
        
        # Print convergence values to allow for analysis for chosen future N value.
        for i in range(len(N_values)):
            N = N_values[i]
            d = delta_logL[i]
            print(f"N={N:5d}  |ΔlogL|={d:.6f}")

        # Plot convergence

        plt.plot(N_values, delta_logL, marker="o")

        # Mark line to show difference from 1.
        plt.axhline(1.0, linestyle="--")   
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of integration points (N)")
        plt.ylabel(r"$\Delta \log L$ relative to $N_{ref}$")
        plt.title("Convergence of log-likelihood with N")
        plt.show()

        return delta_logL
    
    def negative_loglike(theta, likelihood_obj, model):
        """
        Takes given parameters and computes negative log-likelihood to optimise
        rather than minimise. 
        """
        return -likelihood_obj(theta, model = model)
    
    
    def optimize(self, theta_estimate, model = "lambda_cdm", bounds = [(0, 1), (0, 1), (50, 90)]):
        """
        Optimisation method. Uses previous negative log likelihood along with scipy to find the best 
        fitting parameters. 

        Bounds keyword allows restricted, positive values for parameters.

        Includes best guess initial values for paramters.
        """
        result = minimize(Likelihood.negative_loglike, theta_estimate, args=(self, model), bounds = bounds)

        print("Optimization result object:")
        print(result)

        print("Best-fit parameters:")
        print(result.x)

        print("Maximum log-likelihood with Best-Fit parameters:")

        # flip sign back to get log-likelihood
        print(-result.fun) 
        return result
    
    def plot_fit(self, theta, M=-19.3, n=5000, model="lambda_cdm"):
        """
        Plots best fit model with chosen n value, dependent on the model chosen which goes through if/else statement.

        m_model then calculated. And then plotted against the observed data with error bars.
        """
        # If statement allows case when omega_lambda = 0
        if model == "no_lambda":
            theta_3vector = np.array([theta[0], 0.0, theta[1]])
        else:
            theta_3vector = theta

        m_model = self.model_m(self.z, theta_3vector, M=M, n=n)

        # Plot
        plt.figure(figsize=(10,6))

        plt.errorbar(self.z, self.mu_obs, yerr=self.sigma, fmt='o', label='Observed data', capsize=5, markersize=4, alpha=0.6)

        #sort z values numerically from 0.
        sorted_indices = np.argsort(self.z)

        # Plot aligning values of z and m_model to ensure correct plotting of the best fit line.
        plt.plot(self.z[sorted_indices], m_model[sorted_indices], 'r-', label='Best-fit model', linewidth=2)
        plt.xlabel("Redshift (z) [-]")
        plt.ylabel("Apparent magnitude m(z) [-]")
        plt.title("Best-fit Model vs Observed Data")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, theta, M=-19.3, n=5000, model="lambda_cdm"):
        """
        Plots the normalized residuals (data - model) / errors.

        Residuals are plotted with error bars of 1 sigma, and horizontal lines at 1 and 2 sigma are positioned to visually assess the fit quality.
        """

        # Three vector defined for residual calculation based on model chosen.
        if model == "no_lambda":
            theta_3vector = np.array([theta[0], 0.0, theta[1]])
        else:
            theta_3vector = theta
    

        m_model = self.model_m(self.z, theta_3vector, M=M, n=n)
        
        # Calculate residuals
        residuals = (self.mu_obs - m_model) / self.sigma

        # Calculate mean and standard deviation.
        mean_res = np.mean(residuals)
        std_res = np.std(residuals, ddof=1)  # sample std is standard in stats

        # Creates array of 1 sigma errors which aligns with residual values for plotting error bars.
        normalized_errors = np.ones_like(residuals)
        
        # Plot residuals with error bars
        plt.figure(figsize=(10, 6))
        plt.errorbar(self.z, residuals, yerr=normalized_errors, fmt='o', capsize=5, markersize=4, alpha=0.6, label='Residuals')
        
        # Add horizontal lines at 1 sigma and 2 sigma for 95% assurance.
        plt.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.3)
        plt.axhline(1, color='r', linestyle='--', linewidth=1, alpha=0.5, label='±1σ')
        plt.axhline(-1, color='r', linestyle='--', linewidth=1, alpha=0.5)
        plt.axhline(2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='±2σ')
        plt.axhline(-2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        
        # Label axes and title
        plt.xlabel('Redshift (z) [-]')
        plt.ylabel('Normalized residuals (data - model) / σ')
        plt.title('Residuals of Best-fit Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        plt.savefig("residuals.png")
        plt.show()

        return mean_res, std_res



L = Likelihood("pantheon_data.txt")

# Initial test for log likelihood with suggested parameters.
theta_test = np.array([0.3, 0.7, 70.0])

# Evaluate log-likelihood
loglike = L(theta_test)
L.convergence_plot(theta_test)
print("Log-likelihood at Ωm=0.3, ΩΛ=0.7, H0=70:")
print(loglike)


# Theta estimated parameters fed into optimise with appropriate.
theta_estimate_lambda_cdm = np.array([0.3, 0.7, 72])
bounds_lambda_cdm = [(0, 1), (0, 1), (50, 90)]
result_lambda_cdm = L.optimize(theta_estimate_lambda_cdm, model = "lambda_cdm")


# Length two vector for the model excluding omega_lambda.
theta_estimate_nolam = np.array([0.3, 72])
bounds_nolam = [(0, 1), (50, 90)]
result_nolam = L.optimize(theta_estimate_nolam, model="no_lambda", bounds=bounds_nolam)



# Plot the data with best-fit model.
L.plot_fit(result_lambda_cdm.x, model ="lambda_cdm")
print("Residual statistics:")

# Take means and stddev out of residuals for both models to compare fit quality.
mean_res_lambda_cdm, std_res_lambda_cdm = L.plot_residuals(result_lambda_cdm.x, model = "lambda_cdm")
print(f"Variable Omega Lambda residuals → mean: {mean_res_lambda_cdm:.4f}, std: {std_res_lambda_cdm:.4f}")

L.plot_fit(result_nolam.x, model="no_lambda")
mean_res_nolam, std_res_nolam = L.plot_residuals(result_nolam.x, model = "no_lambda")
print(f"Omega Lambda = 0 residuals → mean: {mean_res_nolam:.4f}, std: {std_res_nolam:.4f}")