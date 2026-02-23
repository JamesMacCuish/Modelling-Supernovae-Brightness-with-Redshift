import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Speed of light in km/s, constants okay to be used as Global Variable.
c = 299792.458  

class Cosmology:


    def __init__(self, H0, Omega_m, Omega_lambda, z):
        """
        __init__ is a method known as a constructor that
        runs automatically when a class object is created such
        as c = Cosmology(70, 0.3, 0.7), it assigns values 
        to your parameters.
    
        Self assigns values to attributes that are being passed 
        in through objects, and can be used in subsequent methods.
        """

        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.z = z
        self.Omega_k = 1 - (Omega_m + Omega_lambda)
        self.h = (H0) / 100
        self.Omega_mh2 = round(Omega_m * (self.h**2), 6)


    def distance_integrand(self, z):
        """
        All Omega_m/lambda/k & Redshift imputed to return distance integrand.

        This doesn't include the factor of c/H0 in the full integral.
        """

        #self.(parameter) takes the value passed through __init__from when cosmology class called.
        D_integrand = 1 / ((self.Omega_m * (1 + z)**3 + self.Omega_k * (1 + z)**2 + self.Omega_lambda)**0.5)
        return D_integrand
    
    def __str__(self):
        """
        The __str__ method is a special method which is used to define a string as an object.
        """

        # 'f' strings allow objects to be easy formatted into strings for printing.
        print(f"With H0 = {self.H0} km/s/Mpc, Omega_m = {self.Omega_m}, Omega_lambda = {self.Omega_lambda}, Omega_k = {self.Omega_k}")
        return f"With H0 = {self.H0}, Omega_m = {self.Omega_m}, Omega_lambda = {self.Omega_lambda}, Omega_k = {self.Omega_k}"

    def universe_shape(self):
        """
        A checks if the universe is flat, open or closed based on omega_k value.
        """

        # == sets an object to a value, = is an assignment operator only.
        if self.Omega_k == 0.0:
            print("The Universe is flat as Omega_k = 0.")
            return True
        else:
            print("The Universe is not flat as Omega_k does not equal 0.")
            return False
        
    def set_Omega_m(self, updated_Omega_lambda):
        """
        This method allows new values for omega_m & omega_lambda to update omega_k,
        to a new value.
        """

        # Set Omega_m to __init__ value to allows lambda and k to vary shape of universe.
        self.Omega_k = 1 - (self.Omega_m + updated_Omega_lambda)
        updated_Omega_k = self.Omega_k
        print(f"Omega_lambda has been updated to {updated_Omega_lambda} and Omega_k is now {self.Omega_k}.")
        return updated_Omega_k
    
    def plot_distance_integrand(self, updated_Omega_lambda):
        """
        This method plots the distance integrand against redshift for a given z &
        updated value of omega_lambda.
        """
        
        # linspace creates an array of values from 0 to 1 with 100 points.
        z = np.linspace(0, 1, 100)
        D_integrand = 1 / ((self.Omega_m * (1 + z)**3 + self.Omega_k * (1 + z)**2 + updated_Omega_lambda)**0.5)
        plt.plot(z, D_integrand)
        plt.ylabel('Distance Integrand [-]')
        plt.xlabel('Redshift (z) [-]')
        plt.title('Distance Integrand vs Redshift')
        plt.show()

    def plot_varying_Omega_m(self):
        """
        Array of different Omega_m values imputed, before looping through Distance Integrand,

        Varying with redshift again allows plot to show how Omega_m affects distance at same redshift.
        """
        z = np.linspace(0, 1, 100)
        Omega_m_values = [0.1, 0.2, 0.3, 0.4, 0.5] # Different values of Omega_m to plot
        for Omega_m in Omega_m_values:
            Omega_k = 1 - (Omega_m + self.Omega_lambda)
            D_integrand = 1 / ((Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + self.Omega_lambda)**0.5)
            plt.plot(z, D_integrand)
        plt.xlabel('Redshift (z) [-]')
        plt.ylabel('Distance Integrand [-]')
        plt.title('Distance Integrand vs Redshift for varying Omega_m')
        plt.legend(['Omega_m = 0.1', 'Omega_m = 0.2', 'Omega_m = 0.3', 'Omega_m = 0.4', 'Omega_m = 0.5'])
        plt.show()

        # What we find is that as omega_m decreases the distance 'D' increases.
        # The distance an object is away from us goes as 1/omega_m.
        
    def plot_setting_Omega_k(self):
        """
        Method plots the distance integrand against redshift for a set value of inputed omega_k,

        This makes omega _k and omega_lambda vary in turn.
        """

        z = np.linspace(0, 1, 100) # Different values of Omega_lambda to plot
        Omega_m_values = [0.1, 0.2, 0.3, 0.4, 0.5] # Different values of Omega_m to plot
        Omega_k = self.Omega_k
        for Omega_m in Omega_m_values:
            Omega_lambda = 1 - (Omega_m + Omega_k)
            D_integrand = 1 / ((Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)**0.5)
            plt.plot(z, D_integrand)
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift for set Omega_lambda')
        plt.legend(['Omega_m = 0.1', 'Omega_m = 0.2', 'Omega_m = 0.3', 'Omega_m = 0.4', 'Omega_m = 0.5'])
        plt.show()

        # When omega_k is set, we see as omega_m increases the distance 'D' decreases.


    
    # Full Galactic Distance Integrals
    def rectangle_rule(self, n):
        """
        Speed of light imputed in km/s to cancel the inverse units provided by Hubble's constant H0.

        The distance output is in Megaparsecs [Mpc]

        Same integrand used as in D_integrand, except now z is incremental from 0 toward self.z.
        """

        Delta_x = self.z / n
        summation1 = 0

        # Sum up to n - 1 as required
        for i in range(n): 
            # Call distance integrand from previous method in order to not repeat.
            summation1 += self.distance_integrand(i * Delta_x)

        # No printing inside method or it will print out all specific values for all 'n'.
        rectangle_integral = (c / self.H0) * (Delta_x) * summation1
        return rectangle_integral
    

    def trapezoid_rule(self, n):
        """
        Number of steps 'n' inputed for new method output. 

        The summation now computes the area under our curve with more accurate trapezoids which gives,
        a better distance approximation.
        This rule is exact for linear functions as it is essentially one large trapezoid.

        Ranges change from the rectangle rule.
        """

        Delta_x = self.z/(n-1)
        summation2 = 0

        # Sum up to n - 2
        for i in range(1, n - 1): 
            summation2 += self.distance_integrand(i * Delta_x)

        # Individual computations at z = 0 and z = (n - 1) * Delta_x used as well as summation.
        trapezoid_integral = (c / self.H0) * (Delta_x / 2) * ((2 * summation2 + self.distance_integrand(0) + self.distance_integrand((n - 1) * Delta_x)))
        return trapezoid_integral
    

    def simpson_rule(self, n):
        """
        Simpson's rule uses quadratic segments to approximate the area under the curve, 
        Delta_x changes and the function now includes two summations.
        """

        Delta_x = self.z/ (2 * n)
        summation3part1 = 0

        # Two summations, so two variables i, t to loop through range to avoid confusion.
        for i in range(n):
            summation3part1 += self.distance_integrand((2 * i + 1) * Delta_x)

        summation3part2 = 0
        for t in range(1, n):
            summation3part2 += self.distance_integrand(2 * t * Delta_x)

        simpson_integral = (c / self.H0) * (Delta_x / 3) * (4 * summation3part1 + 2 * summation3part2 + self.distance_integrand(0) + self.distance_integrand(2 * n * Delta_x))
        return simpson_integral 

    def compute_all_values(self, n):
        """
        This method computes all three integral methods at once for a given inputed n value.
        """

        rect = self.rectangle_rule(n)
        trap = self.trapezoid_rule(n)
        simp = self.simpson_rule(n)

        return rect, trap, simp

        # Absolute error as a function of number of trials, ref value will be simpsons method at n = 10^4
        # Different approaches converge at different rates to different accuracies.
        # Simpson > Trapezoid > Rectangle, as expected geometrically.

    def error_plot(self, n): 
        """
        This method plots the absolute fractional error for each of the three methods.

        It involves creating an array of desired 'n' values and using, with a reference (exact) value,
        to loop through and compute the fractional error for each method.
        """

        n_values = np.arange(10, n + 1, 1000)
        ref_value = self.simpson_rule(10000)

        # Empty array ready to have y - axis values appended into.
        rectangle_errors = []
        trapezoid_errors = []
        simpson_errors = []

        for i in range(10, n + 1, 1000):
            
            rectangle_errors.append(abs(self.rectangle_rule(i) - ref_value) / self.rectangle_rule(i))
            trapezoid_errors.append(abs(self.trapezoid_rule(i) - ref_value) / self.trapezoid_rule(i))
            simpson_errors.append(abs(self.simpson_rule(i) - ref_value) / self.simpson_rule(i))
        
        
        # Target accuracy given as the ratio of the avg. galaxy diameter (50,000 pc) divided by 3.2 x 10^9 pc
        # This allows for an uncertainty/ resolution that accounts for the size, shape and orientation of the galaxy.
        # A galaxy isn't a point object, so being within this accuracy is acceptable for measurements.
        # Ratio in parsecs.
        accurracy = 50000 / 3200000000  

        
        plt.loglog(n_values, rectangle_errors, label='Rectangle Rule')
        plt.loglog(n_values, trapezoid_errors, label='Trapezoid Rule')
        plt.loglog(2*n_values, simpson_errors, label='Simpson\'s Rule')
        plt.axhline(y = accurracy, color = 'b', linestyle = '--', label = 'Desired Accuracy (approx. 1e-5)')
        plt.xlim(10, 2*n)
        plt.xlabel('Number of Intervals (n) [-]')
        plt.ylabel('Absolute Fractional Error [-]')
        plt.title('Absolute Fractional Error vs Number of Intervals')
        plt.legend(loc=1, prop={'size': 9})
        plt.show()

    def cumulative_trap(self, zmax, n):
        """
        This method uses the cumulative form of the Trapezoid rule to compute an array
        of distances to a range of redshifts. It inputs the maximum redshift self.z, 
        and the number of steps n, and returns the z sample points and distances at those points.
        """

        #create an array of 'n' numbers from 0 - z to allow z to 'vary'.
        z = np.linspace(0, zmax, n)
        y = np.array([self.distance_integrand(z_i) for z_i in z])

        Delta_x = zmax / (n - 1)
        D_array = np.zeros(n)
        D_array[0] = 0

        for j in range(1, n):
        
            D_array[j] = D_array[j - 1] + (Delta_x / 2) * (y[j] + y[j - 1])

        D_array = D_array * (c / self.H0)
        
        #plot z against D_array
        plt.xlabel('Redshift (z) [-]')
        plt.ylabel('Galactic Distance (D) [Mpc]')
        plt.title('Galactic Distance vs Redshift \n Using the Cumulative Trapezoid Method')
        plt.plot(z, D_array, color = 'r')
        plt.show()

    def interpolate(self, n, z_values, plot=True):
        """
        Method interpolates the galactic distances for a given set of redshift values.

        z can be disordered, the interpolation will still estimate the value (distance) of the function at,
        that sample point and assign it to a distance array for plotting.
        """
        # Create an array of z values, which might be in any order, zmax and associated z_grid follows.
        z = np.array(z_values)
        zmax = np.max(z)
        z_array = np.linspace(0, zmax, n)

        # Evaluate integrand for each index of the z_grid array.
        D = self.distance_integrand(z_array)

        # 'zeros_like' replicates shape of desired array, essential for plot.
        D_values = np.zeros_like(z_array)
        for j in range(1, n):
            # b - a in Delta_x, divided by 1 as only one space between them for different values of z, not using zmax on top like none interpolated integrals.
            # b - a, here is just the given z_grid index divided by the value of the previous index.
            Delta_x = z_array[j] - z_array[j - 1]
            D_values[j] = D_values[j - 1] + 0.5 * Delta_x * (D[j] + D[j - 1])

        # Apply the conversion factor c/H0 to get distance values distance in Mpc.
        D_values = (c / self.H0) * D_values

        # Create interpolator and evaluate at requested (possibly unordered) z values
        # ='cubic' joins the scattered points in a best fit manner, to show how it resembles fully integrated version through estimation.
        interpolation = interp1d(z_array, D_values, kind = 'cubic')
        D_interp = interpolation(z)

        # if statement allows graph to be plotted only once rather than every time method is called.
        if plot:
            plt.plot(z_array, D_values, color ='r', label = 'grid D(z)')
            plt.scatter(z, D_interp, color ='g', label = 'interpolated')
            plt.xlabel('Redshift (z) [-]')
            plt.ylabel('Galactic Distance (D) [Mpc]')
            plt.title('Galactic Distance vs Redshift with Interpolation')
            plt.legend()
            plt.show()

        return z, D_interp
    

    def distance_moduli_mu(self, n, z_values):
        """
        Redshift values inputed and ran through interpolate method to produce interpolated distances ready,
        to be fed into distance moduli formula, dependent on luminosity distance D_l.
        """
        
        z_values, D_interp = self.interpolate(n, z_values, plot=False)

        z_ordered = np.sort(z_values)
        # if, elif, else statements needed as luminosity distance formula changes based on universe shape.
        if self.Omega_k == 0:
            D_l = (1 + z_ordered) * D_interp
        elif self.Omega_k < 0:
            D_l = (1 + z_ordered) * (c / self.H0) * (1 / np.sqrt(abs(self.Omega_k))) * np.sin(np.sqrt(abs(self.Omega_k)) * (D_interp * (self.H0 / c)))
        else:
            D_l = (1 + z_ordered) * (c / self.H0) * (1 / np.sqrt(abs(self.Omega_k))) * np.sinh(np.sqrt(abs(self.Omega_k)) * (D_interp * (self.H0 / c)))
        
        #vectorised approach, do not need to loop over indices.
        mu = 5 * np.log10(D_l) + 25
    
        #print(mu)
        #plot mu against z_values
        #plt.plot(z_ordered, mu, color = 'b')
        #plt.title(f'Distance Moduli vs Redshift \n H0 = {self.H0} km/s/Mpc, $\\Omega_m$ = {self.Omega_m}, $\\Omega_\\lambda$ = {self.Omega_lambda}, $\\Omega_K$ = {round((1 - (self.Omega_lambda + self.Omega_m)), 2)}')
        #plt.xlabel('Redshift (z) [-]')
        #plt.ylabel('Distance Moduli (mu) [mag]')
        #plt.show()

        return mu
    
    def plot_distance_moduli(self):
        """
        Nested mu_for_values function creates a second instance of Cosmology which uses the interpolate method
        to produce values for D_l for variable conditions (multiple H0's for constant Omega_m/lambda for example).

        distance_moduli_mu utilised in nested function, after accounting for +ve/-ve/0 Omega_k, to return a mu value which
        is called for values for the plots below.
        """


        # Linspace allows for a smooth curve but equally spreading large numbers of z values up to zmax, not just the inputed values.
        zmax = self.z
        z_grid = np.linspace(0, zmax, 400)
        n = 1000

        # Nested function saves space by taking values for each plot, ordering, sorting Omega_k, D_l and then array of mu values.
        def mu_for_values (H0_vals, Omega_m, Omega_lambda):
            mu_instance = Cosmology(H0_vals, Omega_m, Omega_lambda, zmax)
            z_ret, D_interp = mu_instance.interpolate(n, z_grid, plot=False)
            z_ordered = np.sort(z_ret)
            D_ordered = np.sort(D_interp)

            # Use previous method's definition of Omega_k to set up formula which uses new m & lambda values to compute.
            Omega_k = mu_instance.Omega_k


            if Omega_k == 0:
                D_l = (1 + z_ordered) * D_ordered
            elif Omega_k < 0:
                D_l = (1 + z_ordered) * (c / mu_instance.H0) * (1 / np.sqrt(abs(Omega_k))) * np.sin(np.sqrt(abs(Omega_k)) * ((D_ordered * mu_instance.H0 ) / c))
            else:
                D_l = (1 + z_ordered) * (c / mu_instance.H0) * (1 / np.sqrt(abs(Omega_k))) * np.sinh(np.sqrt(abs(Omega_k)) * ((D_ordered * mu_instance.H0) / c))

            # Solves bug where z = 0 gives a D_l of 0, which cannot be logged.
            D_l = np.where(D_l > 0, D_l, np.nan)

            # Returns mu array for y - axes of plots.
            mu = 5 * np.log10(D_l) + 25


            return z_ordered, mu

    
        # Select sensible Hubble constants that are close to known value, but far enough away to show trends.
        H0_values = [68, 72, 76]

        # Loop through nested function using new instance.
        for H0_vals in H0_values:

            z_plot, mu_plot = mu_for_values(H0_vals, 0.3, 0.7)

            # Plot with for loop, to access all three lines at once.
            plt.plot(z_plot, mu_plot, label = f'H0 = {H0_vals} km/s/Mpc')


        plt.xlabel('Redshift (z) [-]')
        plt.ylabel('Distance Modulus (mu) [mag]')
        # \\ used when importing syntax, one \ will cause an invalid escape sequence.
        plt.title('Distance Modulus vs Redshift for varying H0  \n ($\\Omega_m$ = 0.3, $\\Omega_\\lambda$ = 0.7)')

        # Some axes limits which show a closer look at high redshift trends of varying Hubble's constant.
        #plt.ylim(37, 46)
        #plt.xlim(0, 1)
        plt.legend()
        plt.show()


        Omega_m_values = [0.1, 0.3, 0.5]
        for Omega_m in Omega_m_values:
            z_plot, mu_plot = mu_for_values(self.H0, Omega_m, self.Omega_lambda)


            plt.plot(z_plot, mu_plot, label = f'$\\Omega_m$ = {Omega_m}')

        plt.xlabel('Redshift (z) [-]')
        plt.ylabel('Distance Modulus (mu) [mag]')
        plt.title(f'Distance Modulus vs Redshift for varying $\\Omega_m$  \n (H0 = {self.H0} km/s/Mpc , $\\Omega_\\lambda$ = {self.Omega_lambda})')
        
        # Some axes limits which show a closer look at high redshift trends of varying Omega_m.
        #plt.ylim(40, 46)
        #plt.xlim(0.2, 1)
        plt.legend()
        plt.show()

        
        Omega_lambda_values = [0.5, 0.7, 0.9]
        for Omega_lambda in Omega_lambda_values:
            z_plot, mu_plot = mu_for_values(self.H0, self.Omega_m, Omega_lambda)
            plt.plot(z_plot, mu_plot, label=f'$\\Omega_\\lambda$={Omega_lambda}')
        plt.xlabel('Redshift (z) [-]')
        plt.ylabel('Distance Modulus (mu) [mag]')
        plt.title(f'Distance Modulus vs Redshift for varying $\\Omega_\\lambda$  \n (H0 = {self.H0} km/s/Mpc , $\\Omega_m$ = {self.Omega_m})')

        # Some axes limits which show a closer look at high redshift trends of varying Omega_lambda.
        #plt.ylim(40, 45)
        #plt.xlim(0.2, 1)
        plt.legend()
        plt.show()

        """
        All three plots show that as reshift increases, past approx z = 0.3 - 0.4, that the distnace of objects
        increases as we increase Hubble's constant and Omega_m, and when we decrease Omega_lambda.

        As distance moduli has units of magnitude, these distances in Megaparsecs are significant in comparison.

        """
        
