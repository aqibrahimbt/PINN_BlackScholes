from scipy.stats import norm
import numpy as np
from .OptionsPricing import OptionsPricing

class BSClosedForm(OptionsPricing):
    # call or put
    def price(self):
        d1 = ((self.r + 0.5 * self.sigma**2) * self.T - np.log(self.K / self.S0)) / (self.sigma * np.sqrt(self.T))
        d2 = ((self.r - 0.5 * self.sigma**2) * self.T - np.log(self.K / self.S0)) / (self.sigma * np.sqrt(self.T))
        return (
            self.S0 * norm.cdf(d1)
            - np.exp(-self.r * self.T) * self.K * norm.cdf(d2)
            if self.is_call
            else np.exp(-self.r * self.T) * self.K * norm.cdf(-d2)
            - self.S0 * norm.cdf(-d1)
        )