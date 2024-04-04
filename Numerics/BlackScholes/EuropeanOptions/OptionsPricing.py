
class OptionsPricing(object):
    def __init__(self, S0, K, r, T_start, T_end, sigma, u0, is_call=True):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T_start = T_start
        self.T_end = T_end
        self.sigma = sigma
        self.u0 = u0
        self.is_call = is_call
