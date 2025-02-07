import pandas as pd
import numpy as np
import scipy.stats as stats


file = pd.read_csv(("Small-diameter-flow.csv"), index_col = 0)



class LinearRegression:

    def __init__(self, X, Y):
        
        self.X = X
        self.Y = Y
        
    @property
    #dimensions
    def d(self):
        b = np.linalg.pinv (self.X.T @ self.X) @ self.X.T @ self.Y #b0 - intercept

        d = len(b)-1
        return d
    
    @property
    #sizesample
    def n(self):
        n = self.Y.shape[0]
        return n
    
    
    
    ###########################
    
    def b0(self):
        
        b = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
        return b
    

    def variance(self):
        SEE = np.sum(np.square(self.Y - (self.X @ self.b0())))  
        variance = SEE / ( self.n - self.d - 1) 
        return variance


    def standard_deviation(self):
        return np.sqrt(self.variance()) # S i lektion 4
    

    def significance(self):
        
        Syy = (self.n*np.sum(np.square(self.Y)) - np.square(np.sum(self.Y)))/self.n        
        SEE = np.sum(np.square(self.Y - (self.X @ self.b0()))) 
        SSR = (Syy - SEE)
        
        
        sig_statistic = (SSR / self.d) / self.standard_deviation() #lektion 6
        p_significance = stats.f.sf(sig_statistic, self.d, self.n - self.d - 1)
        return p_significance
    
    


    def R2(self):
        ssr = np.sum(np.square((self.X @ self.b0()) - self.Y.mean())) 
        sst = ssr + np.sum(np.square(self.Y - self.Y.mean())) 
        return ssr / sst
    
