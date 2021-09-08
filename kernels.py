import numpy as np

class Gaussian_kernel:

    #takes in feature points where axis = 0 -->represents features
    #                              axis = 1 -->represents each point
    def __init__(self,feature_pointes = None,sigma = 0.1):
        self.feature_points = feature_pointes
        self.sigma = sigma
        self.type = "gaussian"


    def transform(self,X,feature_points = None):

        #check if feature points are passed
        if feature_points is not None:
            self.feature_points = feature_points
        elif self.feature_points is None:
            self.feature_points = X

        gram_matrix = np.zeros((self.feature_points.shape[1],X.shape[1]))

        #now we transform
        for i,x in enumerate(X.T):
            x = np.expand_dims(x,axis=1)
            norm_sq = np.linalg.norm(self.feature_points - x,axis=0)**2
            pow = -norm_sq/(2*(self.sigma**2))
            gram_matrix[:,i] = np.exp(pow)

        return gram_matrix

class Linear_kernel:
    def __init__(self):
        self.type = 'linear'

    def transform(self,X):
        return X


