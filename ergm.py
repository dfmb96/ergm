import random
import numpy as np
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
import warnings

from matplotlib import pyplot as plt
warnings.filterwarnings("error")


class ergm:
    def __init__(self, n_nodes, features, coefs = None):
        self.features = features
        self.coefs = [0.] * len(features) if coefs is None else coefs
        self.dim = len(features)
        self.n_nodes = n_nodes
    
    def log_likelihood_frac(self, coefs, G1, G2):
        return np.sum( coefs * np.array([np.sum([f(G1[i]) - f(G2[i]) for i in range(len(G1))]) for f in self.features]))
    
    def likelihood_frac(self, coefs, G1, G2):
        return np.exp(self.log_likelihood_frac(coefs, G1, G2))

    def log_unnormed_likelihood(self, coefs, G):
        return np.sum(coefs * np.sum([[f(g) for f in self.features] for g in G], axis=0))
    
    def unnormed_likelihood(self, coefs, G):
        return np.exp(self.log_unnormed_likelihood(coefs, G))
    
    @staticmethod
    def generate_random_erdos_renyi(n, p):
        random.seed()

        x = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                if random.random() < p:
                    x[i][j] = 1
                    x[j][i] = 1

        return x

    def gibbs_sample(self, coefs=None, warmup_num=0, size=50, starting_point=None):
        random.seed()
        sampling_coefs = self.coefs if coefs is None else coefs

        x_0 = self.generate_random_erdos_renyi(self.n_nodes, 0.5) if starting_point is None else starting_point
        res = [x_0]

        for _ in range(1, warmup_num + size):
            x_curr = res[-1].copy()
            x_new = x_curr.copy()
            for i in range(1, self.n_nodes):
                for j in range(i):
                    old_val = x_curr[i, j]
                    new_val = int(not old_val)
                    x_new[i, j], x_new[j, i] = new_val, new_val

                    acceptance_rate = self.likelihood_frac(sampling_coefs, [x_new], [x_curr])
                    u = random.random()
                    if u < acceptance_rate:
                        x_curr = x_new.copy()
                    else:
                        x_new = x_curr.copy()

            res.append(x_curr)

        return np.array(res[warmup_num:])
    
    def fit(self, X, n_iterations=10500, s=0.2, method='DMH', warmup_num=500, verbose=1):
        assert method in ('DMH', 'MCMCMLE', 'MPLE'), 'Unknown method'
        assert X.shape[1] == X.shape[2], 'Adjacency matrices in the sample should be square'
        assert X.shape[1] == self.n_nodes, 'Invalid number of nodes in sample'
        
        n_sample = X.shape[0]

        if method == 'DMH':
            coefs_0 = self.fit(X, method='MPLE')
            coefs = [coefs_0]
            
            for cnt in tqdm(range(n_iterations), disable=not verbose):
                curr_coefs = coefs[-1]

                if verbose > 1:
                    if cnt % 1000 == 0:
                        print(curr_coefs)

                new_coefs = np.random.multivariate_normal(curr_coefs, np.identity(self.dim) * s**2)

                Y = [self.gibbs_sample(coefs=new_coefs, size=2, starting_point=x)[-1] for x in X]
                
                log_acceptance_rate = self.log_likelihood_frac(curr_coefs, Y, X) + self.log_likelihood_frac(new_coefs, X, Y)

                u = random.random()
                
                if np.log(u) < log_acceptance_rate:
                    coefs.append(new_coefs)
                else:
                    coefs.append(curr_coefs)
        
            self.coefs = np.mean(coefs[warmup_num:], axis=0)

        elif method == 'MCMCMLE':
            def w(new_coefs, old_coefs, sample):
                estimate_normalization_frac = np.log(np.mean([self.unnormed_likelihood(new_coefs - old_coefs, [s]) for s in sample]))
                sample_log_unnormed_likelihood = self.log_unnormed_likelihood(new_coefs - old_coefs, X)
                res = -sample_log_unnormed_likelihood + n_sample * estimate_normalization_frac
                return res
            
            coefs = self.fit(X, method='MPLE')

            for _ in range(warmup_num):
                sample = self.gibbs_sample(coefs=coefs, warmup_num=1000, size=1000)
                coefs = minimize(w, coefs, args=(coefs, sample), method='L-BFGS-B').x

            self.coefs = coefs

        elif method == 'MPLE':
            train, target = [], []
            for x in X:
                for i in range(1, self.n_nodes):
                    for j in range(i):
                        x_0, x_1 = x.copy(), x.copy()
                        x_0[i, j] = 0
                        x_1[i, j] = 1
                        
                        train.append([f(x_1) - f(x_0) for f in self.features])
                        target.append(x[i, j])

            clf = LogisticRegression(C=0.1, fit_intercept=False, random_state=42)
            clf.fit(train, target)
            self.coefs = clf.coef_[0]

        return self.coefs