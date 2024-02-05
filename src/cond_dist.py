"""
Implement a class that compute P(X_U |X_R) and P(X_i |X_R) for i in U
"""
from utils import *
from bayesian_inference import *

class GaussCondDist(object):
    def __init__(self, x_train, y_train, private):
        """
        Pass
        """
        self.x_train = x_train
        self.y_train = y_train
        self.private = private
        self.mu = np.mean(self.x_train, axis=0)
        self.d = self.x_train.shape[1]
        self.Sigma = np.cov(self.x_train, rowvar=False)

    def compute_cond_gauss(self, x, U):
        """
        Compute P(x_U |X_R = x_R)  given that X  ~N(m, Sigma)
        :param x:  dx1 numpy for one instance sample
        :param U:  sorted list of indices of unrevealed features
        :return: E[X_U] and Cov[X_U].
        """

        R = [i for i in range(self.d) if i not in U]
        m_U, m_R = self.mu[U], self.mu[R]
        # need a better concise way to rewrite the following both cases
        if len(R) == 1:
            inv_mat = 1 / self.Sigma[R, R]
            temp = self.Sigma[np.ix_(U, R)] * inv_mat * (x[R] - self.mu[R])
            temp = temp.reshape(-1)
            cond_mean = m_U + temp
        else:
            inv_mat = np.linalg.inv(self.Sigma[np.ix_(R, R)])
            cond_mean = m_U + np.dot(np.dot(self.Sigma[np.ix_(U,R)], inv_mat), (x[R] - self.mu[R]))

        cond_sigma = self.Sigma[np.ix_(U, U)] - np.dot(np.dot(self.Sigma[np.ix_(U, R)], inv_mat),
                                               self.Sigma[np.ix_(R, U)])

        return cond_mean, cond_sigma

    def compute_univ_cond_gauss(self, x, U, i):
        cond_mean, cond_sigma = self.compute_cond_gauss(x, U)
        idx_found = U.index(i)

        return cond_mean[idx_found], cond_sigma[idx_found, idx_found]

    def draw_multivariate_samples(self, x, U, options ={'n_mc':100}):
        if U!=sorted(U):
            print('WARNING, U should be sorted')
        cond_mean, cond_sigma = self.compute_cond_gauss(x, U)
        if not is_pos_def(cond_sigma):
          cond_sigma +=1e-4* np.ones(cond_sigma.shape[0])

        return np.random.multivariate_normal(cond_mean, cond_sigma, options['n_mc'])

    def draw_univariate_samples(self, x, U, i, options = {'n_mc':100}):
        """
        Compute P(X_i |X_R= x_R), where U = [d] \R, i in U.
        Basically what is the unknow income given the reveealed job, education.,
        :param x: nx1 numpy object
        :param U:  list of unrevealed features
        :param i: one single number
        :return: n_mcx1 numpy object
        """
        cond_mean, cond_sigma = self.compute_univ_cond_gauss( x, U, i)
        cond_sigma = max(cond_sigma, 1e-10)
        return np.random.normal(cond_mean, np.sqrt(cond_sigma), options['n_mc'])



class BayesianCondDist(object):
    def __init__(self, x_train, y_train, private):
        """
        Pass
        """
        self.x_train = x_train
        self.y_train = y_train
        self.private = private
        self.d = self.x_train.shape[1]
        self.public = [i for i in range(self.d) if i not in self.private]
        self.bayes_dict = construct_bayes_dict(self.x_train, self.public, self.private)

    def draw_multivariate_samples(self, x, U, options = {'n_mc':100}):
        """
        Draw the samples from P(X_U |X_R =x_R), where x_R is contrained in x

        """
        if U!=sorted(U):
            print('Warning, the input should already sorted')

        R = [i for i in range(self.d) if i not in U]
        return infer_bayesian_lr( self.bayes_dict[tuple(R)]['model'], x, R, n_mc=options['n_mc'])

    def draw_univariate_samples(self, x, U, i, options={'n_mc':100}):
        i_found = U.index(i)
        return self.draw_multivariate_samples(x, U, options)[:,i_found]




