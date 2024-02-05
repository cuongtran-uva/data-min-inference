"""
Implement a class that compute  the distribution of model prediction P(f(X_U, X_R =x_R)
given the distribution X_U |X_R =x_R and model parameter f.
as from a Gaussian or from general distribution
"""
from scipy.stats import norm
from utils import *
from cond_dist import *

class DistPred(object):
    def __init__(self,  x_train, y_train, private, params):
        for key, val in params.items():
            setattr(self, key, val)
        self.x_train = x_train
        self.y_train = y_train
        self.private = private
        self.mu = np.mean(self.x_train, axis=0)
        self.Sigma = np.cov(self.x_train, rowvar=False)
        self.d = self.x_train.shape[1]
        self.C = len(np.unique(self.y_train))
        self.multi_class = 1 if self.C > 2 else 0



class GaussDistPred(DistPred):
    def __init__(self, x_train ,y_train, private, params):
        super().__init__(x_train, y_train, private, params)
        self.cond_dist = GaussCondDist(x_train, y_train, private)

    def get_dist_pred(self, x, U, options = { 'n_mc':10**3}):
        if self.linear:
            if not self.multi_class:
                return self.get_dist_bin_ln(x, U)
            else:
                return self.get_dist_mult_ln(x, U, options)

        else:
            return self.get_dist_nln(x, U, options)

    def get_dist_bin_ln(self, x, U):
        """
        Compute model prediction in case when X_U |X_R =x_R ~N(mu, Sigma)
        and in case the classifier is linear + binary
        :param x: 1xd Numpy, one single testing sample that have d features
        :param U:  U in [d] is a list of indices of features that ARE NOT revealed
        :return:prediction and confidence of prediction. E.g, given all features released in R
        only features in U is not revealed, we believe the model prediction is 1 with prob 90%.
        """
        R = np.array([ i for i in range(self.d) if i not in U])
        mu_U, Sigma_U = self.cond_dist.compute_cond_gauss(x, U) # TO DO
        curr_score = np.dot(self.w[R], x[R]) + self.b# current scores applied to revealed features in R
        new_score = np.dot(self.w[U], mu_U)
        total_score = curr_score + new_score
        std = max( np.sqrt( np.dot( np.dot(self.w[U], Sigma_U), self.w[U])), 1e-6)
        conf = norm.cdf(- total_score/std)
        p_list = [conf, 1-conf]
        if total_score >=0:
            return 1, np.max(p_list), p_list
        else:
            return 0, np.max(p_list), p_list

    def get_dist_mult_ln(self, x, U, options):
        n_mc = options.get('n_mc', 100)
        samples = self.cond_dist.draw_multivariate_samples(x, U, options  = {'n_mc':n_mc})
        x_rnd = np.array([x]*n_mc).reshape(n_mc, self.d)
        x_rnd[:,U] = samples
        pred = np.dot( x_rnd, self.w.T) + self.b
        y_hard_pred = np.argmax(pred, axis = 1)
        p_list = [len([y for y in y_hard_pred if y == i]) / float(n_mc) for i in range(self.C)]
        return np.argmax(p_list), np.max(p_list), p_list

    def is_core_set(self,x, U, options):
        if self.linear:
            if self.multi_class:
                res = self.get_dist_mult_ln(x, U, options)
                return res[1] >=1- options['delta'], res[0]
            else:
                res = self.get_dist_bin_ln(x, U)
                return res[1] >=1- options['delta'], res[0]
        else:
            res = self.get_dist_nln(x,U,options)
            return res[1] >=1 - options['delta'] , res[0]

    def get_dist_nln(self, x, U, options = {'n_mc': 10*3}):
        """
        Compute the distribution of model distribution in case of general non-linear/multi-class
        classifiers i.e, compute P(f_theta(X_U ,X_R =x_R)), given X_U is multivariate-random variables
        :param x: the instance x that we want to compute distribution
        :param U: the unknown(unrevealed) features
        :param options: dictionary provide some additional info like number of Monte Carlo samples
        :return: model prediction, with confidence
        """
        n_mc = options.get('n_mc', 10**2)
        mu_U, Sigma_U =  self.cond_dist.compute_cond_gauss(x, U)  # TO DO
        x_rnd = np.array([x]*n_mc).reshape(n_mc,-1) # should be n_mcxd numpy, d is the number of features
        x_rnd[:, U] = np.random.multivariate_normal(mu_U, Sigma_U,n_mc)
        y_pred = self.model(torch.Tensor(x_rnd)).detach().numpy()
        y_pred = np.argmax(y_pred, axis=1) # y_pred[i] is prediction for i-th row among n_mc rows
        p_list = [len([y for y in y_pred if y==i])/float(n_mc) for i in range(self.C)]

        return np.argmax(p_list), np.max(p_list), p_list

    def get_entropy_pred(self, x, U, options ={'n_mc':10**3}):

        prob_vec = self.get_dist_pred(x, U, options)

        return get_entropy(prob_vec)

class BayesianDistPred(object):

    def __init__(self, x_train, y_train, private, params):
        """
        Pass
        """
        super().__init__(x_train, y_train, private, params)
        self.cond_dist = BayesianCondDist(x_train, y_train, private)

    def get_pred(self, x):
        if self.linear:
            return self.logreg.predict(x)
        else:
            y_pred = self.model(torch.Tensor(x)).detach().cpu().numpy()
            return np.argmax(y_pred, axis = 1)

    def get_dist_pred(self, x, U, options):
        n_mc = options.get('n_mc', 100)
        samples = self.cond_dist.draw_multivariate_samples(x, U, options = {'n_mc':n_mc})
        x_rnd = np.array([x] * n_mc).reshape(n_mc, -1)  # should be n_mcxd numpy, d is the number of features
        x_rnd[:, U] = samples
        y_pred = self.get_pred(x_rnd)
        p_list = [len([y for y in y_pred if y == i]) / float(n_mc) for i in range(self.C)]
        return np.argmax(p_list), np.max(p_list), p_list

    def is_core_set(self, x, U, options):

        return self.get_dist_pred(x, U, options)[1] >=1- options['delta']























