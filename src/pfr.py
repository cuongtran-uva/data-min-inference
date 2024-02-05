"""
Implement the PFR mechanism
"""
from prob_core_set import *
from pure_core_set import *
from dist_pred import *

class FPR (object):
    def __init__(self, x_train,y_train,x_test, params):
        """

        :param x_train: nxd numpy matrix of training input
        :param y_train: nx1 numpy vector of training output
        :param x_test: mxd numpy vector testing inputs
        :param params: provide several parameters such as linear, multiclass
        """
        self.x_train, self.y_train, self.x_test = x_train, y_train, x_test
        for key, val in params.items():
            setattr(self, key, val)
        if type(self.private) != list:
            self.private = self.private.tolist()
        self.d = self.x_train.shape[1]

        self.public = [i for i in range(self.d) if i not in self.private]

        if params['choice'] =='prob':
            self.core_set_test = ProbCoreSet(self.x_train, self.y_train, self.x_test, self.private, params)
        else:
            self.core_set_test = PureCoreSet(self.x_train,self.y_train, self.x_test, self.private,params)

        if self.gaussian:
            self.cond_dist = GaussCondDist(self.x_train, self.y_train, self.private)
            self.dist_pred = GaussDistPred(self.x_train, self.y_train, self.private, params)
        else:
            self.cond_dist = BayesianCondDist(self.x_train, self.y_train, self.private)
            self.dist_pred = BayesianDistPred(self.x_train, self.y_train, self.private, params)

        if self.linear:
            self.y_pred = self.logreg.predict(self.x_test)
        else:
            y_pred = self.model(torch.Tensor(self.x_test)).detach().cpu().numpy()
            self.y_pred = np.argmax(y_pred, axis =1)

    def compute_single_exp_entropy(self, x, U, i_cand, options ):
        """

        :param x: dx1 Numpy object, one instance sample
        :param i:  index of one unrevealed features, this is a single value
        :param U: is a list of unrevealed features. So i should be in U
        :return: The score if we reveal features i-th.
        """
        samples = self.cond_dist.draw_univariate_samples(x,U, i_cand,  {'n_mc':50})
        U_left = [j for j in U if j!=i_cand]
        entropy_list = []
        for i in range(len(samples)):
            x_cp = x.copy()
            x_cp[i_cand] = samples[i]
            entropy_list.append(get_entropy(p) for p in self.dist_pred.get_dist_pred(x_cp, U_left,  options)[2])

        return np.mean(entropy_list)

    def compute_exp_entropy(self, x, U, options):
        scores_list = []
        for i in U:
            scores_list.append( - self.compute_single_exp_entropy(x, U, i,options))
            # we minus since we want to minimize entropy, or maximize minus entropy
        return scores_list

    def compute_score(self, x, U, options):
        """
        Compute the score for feach feature in U (the set of unrevealed features)
        NOTE THAT the better the score, the more likeky the feature is selected
        :param U:  is the sorted list of unrevealed
        :param options:  provide options like method
        :return:
        """
        if options['score'] == 'feat_imp':
            if not self.multi_class:
                return abs(self.w[U]) # we should choose features that are more important
            else:
                return np.linalg.norm(self.w[:,U],axis =1)
        elif options['score'] =='rand':
            return np.random.permutation(range(len(U)))
        else:
            return self.compute_exp_entropy(x, U, options)


    def run_test_set(self, options):
        res = []
        for i in range(self.x_test.shape[0]):
            temp_res = self.online_learning(self.x_test[i,:], self.y_pred[i], options)
            res.append(temp_res)

        return res

    def online_learning(self, x, y_pred, options ):
        U = self.private.copy()
        S = []
        while True:
            U = sorted(U)
            R = sorted(self.public + S)
            core_set_flag, pred = self.core_set_test.is_core_set(x,R, options )
            if core_set_flag:
                return U, pred
                break
            else:
                if len(U)==1: # case only one unreveealed feature left, and still not a core set,
                    return [], y_pred
            scores_list = self.compute_score(x, U, options)
            idx_max = np.argmax(scores_list)
            S.append(U[idx_max])
            U.remove(U[idx_max])










