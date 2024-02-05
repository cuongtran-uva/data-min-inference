"""
Implement class of probablistic core feature set of linear/non-linear classifiers.
"""
from pure_core_set import *
from dist_pred import *
class ProbCoreSet(PureCoreSet):
    def __init__(self, x_train, y_train, x_test, private, params):
        super().__init__(x_train, y_train, x_test, private, params)
        if self.gaussian:
            self.checker = GaussDistPred(x_train,y_train, private, params)
        else:
            self.checker = BayesianDistPred(x_train,y_train, private, params)

    def min_core_test_set(self, options):
        res = []
        for i in range(self.x_test.shape[0]):
            res.append(self.min_core_set(self.x_test[i,:], self.y_pred[i],options))

        return res


    def min_core_set(self, x, y_pred, options):
        """
        Implement the baseline OPT in the paper
        """
        protect_set_list = []
        y_pred_list = []
        for s in self.power_set:
            if len(s) < len(self.private):
                R = self.public + s
                U = [i for i in range(self.d) if i not in R]
                flag, y_partial_pred = self.is_core_set(x, R, options)
                if flag:
                    protect_set_list.append(U)
                    y_pred_list.append(y_partial_pred)

        if len(protect_set_list) == 0:
            return 0, [], y_pred,
        else:
            id_max = np.argmax([len(x) for x in protect_set_list])

            return len(protect_set_list[id_max]), protect_set_list[id_max], y_pred_list[id_max]

    def is_core_set(self, x, R, options):
        U = [i for i in range(self.d) if i not in R]
        return self.checker.is_core_set(x, U, options)





