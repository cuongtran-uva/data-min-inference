"""
Implement a class of methods to detect if a subset R in S is a core feature set
Recall that R in S is a pure core set if f(X_P = x_P, X_R =x_R, X_U) = const for all X_U

"""
from utils import *
from network import *


class PureCoreSet(object):
    def __init__(self, x_train, y_train, x_test, private, params={'w': 1, 'b': 1, 'y_pred': []}):
        """
        y_ln_pred is model prediction over x_test and is given by logistic regression

        """
        self.x_train, self.y_train, self.x_test = x_train, y_train, x_test
        self.private = private
        self.d = self.x_test.shape[1]
        self.public = [i for i in range(self.d) if i not in self.private]
        for key, val in params.items():
            setattr(self, key, val)
        if self.linear:
            self.y_pred = self.logreg.predict(self.x_test)
        else:
            y_pred = self.model(torch.Tensor(x_test)).detach().cpu().numpy()
            self.y_pred = np.argmax(y_pred, axis = 1)

        self.C = self.w.shape[0]  # number of classes in case of multi-class.
        self.power_set = powerset(self.private)

    def set_x(self, x_test):
        self.x_test = x_test.copy()

    def min_core_test_set(self, linear=True, multi_class=True, options={'n_points': 6}):
        res = []
        for i in range(self.x_test.shape[0]):
            res.append(self.min_core_set(self.x_test[i, :], self.y_pred[i], linear, multi_class, options))

        return res

    def min_core_set(self, x, y_pred, linear=True, multi_class=True, options={'n_points': 6}):
        """
        Implement the baseline OPT in the paper
        """
        protect_set_list = []
        y_pred_list = []
        for s in self.power_set:
            if len(s) < len(self.private):
                R = self.public + s
                U = [i for i in range(self.d) if i not in R]
                flag, y_partial_pred = self.is_core_set(x, R, linear, multi_class, options)
                if flag:
                    protect_set_list.append(U)
                    y_pred_list.append(y_partial_pred)

        if len(protect_set_list) == 0:
            return 0, [], y_pred,
        else:
            id_max = np.argmax([len(x) for x in protect_set_list])

            return len(protect_set_list[id_max]), protect_set_list[id_max], y_pred_list[id_max]

    def is_core_set(self, x, R, options={'n_points': 6}):
        n_points = options.get('n_points', 6)

        if self.linear and not self.multi_class:
            return self.is_core_binary_ln(x, R)
        elif self.linear and self.multi_class:
            return self.is_core_multi_ln(x, R)
        else:
            return self.is_core_nln(x, R, n_points)

    def is_core_binary_ln(self, x, R):
        """
        Check if a subset R is a core set for linear binary classifier
        we handle the multi-class here.
        Args:
            R is a  list of indices (public + some private features). E.g., R = [0,1 ,4, 5]

        return  True  if R is actually a pure core feature set, and y_pred is the prediction
        Otherwise return False in case R is not a core feature set
        """
        U = np.array([i for i in range(self.d) if i not in R])
        R = np.array(R)
        if abs(np.dot(x[R], self.w[R]) + self.b) >= np.sum(np.abs(self.w[U])):
            return True, None
        else:
            return False, None

    def is_core_multi_ln(self, x, R):
        """
        Determine if R (a list) a core feature set  in case of multi-classification problem
        This case is more complicated than the binary setting
        :param R:
        :return:
        """
        U = np.array([i for i in range(self.d) if i not in R])
        R = np.array(R)
        current_scores = np.abs(np.dot(x[R], self.w[:, R].T) + self.b)  # current prediction scores when only release
        # features in R .
        i_max = np.argmax(current_scores)  # current class model prediction
        max_alt_score = max(
            [abs(current_scores[i]) + np.sum(np.abs(self.w[i, U])) for i in range(self.C) if i != i_max])
        # The above is the maximum alternate prediction scores if all features in U are revealed.
        if current_scores[i_max] >= max_alt_score:
            return True, None  # it implies reveal features in U will not change the  class prediction
        else:
            return False, None

    def is_core_nln(self, x, R, options):
        """
        Determine if R is a core feature set in case of non-linear-classifiers
        :param R: R is a list of indices of features
               n_points: use  generate n_points^{|U|}  x_U samples equally spaced in [-1, 1]
               to test if the model prediction f(X_R =x_R, X_U =x_U) = const over these x_U samples
        :return:
        """
        n_points = int( np.log(2*10**5)/np.log(len(self.private)))
        U = np.array([i for i in range(self.d) if i not in R])
        R = np.array(R)
        n_samples = n_points ** len(U)
        x_temp_test = np.array([x] * n_samples)
        x_u_replace = linspace_md(-1, 1, len(U), n_points)
        x_temp_test[:, U] = x_u_replace
        x_temp_test = torch.Tensor(x_temp_test)
        y_pred = torch.argmax(self.model(x_temp_test), dim=1)
        y_pred = y_pred.detach().numpy().astype(int)
        if len(np.unique(y_pred)) == 1:
            return True, y_pred[0]
        else:
            return False, -1



