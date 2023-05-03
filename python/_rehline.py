""" ReHLine: Regularized Composite ReHU/ReLU Loss Minimization """

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          C++ support by Yixuan Qiu <yixuanq@gmail.com>

# License: MIT License

import numpy as np
from sklearn.base import BaseEstimator
import rehline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import base

def ReHLine_solver(X, U, V,
        Tau=np.empty(shape=(0, 0)),
        S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)),
        A=np.empty(shape=(0, 0)), b=np.empty(shape=(0)),
        max_iter=1000, tol=1e-4, verbose=True):
    result = rehline.rehline_result()
    rehline.rehline_internal(result, X, A, b, U, V, S, T, Tau, max_iter, tol, verbose)
    return result

class ReHLine(BaseEstimator):
    """Regularized ReLU/ReHU Minimization. (draft version v1.0)

    Parameters
    ----------

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    Attributes
    ----------

    coef_ : array of shape (n_features,)
        Weights assigned to the features (coefficients in the primal
        problem).

    n_iter_: int
        Maximum number of iterations run across all classes.
    """

    def __init__(self, loss={'name':'QR', 'qt':[.25, .75]}, C=1.,
                       U=np.empty(shape=(0,0)), V=np.empty(shape=(0,0)),
                       Tau=np.empty(shape=(0,0)),
                       S=np.empty(shape=(0,0)), T=np.empty(shape=(0,0)),
                       A=np.empty(shape=(0,0)), b=np.empty(shape=(0)),
                       max_iter=1000, tol=1e-4, verbose=False):
        self.loss = loss
        self.C = C
        self.U = U
        self.V = V
        self.S = S
        self.T = T
        self.Tau = Tau
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.L = U.shape[0]
        self.n = U.shape[1]
        self.H = S.shape[0]
        self.K = A.shape[0]

    def make_ReLHLoss(self, X, y, loss={'name':'QR', 'qt':[.25, .75]}):
        """Generate ReLoss params based on the given training data.

        """
        self.loss.update(loss)

        n, d = X.shape

        if (self.loss['name'] == 'hinge') or (self.loss['name'] == 'svm')\
            or (self.loss['name'] == 'SVM'):
            self.U = -(self.C*y).reshape(1,-1)
            self.V = (self.C*np.array(np.ones(n))).reshape(1,-1)
        elif (self.loss['name'] == 'check') \
                or (self.loss['name'] == 'quantile') \
                or (self.loss['name'] == 'quantile regression') \
                or (self.loss['name'] == 'QR'):

            n_qt = len(loss['qt'])
            self.U = np.ones((2, n*n_qt))
            self.V = np.ones((2, n*n_qt))
            X_fake = np.zeros((n*n_qt, d+n_qt))
            
            for l,qt_tmp in enumerate(loss['qt']):
                self.U[0,l*n:(l+1)*n] = - (self.C*qt_tmp*self.U[0,l*n:(l+1)*n])
                self.U[1,l*n:(l+1)*n] = (self.C*(1.-qt_tmp)*self.U[1,l*n:(l+1)*n])

                self.V[0,l*n:(l+1)*n] = self.C*qt_tmp*self.V[0,l*n:(l+1)*n]*y
                self.V[1,l*n:(l+1)*n] = - self.C*(1.-qt_tmp)*self.V[1,l*n:(l+1)*n]*y

                X_fake[l*n:(l+1)*n,:d] = X
                X_fake[l*n:(l+1)*n,d+l] = 1.
            
            self.auto_shape()    
            return X_fake
    
        elif (self.loss['name'] == 'sSVM') \
                or (self.loss['name'] == 'smooth SVM') \
                or (self.loss['name'] == 'smooth hinge'):
            self.S = np.ones((1, n))
            self.T = np.ones((1, n))
            self.Tau = np.ones((1, n))

            self.S[0] = - np.sqrt(self.C)*y
            self.T[0] = np.sqrt(self.C)
            self.Tau[0] = np.sqrt(self.C)
        elif self.loss['name'] == 'TV':
            self.U = np.ones((2, n))*self.C
            self.V = np.ones((2, n))*self.C
            self.U[1] = -self.U[1]
            
            self.V[0] = - X.dot(y)*self.C
            self.V[1] = X.dot(y)*self.C
        elif (self.loss['name'] == 'huber'):
            self.S = np.ones((2, n))
            self.T = np.ones((2, n))
            self.Tau = np.sqrt(self.C) * loss['tau'] * np.ones((2, n))

            self.S[0] = - np.sqrt(self.C)
            self.S[1] =   np.sqrt(self.C)
            self.T[0] = np.sqrt(self.C)*y
            self.T[1] = -np.sqrt(self.C)*y
        elif (self.loss['name'] == 'custom'):
            pass
        else:
            raise Exception("Sorry, ReHLine currently does not support this loss function, \
                            but you can manually set ReLoss params to solve the problem.")
        self.auto_shape()

    def append_l1(self, X, l1_pen=1.0):
        n, d = X.shape
        l1_pen = l1_pen*np.ones(d)
        U_new = np.zeros((self.L+2, n+d))
        V_new = np.zeros((self.L+2, n+d))
        ## Block 1
        if len(self.U):
            U_new[:self.L, :n] = self.U
            V_new[:self.L, :n] = self.V
        ## Block 2
        U_new[-2,n:] = l1_pen
        U_new[-1,n:] = -l1_pen

        if len(self.S):
            S_new = np.zeros((self.H, n+d))
            T_new = np.zeros((self.H, n+d))
            Tau_new = np.zeros((self.H, n+d))

            S_new[:,:n] = self.S
            T_new[:,:n] = self.T
            Tau_new[:,:n] = self.Tau

            self.S = S_new
            self.T = T_new
            self.Tau = Tau_new

        ## fake X
        X_fake = np.zeros((n+d, d))
        X_fake[:n,:] = X
        X_fake[n:,:] = np.identity(d)

        self.U = U_new
        self.V = V_new
        self.auto_shape()
        return X_fake

    def auto_shape(self):
        """
        Automatically generate the shape of the parameters of ReHLine loss functions.
        """
        self.L = self.U.shape[0]
        self.n = self.U.shape[1]
        self.H = self.S.shape[0]
        self.K = self.A.shape[0]

    def call_ReLHLoss(self, input):
        relu_input = np.zeros((self.L, self.n))
        rehu_input = np.zeros((self.H, self.n))
        if self.L > 0:
            relu_input = (self.U.T * input[:,np.newaxis]).T + self.V
        if self.H > 0:
            rehu_input = (self.S.T * input[:,np.newaxis]).T + self.T
        return np.sum(base.relu(relu_input), 0) + np.sum(base.rehu(rehu_input), 0)


    def fit(self, X, sample_weight=None):
        """Fit the model based on the given training data.

        Parameters
        ----------

        X: {array-like} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            An instance of the estimator.

        """
        # X = check_array(X)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        U_weight = self.U * sample_weight
        V_weight = self.V * sample_weight

        if self.H > 0:
            sqrt_sample_weight = np.sqrt(sample_weight)
            Tau_weight = self.Tau * sqrt_sample_weight
            S_weight = self.S * sqrt_sample_weight
            T_weight = self.T * sqrt_sample_weight
        else:
            Tau_weight = self.Tau
            S_weight = self.S
            T_weight = self.T

        result = ReHLine_solver(X=X,
                                U=U_weight, V=V_weight,
                                Tau=Tau_weight,
                                S=S_weight, T=T_weight,
                                A=self.A, b=self.b,
                                max_iter=self.max_iter, tol=self.tol, verbose=self.verbose)

        self.coef_ = result.beta
        self.opt_result_ = result
        self.n_iter_ = result.niter
        self.dual_obj_ = result.dual_objfns

    def decision_function(self, X):
        """The decision function evaluated on the given dataset

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        dec : ndarray of shape (n_samples,)
            Returns the decision function of the samples.
        """
        # Check if fit has been called
        check_is_fitted(self)

        X = check_array(X)
        return np.dot(X, self.coef_)



