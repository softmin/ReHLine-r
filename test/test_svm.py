## SVM on simulated dataset
import numpy as np
import pandas as pd
from qp_solver import qp_admm, qp_cvxpy
import cvxpy as cp
from numpy.linalg import norm
from sklearn.datasets import make_classification
from obj import P_obj

X, y = make_classification(n_samples=100, n_features=20, random_state=0)
y = 2*y - 1
C = .5

### Test for SVM

## generate dataset for `L3-solver`
n, d = X.shape
U = -(C*y).reshape(1,-1)
L = U.shape[0]
V = (C*np.array(np.ones(n))).reshape(1,-1)

np.savez('./dataset/sim/exp_svm', X=X, y=y, U=U, V=V)

svm_data = np.load('/home/ben/github/L3-solver/test/dataset/sim/exp_svm.npz')
X, y, U, V = svm_data['X'], svm_data['y'], svm_data['U'], svm_data['V']

# test
assert U.shape[1] == n
assert V.shape[0] == L
assert V.shape[1] == n

## solution by liblinear
from sklearn.svm import LinearSVC
clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=0, tol=1e-6, max_iter=1000000)
clf.fit(X, y)
sol = clf.coef_.flatten()

## parameters for a general QP
Ux = np.array([U.T*X])
mat_Ux = np.reshape(Ux, (-1, d)).T
P = np.dot(mat_Ux.T, mat_Ux) + 1e-8*np.eye(L*n)
q = -V.flatten()
lb = np.zeros(L*n)
ub = np.ones(L*n)

## solution by ADMM
Dsol_admm, __ = qp_admm(P, q, lb, ub, atol=1e-5, rtol=1e-5)
Psol_admm = - mat_Ux.dot(Dsol_admm)

## solution by CVXOPT
Dsol_cvxopt, __ = qp_cvxpy(P, q, lb, ub)
Psol_cvxopt = - mat_Ux.dot(Dsol_cvxopt)

## Check solution

## Compare the results for different methods
print('diff: admm-liblinear: %.4f' %norm(Psol_admm - sol))
print('diff: cvxopt-liblinear: %.4f' %norm(Psol_cvxopt - sol))
print('diff: cvxopt-admm: %.4f' %norm(Psol_cvxopt - Psol_admm))

# diff: admm-liblinear: 0.0004
# diff: cvxopt-liblinear: 0.0000
# diff: cvxopt-admm: 0.0004

## liblinear
# array([ 0.40693637,  0.26570757,  0.04150625,  0.95287774,  0.78164026,
#         0.05321627, -0.03329784,  0.03606953,  0.04388525, -0.17519925,
#         0.71490641,  0.16579052, -0.17195273, -0.1856494 ,  0.41504592,
#         0.0999136 ,  0.20471563,  0.23528203,  0.14956129, -0.40864474])

## ADMM
# array([ 0.4071422 ,  0.26565045,  0.04137691,  0.95300578,  0.78177337,
#         0.05313382, -0.03334962,  0.0360213 ,  0.04383611, -0.17523862,
#         0.71498571,  0.16573559, -0.1719371 , -0.18556746,  0.41523198,
#         0.09990122,  0.20475539,  0.23534489,  0.14954798, -0.40868137])

## CVXOPT
# array([ 0.40693616,  0.26570768,  0.04150637,  0.95287762,  0.78164012,
#         0.0532164 , -0.03329769,  0.03606962,  0.04388544, -0.17519926,
#         0.71490634,  0.16579061, -0.17195275, -0.1856495 ,  0.41504564,
#         0.09991359,  0.20471551,  0.23528196,  0.14956124, -0.40864464])

## check objective function
obj_true = C * np.sum(np.maximum(1 - y[:,np.newaxis] * X @ sol, 0)) + .5*np.sum(sol**2)

assert obj_true == P_obj(X=X, beta=sol, U=U, V=V, S=0, T=0, tau=0)

print('obj: linlinear: %.4f' %P_obj(X=X, beta=sol, U=U, V=V, S=0, T=0, tau=0))
# obj: linlinear: 11.6221

print('obj: admm: %.4f' %P_obj(X=X, beta=Psol_admm, U=U, V=V, S=0, T=0, tau=0))
# obj: admm: 11.6225

print('obj: cvxopt: %.4f' %P_obj(X=X, beta=Psol_cvxopt, U=U, V=V, S=0, T=0, tau=0))
# obj: cvxopt: 11.6221