##' Solving Elastic Net Regularized Huber Regression
##'
##' @description
##' This function solves the elastic net regularized Huber regression
##' (ElasticHuber) of the following form:
##' \deqn{
##' \min_{\beta}\ \frac{1}{n}\sum_{i=1}^n H_\kappa(y_i-x_i^T \beta)+
##' \lambda_1\Vert \beta \Vert_1 + \frac{\lambda_2}{2}\Vert \beta \Vert_2^2
##' }
##' where \eqn{H_\kappa(\cdot)}{H_\kappa(.)} is the Huber loss with parameter \eqn{\kappa},
##' \deqn{
##' H_\kappa(z)=
##' \begin{cases}
##' z^2/2,                & |z|\le\kappa \\
##' \kappa(|z|-\kappa/2), & |z|>\kappa
##' \end{cases},
##' }
##' \eqn{\beta\in\mathbb{R}^d}{\beta} is a length-\eqn{d} vector,
##' \eqn{x_i\in\mathbb{R}^d}{x_i} is the feature vector for the \eqn{i}-th observation,
##' \eqn{y_i\in\mathbb{R}}{y_i} is the \eqn{i}-th response variable value,
##' and \eqn{\lambda_1,\lambda_2>0} are weights of lasso and ridge penalties,
##' respectively.
##'
##' @param x          The data matrix \eqn{X=(x_1,\ldots,x_n)^T} of size
##'                   \eqn{n\times d}{n * d}, representing \eqn{n} observations
##'                   and \eqn{d} features.
##' @param y          The length-\eqn{n} response vector.
##' @param kappa      Parameter of the Huber loss.
##' @param lam1,lam2  Weights of lasso and ridge penalties, respectively.
##' @param max_iter   Maximum number of iterations.
##' @param tol        Tolerance parameter for convergence test.
##' @param shrink     Whether to use the shrinkage algorithm.
##' @param verbose    Level of verbosity.
##' @param trace_freq Trace objective function values every \code{trace_freq}
##'                   iterations. Only works if \code{verbose > 0}.
##'
##' @return A list of the following components:
##' \item{beta}{Optimized value of the \eqn{\beta} vector.}
##' \item{xi,Lambda,Gamma}{Values of dual variables.}
##' \item{niter}{Number of iterations used.}
##' \item{dual_objfns}{Dual objective function values during the optimization process.}
##' \item{primal_objfns}{Primal objective function values during the optimization process.}
##'
##' @author Yixuan Qiu \url{https://statr.me}
##'
##'         Ben Dai \url{https://bendai.org}
##'
##' @examples
##' # Simulate data
##' set.seed(123)
##' n = 5000
##' d = 100
##' x = matrix(rnorm(n * d), n, d)
##' beta0 = rnorm(d)
##' y = c(x %*% beta0) + rnorm(n, sd = 0.1)
##' kappa = IQR(y) / 10
##' lam1 = 0.05
##' lam2 = 0.05
##'
##' # Objective function value
##' huber = function(x, kappa)
##' {
##'     xabs = abs(x)
##'     ifelse(xabs <= kappa, xabs^2 / 2, kappa * (xabs - kappa / 2))
##' }
##' objfn = function(beta, x, y, kappa, lam1, lam2)
##' {
##'     d = length(beta)
##'     pred = c(x %*% beta)
##'     loss = huber(y - pred, kappa)
##'     reg = lam1 * sum(abs(beta)) + lam2 * sum(beta^2) / 2
##'     mean(loss) + reg
##' }
##'
##' # ReHLine
##' res1 = elastic_huber(x, y, kappa = kappa, lam1 = lam1, lam2 = lam2,
##'                      max_iter = 1000, tol = 1e-5)
##' bhat1 = res1$beta
##' obj1 = objfn(bhat1, x, y, kappa, lam1, lam2)
##' print(obj1)
##'
##' # Reparameterization of penalty parameters
##' # lam1 = lambda * alpha
##' # lam2 = lambda * (1 - alpha)
##' # => alpha = lam1 / (lam1 + lam2), lambda = lam1 + lam2
##' if (require(hqreg))
##' {
##'     res2 = hqreg_raw(x, y, method = "huber", gamma = kappa,
##'         alpha = lam1 / (lam1 + lam2), lambda = c(1, lam1 + lam2),
##'         intercept = FALSE, max.iter = 1000, eps = 1e-5)
##'     bhat2 = as.numeric(res2$beta[, 2])
##'     obj2 = objfn(bhat2, x, y, kappa, lam1, lam2)
##'     print(obj2)
##'
##'     # Compare estimates
##'     plot(bhat1, bhat2)
##'     abline(0, 1, col = "red")
##' }
elastic_huber = function(x, y, kappa = 1.0, lam1 = 0.1, lam2 = 0.1,
    max_iter = 1000, tol = 1e-5, shrink = TRUE, verbose = 0, trace_freq = 100)
{
    n = nrow(x)
    d = ncol(x)
    c1 = sqrt(1 / (n * lam2))
    c2 = lam1 / lam2

    Xmat = rbind(x, diag(d))

    Uc1 = matrix(0, 2, n)
    Uc2 = rbind(rep(c2, d), rep(-c2, d))
    Umat = cbind(Uc1, Uc2)
    Vmat = matrix(0, 2, n + d)

    Sc1 = rbind(rep(-c1, n), rep(c1, n))
    Sc2 = matrix(0, 2, d)
    Smat = cbind(Sc1, Sc2)
    Tc1 = rbind(c1 * y, -c1 * y)
    Tc2 = Sc2
    Tmat = cbind(Tc1, Tc2)
    Tauc1 = matrix(c1 * kappa, 2, n)
    Tauc2 = Sc2
    Tau = cbind(Tauc1, Tauc2)

    res = rehline(Xmat, Umat, Vmat, Smat, Tmat, Tau, max_iter = max_iter, tol = tol,
                  shrink = shrink, verbose = verbose, trace_freq = trace_freq)
    res
}
