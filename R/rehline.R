##' Solving Regularized Composite ReLU-ReHU Loss Minimization Problems
##'
##' @description
##' This function solves the regularized composite ReLU-ReHU minimization
##' problem (called \emph{ReHLine optimization} for short) of the following form:
##' \deqn{
##' \min_{\beta}\ \sum_{i=1}^n \sum_{l=1}^L\mathrm{ReLU}(u_{li}x_i^T \beta+v_{li})+
##' \sum_{i=1}^n \sum_{h=1}^H\mathrm{ReHU}_{\tau_{hi}}(s_{hi}x_i^T \beta+t_{hi})+
##' \frac{1}{2}\Vert \beta \Vert^2
##' }
##' subject to general linear constraints \eqn{A\beta+b\ge 0}{A * beta >= 0},
##' where \eqn{\beta\in\mathbb{R}^d}{\beta} is a length-\eqn{d} vector,
##' \eqn{x_i\in\mathbb{R}^d}{x_i} is the feature vector for the \eqn{i}-th observation,
##' \eqn{U=(u_{li})} and \eqn{V=(v_{li})} are \eqn{L\times n}{L * n} matrices,
##' \eqn{S=(s_{hi})}, \eqn{T=(t_{hi})}, and \eqn{\tau=(\tau_{hi})} are
##' \eqn{H\times n}{H * n} matrices, \eqn{A} is an \eqn{m\times d}{m * d} matrix,
##' and \eqn{b} is a length-\eqn{m} vector.
##'
##' The \eqn{\mathrm{ReLU}}{ReLU} function is \eqn{\mathrm{ReLU}(x)=\max(x, 0)}{ReLU(x)=max(x, 0)},
##' and \eqn{\mathrm{ReHU}_\tau}{ReHU} is defined as
##' \deqn{
##' \mathrm{ReHU}_\tau(z)=
##' \begin{cases}
##' 0,              & z\le 0 \\
##' z^2/2,          & 0<z\le\tau \\
##' \tau(z-\tau/2), & z>\tau
##' \end{cases}.
##' }
##'
##' Many popular empirical risk minimization problems can be expressed
##' in the form of ReHLine optimization, such as SVM, quantile regression,
##' Huber regression, etc.
##'
##' @param Xmat                 The data matrix \eqn{X=(x_1,\ldots,x_n)^T} of size
##'                             \eqn{n\times d}{n * d}, representing \eqn{n} observations
##'                             and \eqn{d} features.
##' @param Umat,Vmat,Smat,Tmat  The matrices \eqn{U=(u_{li})}, \eqn{V=(v_{li})},
##'                             \eqn{S=(s_{hi})}, and \eqn{T=(t_{hi})} in the
##'                             ReHLine optimization problem. \eqn{U} and \eqn{V}
##'                             are of size \eqn{L\times n}{L * n}, and \eqn{S} and
##'                             \eqn{T} are of size \eqn{H\times n}{H * n}.
##'                             Some of these matrices can be set to
##'                             \code{NULL}, meaning excluding the ReLU or ReHU
##'                             terms in the objective function.
##' @param Tau                  Either a numeric scalar, or an \eqn{H\times n}{H * n} matrix
##'                             representing \eqn{\tau=(\tau_{hi})}.
##' @param Amat                 An \eqn{m\times d}{m * d} matrix representing the
##'                             coefficients of \eqn{m} constraints. Can be
##'                             set to \code{NULL}, meaning no constraint is imposed.
##' @param bvec                 A length-\eqn{m} vector. Can be set to \code{NULL},
##'                             meaning no constraint is imposed.
##' @param max_iter             Maximum number of iterations.
##' @param tol                  Tolerance parameter for convergence test.
##' @param shrink               Whether to use the shrinkage algorithm.
##' @param verbose              Level of verbosity.
##' @param trace_freq           Trace objective function values every \code{trace_freq}
##'                             iterations. Only works if \code{verbose > 0}.
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
##' set.seed(123)
##' n = 500
##' d = 10
##' L = 5
##' H = 4
##' m = 3
##'
##' Xmat = matrix(rnorm(n * d), n, d)
##' Umat = matrix(rnorm(L * n), L, n)
##' Vmat = matrix(rnorm(L * n), L, n)
##' Smat = matrix(rnorm(H * n), H, n)
##' Tmat = matrix(rnorm(H * n), H, n)
##' Tau = Inf
##' Amat = matrix(rnorm(m * d), m, d)
##' bvec = rnorm(m)
##'
##' res = rehline(
##'     Xmat, Umat, Vmat, Smat, Tmat, Tau, Amat, bvec,
##'     max_iter = 1000, tol = 1e-3, verbose = 0
##' )
##' print(res$beta)
##'
##' res = rehline(
##'     Xmat, Umat = NULL, Vmat = NULL,
##'     Smat = Smat, Tmat = Tmat, Tau = Tau,
##'     Amat = NULL, bvec = NULL,
##'     max_iter = 1000, tol = 1e-3, verbose = 0
##' )
##' print(res$beta)
rehline = function(
    Xmat, Umat, Vmat, Smat = NULL, Tmat = NULL, Tau = Inf,
    Amat = NULL, bvec = NULL,
    max_iter = 1000, tol = 1e-5, shrink = TRUE, verbose = 0, trace_freq = 100)
{
    n = nrow(Xmat)
    d = ncol(Xmat)

    # If U is NULL, exclude U and V
    # If U is not NULL but V is NULL, set V to zero
    if(is.null(Umat))
    {
        Umat = Vmat = matrix(0, 0, n)
    } else if(is.null(Vmat)) {
        Vmat = matrix(0, nrow(Umat), ncol(Umat))
    }
    # Similar for S and T
    if(is.null(Smat))
    {
        Smat = Tmat = matrix(0, 0, n)
    } else if(is.null(Tmat)) {
        Tmat = matrix(0, nrow(Smat), ncol(Smat))
    }
    # Similar for A and b
    if(is.null(Amat))
    {
        Amat = matrix(0, 0, d)
        bvec = numeric(0)
    } else if(is.null(bvec)) {
        bvec = numeric(nrow(Amat))
    }

    # Expand Tau to a matrix
    if(length(Tau) == 1)
        Tau = matrix(Tau, nrow(Tmat), ncol(Tmat))

    rehline_(
        Xmat, Amat, bvec, Umat, Vmat, Smat, Tmat, Tau,
        max_iter, tol, shrink, verbose, trace_freq
    )
}
