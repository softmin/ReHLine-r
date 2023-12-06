#include <RcppEigen.h>
#include "rehline.h"

using Rcpp::List;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

using Matrix = Eigen::MatrixXd;
using MapMat = Eigen::Map<Matrix>;
using Vector = Eigen::VectorXd;
using MapVec = Eigen::Map<Vector>;

// [[Rcpp::export(rehline_)]]
List rehline_impl(
    NumericMatrix Xmat, NumericMatrix Amat, NumericVector bvec,
    NumericMatrix Umat, NumericMatrix Vmat,
    NumericMatrix Smat, NumericMatrix Tmat, NumericMatrix TauMat,
    int max_iter, double tol, bool shrink = true, int verbose = 0, int trace_freq = 100
)
{
    MapMat X = Rcpp::as<MapMat>(Xmat);
    MapMat A = Rcpp::as<MapMat>(Amat);
    MapVec b = Rcpp::as<MapVec>(bvec);
    MapMat U = Rcpp::as<MapMat>(Umat);
    MapMat V = Rcpp::as<MapMat>(Vmat);
    MapMat S = Rcpp::as<MapMat>(Smat);
    MapMat T = Rcpp::as<MapMat>(Tmat);
    MapMat Tau = Rcpp::as<MapMat>(TauMat);

    rehline::ReHLineResult<Matrix> result;
    rehline::rehline_solver(result, X, A, b, U, V, S, T, Tau,
                            max_iter, tol, static_cast<int>(shrink),
                            verbose, trace_freq, Rcpp::Rcout);

    return List::create(
        Rcpp::Named("beta")          = result.beta,
        Rcpp::Named("xi")            = result.xi,
        Rcpp::Named("Lambda")        = result.Lambda,
        Rcpp::Named("Gamma")         = result.Gamma,
        Rcpp::Named("niter")         = result.niter,
        Rcpp::Named("dual_objfns")   = result.dual_objfns,
        Rcpp::Named("primal_objfns") = result.primal_objfns
    );
}
