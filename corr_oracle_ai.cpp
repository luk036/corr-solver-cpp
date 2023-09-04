#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
using namespace std;
using Arr = Eigen::MatrixXd;
using Cut = tuple<Arr, double>;

Arr create_2d_sites(int nx = 10, int ny = 8) {
    int n = nx * ny;
    Arr s_end(1, 2);
    s_end << 10.0, 8.0;
    Eigen::MatrixXd hgen(n, 2);
    hgen << 2, 3;
    Arr site = s_end.array()
               * (hgen.unaryExpr([](double x) { return Halton(x); }).matrix().transpose().array());
    return site;
}

Arr create_2d_isotropic(Arr site, int N = 3000) {
    int n = site.rows();
    double sdkern = 0.12;
    double var = 2.0;
    double tau = 0.00001;
    Eigen::MatrixXd Sig(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            Eigen::VectorXd d = site.row(j) - site.row(i);
            Sig(i, j) = exp(-sdkern * (d.dot(d)));
            Sig(j, i) = Sig(i, j);
        }
    }
    Eigen::MatrixXd A = Sig.llt().matrixL();
    Eigen::MatrixXd Y(n, n);
    for (int i = 0; i < N; i++) {
        Eigen::VectorXd x = var * Eigen::VectorXd::Random(n);
        Eigen::VectorXd y = A * x + tau * Eigen::VectorXd::Random(n);
        Y += y * y.transpose();
    }
    Y /= N;
    return Y;
}

vector<Arr> construct_poly_matrix(Arr site, int m) {
    int n = site.rows();
    Arr D1 = construct_distance_matrix(site);
    Arr D = Arr::Ones(n, n);
    vector<Arr> Sig = {D};
    for (int i = 1; i < m; i++) {
        D = D.array() * D1.array();
        Sig.push_back(D);
    }
    return Sig;
}

tuple<Eigen::VectorXd, int, bool> corr_poly(
    Arr Y, Arr site, int m, function<Arr(vector<Arr>, Arr)> oracle,
    function<tuple<Eigen::VectorXd, int, bool>(Arr, int, Arr)> corr_core) {
    vector<Arr> Sig = construct_poly_matrix(site, m);
    Arr omega = oracle(Sig, Y);
    tuple<Eigen::VectorXd, int, bool> res = corr_core(Y, m, omega);
    Eigen::VectorXd a = get<0>(res);
    int num_iters = get<1>(res);
    bool feasible = get<2>(res);
    Eigen::VectorXd pa = a.reverse().eval();
    Eigen::VectorXd b = a.array().square();
    return make_tuple(pa, num_iters, feasible);
}
