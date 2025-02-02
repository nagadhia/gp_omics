#include <iostream>
#include <armadillo>
#include <cmath>
#include <random>
#include <chrono>

using namespace arma;
using namespace std;

// RBF Kernel
double RBFKernel(const vec &x1, const vec &x2, double theta) {
    return exp(-sum(square(x1 - x2)) / theta);
}

// Hybrid Kernel
double HybridKernel(const vec &x1, const vec &x2, vec theta) {
    uword num_genes = x1.n_elem - 2; // Dynamically determine gene feature count
    double k_genes = RBFKernel(x1.subvec(0, num_genes - 1), x2.subvec(0, num_genes - 1), theta(0));
    double k_vaccine = RBFKernel(x1.subvec(num_genes, num_genes), x2.subvec(num_genes, num_genes), theta(1));
    double k_days = RBFKernel(x1.subvec(num_genes + 1, num_genes + 1), x2.subvec(num_genes + 1, num_genes + 1), theta(2));
    return k_genes + k_vaccine;
}

// Compute Kernel Matrix
mat KernelMatrix(const mat &X1, const mat &X2, vec theta) {
    mat K(X1.n_rows, X2.n_rows, fill::zeros);
    for (uword i = 0; i < X1.n_rows; ++i) {
        for (uword j = 0; j < X2.n_rows; ++j) {
            K(i, j) = HybridKernel(X1.row(i).t(), X2.row(j).t(), theta);
        }
    }
    return K;
}

// Logistic Likelihood
double LogisticLikelihood(const vec &f, const vec &y) {
    return accu(y % log(1.0 / (1.0 + exp(-f))) + (1 - y) % log(1 - 1.0 / (1.0 + exp(-f))));
}

// Marginal Log Likelihood
double MarginalLogLikelihood(const mat &X, const vec &y, vec theta) {
    mat K = KernelMatrix(X, X, theta) + 1e-6 * eye(X.n_rows, X.n_rows);
    mat L;
    bool chol_success = chol(L, K, "lower");
    if (!chol_success) {
        return -datum::inf; // Return negative infinity if Cholesky fails
    }
    vec alpha = solve(L.t(), solve(L, y)); 
    double log_likelihood = -0.5 * as_scalar(y.t() * alpha) - sum(log(L.diag())) - 0.5 * X.n_rows * log(2 * datum::pi);
    return log_likelihood;
}

// MCMC Update
vec update_theta(const mat &X, const vec &y, vec theta, vec step_size) {
    for (uword row = 0; row < theta.n_rows; ++row) {
        double current_ll = MarginalLogLikelihood(X, y, theta);
        double log_current_theta = log(theta(row)); // Work in log-space
        double proposed_log_theta = log_current_theta + randn() * step_size(row);
        vec new_theta = theta;
        new_theta(row) = exp(proposed_log_theta); // Transform back
        double new_ll = MarginalLogLikelihood(X, y, new_theta);

        double acceptance_ratio = exp(new_ll - current_ll + proposed_log_theta - log_current_theta); // Correct MH ratio
        if (randu<double>() < acceptance_ratio) {
            theta = new_theta;
        }
    }
    return theta;
}



// GPC Prediction
mat GPC_Predict(const mat &X_train, const vec &y_train, const mat &X_test, const mat &param_samples) {
    mat predictions(X_test.n_rows, param_samples.n_rows, fill::zeros);
    for (uword iter = 0; iter < param_samples.n_rows; ++iter) {
        mat K = KernelMatrix(X_train, X_train, param_samples.row(iter).t()) + 1e-6 * eye(X_train.n_rows, X_train.n_rows);
        mat K_inv = pinv(K); // Use pseudoinverse for stability
        mat K_test_train = KernelMatrix(X_test, X_train, param_samples.row(iter).t());
        vec f_pred = K_test_train * K_inv * y_train;
        predictions.col(iter) = 1.0 / (1.0 + exp(-f_pred));
    }
    return mean(predictions, 1);
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    mat X;
    //X.load("/Users/nandini.gadhia/Documents/projects/gp_omics/data_rvc/fulldata(after_PCA).csv", csv_ascii);
    X.load("/Users/nandini.gadhia/Documents/projects/gp_omics/data_rvc/OTU_table_umap.csv", csv_ascii);
    cout << "Data loaded!" << endl;

    int num_features = X.n_cols - 1;
    mat X_all = X.cols(0, num_features - 1);
    vec y_all = X.col(num_features);
    int total_samples = X_all.n_rows;
    int train_size = 70;
    
    for (int i = 0; i < 60; i++) {
        cout << "Processing split " << i + 1 << " of 60..." << endl;
        uvec indices = shuffle(linspace<uvec>(0, total_samples - 1, total_samples));
        uvec train_idx = indices.head(train_size);
        uvec test_idx = indices.tail(total_samples - train_size);
        mat X_train = X_all.rows(train_idx);
        vec y_train = y_all.rows(train_idx);
        mat X_test = X_all.rows(test_idx);
        vec y_test = y_all.rows(test_idx);

        vec current_theta = ones<vec>(3) * 0.1;
        mat param_samples(10000, 3, fill::zeros);
        for (int iter = 0; iter < 10000; ++iter) {
            current_theta = update_theta(X_train, y_train, current_theta, ones<vec>(3) * 0.1);
            param_samples.row(iter) = current_theta.t();
        }

        string outdir = "/Users/nandini.gadhia/Documents/projects/gp_omics/GP_OMICS/outdir_nan_umap";
        system(("mkdir -p " + outdir).c_str());
        param_samples.save(outdir + "/split_" + to_string(i) + "_param_samples.csv", csv_ascii);

        
        mat Pred_Y = GPC_Predict(X_train, y_train, X_test, param_samples);
        Pred_Y.save("/Users/nandini.gadhia/Documents/projects/gp_omics/GP_OMICS/outdir_nan_umap/split_" + to_string(i) + "_Pred_Y_mean_mean.csv", csv_ascii);
    }
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Total execution time: " << chrono::duration<double>(end - start).count() << " s" << endl;
    return 0;
}
