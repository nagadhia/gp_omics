#include <iostream>
#include <armadillo>
#include <cmath>
#include <random>
#include <chrono>

using namespace arma;
using namespace std;

// RBF Kernel
double RBFKernel(const vec &x1, const vec &x2, double theta) {
    return exp(-sum(square(x1 - x2)) / (2 * theta * theta));
}

// Hybrid Kernel
double HybridKernel(const vec &x1, const vec &x2, const vec &theta) {
    uword num_genes = x1.n_elem - 2;
    double k_genes = RBFKernel(x1.subvec(0, num_genes - 1), x2.subvec(0, num_genes - 1), theta(0));
    double k_vaccine = RBFKernel(x1.subvec(num_genes, num_genes), x2.subvec(num_genes, num_genes), theta(1));
    double k_days = RBFKernel(x1.subvec(num_genes + 1, num_genes + 1), x2.subvec(num_genes + 1, num_genes + 1), theta(2));
    return k_genes + k_vaccine + k_days;
}

// Compute Kernel Matrix
mat KernelMatrix(const mat &X1, const mat &X2, const vec &theta) {
    mat K(X1.n_rows, X2.n_rows, fill::zeros);
    for (uword i = 0; i < X1.n_rows; ++i) {
        for (uword j = 0; j < X2.n_rows; ++j) {
            K(i, j) = HybridKernel(X1.row(i).t(), X2.row(j).t(), theta);
        }
    }
    return K;
}


// Marginal Log Likelihood
double MarginalLogLikelihood(const mat &X, const vec &y, const vec &theta) {
    mat K = KernelMatrix(X, X, theta) + 1e-6 * eye(X.n_rows, X.n_rows);
    mat L;
    bool chol_success = chol(L, K, "lower");
    if (!chol_success) {
        return -datum::inf;
    }

    vec alpha = solve(L.t(), solve(L, y));
    double log_likelihood = -0.5 * as_scalar(y.t() * alpha) - sum(log(L.diag())) - 0.5 * X.n_rows * log(2 * datum::pi);
    return log_likelihood;
}

// MCMC Update
vec update_theta(const mat &X, const vec &y, vec theta, const vec &step_size) {
    double current_ll = MarginalLogLikelihood(X, y, theta);
    vec new_theta = theta;

    for (uword row = 0; row < theta.n_rows; ++row) {
        double log_current_theta = log(theta(row));
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, step_size(row));
        double proposed_log_theta = log_current_theta + distribution(generator);

        new_theta(row) = exp(proposed_log_theta);

        double new_ll = MarginalLogLikelihood(X, y, new_theta);

        double acceptance_ratio = exp(new_ll - current_ll + proposed_log_theta - log_current_theta);

        if (randu<double>() < acceptance_ratio) {
            theta = new_theta;
        }
    }
    return theta;
}

// GPC Prediction
vec GPC_Predict(const mat &X_train, const vec &y_train, const mat &X_test, const mat &param_samples) {
    mat predictions(X_test.n_rows, param_samples.n_rows, fill::zeros);
    for (uword iter = 0; iter < param_samples.n_rows; ++iter) {
        const vec& theta = param_samples.row(iter).t();
        mat K = KernelMatrix(X_train, X_train, theta) + 1e-6 * eye(X_train.n_rows, X_train.n_rows);
        mat K_inv = pinv(K);
        mat K_test_train = KernelMatrix(X_test, X_train, theta);

        vec f_pred = K_test_train * K_inv * y_train;
        predictions.col(iter) = 1.0 / (1.0 + exp(-f_pred));
    }
    return mean(predictions, 1);
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    //load data
    mat X;
    X.load("/Users/nandini.gadhia/Documents/projects/gp_omics/data_rvc/OTU_table_umap.csv", csv_ascii);
    //X.load("/Users/nandini.gadhia/Documents/projects/gp_omics/data_rvc/fulldata(after_PCA).csv", csv_ascii); //need to change the path
    
    cout << "Data loaded!" << endl;

    int num_features = X.n_cols - 1;
    mat X_all = X.cols(0, num_features - 1);
    vec y_all = X.col(num_features);
    int total_samples = X_all.n_rows;
    int train_size = 70;

    int n_splits = 40; // number of the trails

    int n_iter = 1000; // number of the interation for MCMC

    vec step_size_theta = ones<vec>(3) * 0.01; // Initialize step sizes (important!)

    for (int i = 0; i < n_splits; i++) {

        std::random_device rd;

        std::mt19937 gen(rd());

        arma::uvec indices = arma::linspace<arma::uvec>(0, total_samples - 1, total_samples);

        indices = arma::shuffle(indices);

        uvec train_idx = indices.head(train_size);

        uvec test_idx = indices.tail(total_samples - train_size);

        mat X_train = X_all.rows(train_idx);

        vec y_train = y_all.rows(train_idx);

        mat X_test = X_all.rows(test_idx);

        vec y_test = y_all.rows(test_idx);

        // Make output directory if it doesn't exist
        string outdir = "/Users/nandini.gadhia/Documents/projects/gp_omics/GP_OMICS/outdir_nan_umap/"; // the path for saving data
        system(("mkdir -p " + outdir).c_str());

        string base_path = outdir + "split_" + to_string(i);

        // save data
        X_train.save(base_path + "_X_train.csv", csv_ascii);
        y_train.save(base_path + "_Y_train.csv", csv_ascii);
        X_test.save(base_path + "_X_test.csv", csv_ascii);
        y_test.save(base_path + "_Y_test.csv", csv_ascii);

        //MCMC

        vec current_theta = ones<vec>(3) * 0.1;
        mat param_samples(n_iter, 3, fill::zeros);
        for (int iter = 0; iter < n_iter; ++iter) {
            current_theta = update_theta(X_train, y_train, current_theta, step_size_theta);
            param_samples.row(iter) = current_theta.t();
        }


        // Save parameter samples
        param_samples.save(base_path + "_param_samples.csv", csv_ascii);

        // Prediction
        vec Pred_Y_mean = GPC_Predict(X_train, y_train, X_test, param_samples);

        // Save predictions
        Pred_Y_mean.save(base_path + "_Pred_Y_mean.csv", csv_ascii);
        cout << Pred_Y_mean << endl;

        vec Pred_Y_mean_01 = conv_to<vec>::from(Pred_Y_mean > 0.5);

        Pred_Y_mean_01.save(base_path + "_Pred_Y_mean_01.csv", csv_ascii); 
    }

    auto end = std::chrono::high_resolution_clock::now();
    cout << "Total execution time: " << chrono::duration<double>(end - start).count() << " s" << endl;

    return 0;
}