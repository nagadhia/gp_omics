//
//  main.cpp
//  GP_omics_parallel
//
//  Created by on 15/01/2025.
//



#include <iostream>
#include <armadillo>
#include <cmath>
#include <random>
#include <chrono>

using namespace arma;
using namespace std;

// Exp2 kernel

double RBFKernel(const vec &x1, const vec &x2, double theta) {
    
    double sqdist = sum(square(x1 - x2));
    
    return exp(-sqdist / theta);
    
}


// Hamming Kernel

double HammingKernel(double v1, double v2, double weight) {
    
    return (v1 == v2) ? weight : 0.0;
    
}

// Kernel

double HybridKernel(const vec &x1, const vec &x2, vec theta) {
    
    //kernel for gene
    
    double k_genes = RBFKernel(x1.subvec(0, 70), x2.subvec(0, 70), theta(0));

    //kernel for vaccine
    
    double k_vaccine = RBFKernel(x1.subvec(71, 71), x2.subvec(71, 71), theta(1));

    //kernel for DPI
    
    double k_days = RBFKernel(x1.subvec(72, 72), x2.subvec(72, 72), theta(2));

    //
    return (k_genes + k_vaccine) * k_days;
}

// kernel matrix

mat KernelMatrix(const mat &X1, const mat &X2, vec theta) {
    
    uword n1 = X1.n_rows;
    
    uword n2 = X2.n_rows;
    
    mat K(n1, n2, fill::zeros);
    
    for (int i = 0; i < n1; ++i) {
        
        for (int j = 0; j < n2; ++j) {
            
            K(i, j) = HybridKernel(X1.row(i).t(), X2.row(j).t(), theta);
        }
    }
    return K;
}

//logistic likelihod

double LogisticLikelihood(const vec &f, const vec &y) {
    
    return accu(y % log(1.0 / (1.0 + exp(-f))) + (1 - y) % log(1 - 1.0 / (1.0 + exp(-f))));
    
}

double MarginalLogLikelihood(const mat &X, const vec &y, vec theta) {
    
    mat K = KernelMatrix(X, X, theta) + 1e-6 * eye(X.n_rows, X.n_rows);
    
    mat L = chol(K, "lower");
    
    vec f = mvnrnd(zeros<vec>(X.n_rows), K, 1);
    
    double log_likelihood = LogisticLikelihood(f, y);
    
    log_likelihood -= sum(log(L.diag()));
    
    return log_likelihood;
}

double uniformPrior(double x) {
    
    double a = 0.0;
    
    double b = 100;

    if (x >= a && x <= b) {
        
        return 1.0 / (b - a);
        
    } else {
        
        return 0.0;
    }
}


// MCMC

std::tuple<vec, vec> update_theta(const mat &X, const vec &y, vec current_theta, vec step_size) {
    
    arma::uword n_rows= current_theta.n_rows;
    
    vec last_accept_prob = ones<vec>(3) * 0.0;
    
    for (arma::uword row = 0; row < n_rows; ++row){
        
        double current_ll_row = MarginalLogLikelihood(X, y, current_theta) + log(current_theta(row)) + log(uniformPrior(current_theta(row)));;
        
        double current_m = log(current_theta(row));
        
        vec new_theta = current_theta;
        
        double new_m = current_m + randn() * step_size(row);
        
        new_theta(row) = exp(new_m);
        
        //std::cout << "new_theta: " << new_theta <<  std::endl;

        double proposal_ll_row = MarginalLogLikelihood(X, y, new_theta) + log(new_theta(row)) + log(uniformPrior(new_theta(row)));

        double log_ratio = proposal_ll_row - current_ll_row;
        
        double log_accept_prob = std::min(0.0, log_ratio);
        
        last_accept_prob(row) = exp(log_accept_prob);
        
        double aa = randu<double>();
        
        if (aa < last_accept_prob(row)) {
            
            //current_ll_row = proposal_ll_row;
            
            current_theta = new_theta;
            
            //std::cout << "new_theta: " << new_theta <<  std::endl;
            
        } else {
                //std::cout << "reject" << std::endl;
            }
        
            
    }

    return std::make_tuple(current_theta, last_accept_prob);
}



// GPC prediction

std::tuple<arma::mat, std::vector<arma::mat>> GPC_Predict(const mat &X_train, const vec &y_train, const mat &X_test, const mat &param_samples, int &n_iter) {
    
    mat predictions_y = mat(X_test.n_rows, n_iter, fill::zeros);
    
    std::vector<arma::mat> pred_y_var(n_iter, arma::zeros<arma::mat>(predictions_y.n_rows, predictions_y.n_rows));
    
    for (int iter = 0; iter < n_iter; ++iter) {
        
        std::cout << "prediction: " << iter << std::endl;
        
        mat K = KernelMatrix(X_train, X_train, param_samples.row(iter).t()) + 1e-6 * eye(X_train.n_rows, X_train.n_rows);
        
        mat K_inv = inv(K);
        
        mat K_test_train = KernelMatrix(X_test, X_train, param_samples.row(iter).t());
        
        mat K_test_test = KernelMatrix(X_test, X_test, param_samples.row(iter).t()) + 1e-6 * eye(X_test.n_rows, X_test.n_rows);
        
        vec f_pred = K_test_train * K_inv * y_train;
        
        vec y_pred =  1.0 / (1.0 + exp(-f_pred));
        
        predictions_y.col(iter) = y_pred;
        
        mat pred_var = K_test_test - K_test_train * K_inv * K_test_train.t();
        
        pred_y_var[iter] = pred_var;
        
    }
    
    return std::make_tuple(predictions_y , pred_y_var);
    
    
}






int main() {
    auto start = std::chrono::high_resolution_clock::now();

    //data
    
    mat X;
    
    X.load("/Users/zhangyue/Desktop/ZhangY/GP_omics/data.csv", csv_ascii); //need to change the path

    mat X_all = X.cols(0, 72);
    
    vec y_all = X.col(73);

    int total_samples = X_all.n_rows;
    
    int train_size = 70;
    
    int n_splits = 100; // number of the trails
    
    int n_iter = 10000; // number of the interation for MCMC

    vec step_size_theta = ones<vec>(3) * 0.1;
    
    double target_acceptance_rate = 0.234;

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

        // save data
        string base_path = "/Users/zhangyue/Desktop/ZhangY/GP_omics/split_" + to_string(i); // the path for saving data
        X_train.save(base_path + "_X_train.csv", csv_ascii);
        y_train.save(base_path + "_Y_train.csv", csv_ascii);
        X_test.save(base_path + "_X_test.csv", csv_ascii);
        y_test.save(base_path + "_Y_test.csv", csv_ascii);

        //MCMC
        
        vec current_theta = ones<vec>(3) * 0.1;
        
        mat param_samples(n_iter, 3, fill::zeros);

        for (int iter = 0; iter < n_iter; ++iter) {
            
            std::cout << "MCMC Iteration: " << iter << std::endl;
            
            auto theta_result = update_theta(X_train, y_train, current_theta, step_size_theta);
            
            current_theta = std::get<0>(theta_result);

            param_samples(iter, 0) = current_theta(0);
            param_samples(iter, 1) = current_theta(1);
            param_samples(iter, 2) = current_theta(2);
        }

        param_samples.save(base_path + "_param_samples.csv", csv_ascii);

        // prediction
        
        auto Pred_Y = GPC_Predict(X_train, y_train, X_test, param_samples, n_iter);
        
        arma::mat Pred_Y_mean = std::get<0>(Pred_Y);
        
        std::vector<arma::mat> Pred_Y_var = std::get<1>(Pred_Y);

        arma::vec Pred_Y_mean_mean = arma::mean(Pred_Y_mean, 1);
        
        vec Pred_Y_mean_01 = conv_to<vec>::from(Pred_Y_mean_mean > 0.5);

        mat Pred_Y_var_mean = arma::zeros<arma::mat>(y_test.n_rows, y_test.n_rows);
        
        for (const auto &matrix : Pred_Y_var) {
            
            if (matrix.n_rows == Pred_Y_var_mean.n_rows && matrix.n_cols == Pred_Y_var_mean.n_cols) {
                
                Pred_Y_var_mean += matrix;
            }
        }

        mat result = zeros<mat>(y_test.n_rows, y_test.n_rows);
        
        for (int iter = 0; iter < n_iter; ++iter) {
            
            vec diff = Pred_Y_mean.col(iter) - Pred_Y_mean_mean;
            
            result += diff * diff.t();
        }

        mat Pred_Y_var_mean_mat = Pred_Y_var_mean / Pred_Y_var.size() + 1 / (n_iter - y_test.n_rows) * result;

        // save
        Pred_Y_mean_mean.save(base_path + "_Pred_Y_mean_mean.csv", csv_ascii);
        Pred_Y_mean_01.save(base_path + "_Pred_Y_mean.csv", csv_ascii);
        Pred_Y_var_mean_mat.save(base_path + "_Pred_Y_var.csv", csv_ascii);

        //
        cout << "Finished dataset " << i + 1 << " / " << n_splits << endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total execution time: " << duration.count() << " s" << std::endl;

    return 0;
}
