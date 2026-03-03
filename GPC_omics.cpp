
// gaussian processes classification with laplace for graphs with OTU normalised data 88 x 740
// version 18/07/2025

#include <iostream>
#include <armadillo>
#include <cmath>
#include <random>
#include <chrono>
#include <memory>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

using namespace arma;
using namespace std;
namespace fs = std::filesystem;



// load  omic data

vector<mat> load_adjacency_matrices(const string& folder_path) {
    
    vector<mat> matrices;
    
    vector<fs::path> files;
    
    // search all the path
    
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        
        if (entry.path().extension() == ".csv") {
            
            files.push_back(entry.path());
        }
    }
    
    // order
    
    sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        
        return a.filename().string() < b.filename().string();
    });
    
    // load
    
    for (const auto& file : files) {
        
        mat matrix;
        
        if (matrix.load(file.string(), csv_ascii)) {
            
            matrices.push_back(matrix);
            
            cout << "Loaded (sorted): " << file.filename() << endl;
        }
    }
    return matrices;
}



vector<vec> load_node_features(const string& folder_path) {
    
    vector<vec> features_list;
    
    vector<pair<string, fs::path>> file_entries;
    
    // read and order
    
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        
        if (entry.path().extension() == ".csv") {
            
            string filename = entry.path().filename().string();
            
            // donot read .DS_Store!!
            
            if (filename[0] != '.') {
                
                file_entries.emplace_back(filename, entry.path());
            }
        }
    }

    // reorder based on the name
    
    sort(file_entries.begin(), file_entries.end(),
         
        [](const auto& a, const auto& b) {
        
            return a.first < b.first;
        });

    // load
    
    for (const auto& [filename, path] : file_entries) {
        
        ifstream file(path.string());
        
        if (!file.is_open()) {
            
            cerr << "Failed to open: " << path << endl;
            
            continue;
        }

        vec features;
        
        string line;
        
        // no first line
        getline(file, line);
        
        // load
        while (getline(file, line)) {
            
            double value;
            
            istringstream iss(line);
            
            if (iss >> value) {
                
                features.resize(features.n_elem + 1);
                
                features(features.n_elem - 1) = value;
            }
        }

        if (features.n_elem > 0) {
            
            features_list.push_back(features);
            
            cout << "Loaded (sorted): " << filename
            
                 << " with " << features.n_elem << " features" << endl;
            
        } else {
            
            cerr << "Warning: Empty feature file: " << path << endl;
        }
    }

    return features_list;
}



// Propagation Kernel

class PropagationKernel {
public:
    int t_MAX;
    double sigma;

    PropagationKernel(int t = 5, double s = 1.0) : t_MAX(t), sigma(s) {}

    vec compute_graph_embedding(const mat& A, const vec& node_features) {
        
        vec F = node_features;
        
        for (int t = 0; t < t_MAX; ++t) {
            
            vec D_inv = 1.0 / (sum(A, 1) + 1e-8);
            
            F = diagmat(D_inv) * A * F;
        }
        return F;
    }

    double graph_similarity(const vec& g1, const vec& g2) const {
        
        double dist_sq = norm(g1 - g2, 2);
        
        return exp(-dist_sq * dist_sq / (2.0 * sigma * sigma));
    }
};

// HybridKernel

double HybridKernel(const vec& x1, const vec& x2, const mat& A1, const mat& F1, const mat& A2, const mat& F2, vec theta, double lambda, int t_MAX = 5) {
    
    double k_genes = exp(-sum(square(x1 - x2)) / theta(0));
    
    PropagationKernel pk(t_MAX, theta(1));
    
    vec g1 = pk.compute_graph_embedding(A1, F1);
    
    vec g2 = pk.compute_graph_embedding(A2, F2);
    
    double k_graph = pk.graph_similarity(g1, g2);
    
    return k_genes + lambda * k_graph;
}

mat KernelMatrix(const mat& X1, const mat& X2, const vector<mat>& As1, const vector<mat>& Fs1, const vector<mat>& As2, const vector<mat>& Fs2, vec theta, double lambda, int t_MAX = 5) {
    
    mat K(X1.n_rows, X2.n_rows, fill::zeros);
    
    for (uword i = 0; i < X1.n_rows; ++i) {
        
        for (uword j = 0; j < X2.n_rows; ++j) {
            
            K(i, j) = HybridKernel(X1.row(i).t(), X2.row(j).t(), As1[i], Fs1[i], As2[j], Fs2[j], theta, lambda, t_MAX);
        }
    }
    return K;
}

//Laplace Approximation

tuple<vec, mat> LaplaceApprox(const mat& X, const vec& y, const vector<mat>& As, const vector<mat>& Fs, vec theta, double lambda) {
    
    uword n = X.n_rows;
    
    mat K = KernelMatrix(X, X, As, Fs, As, Fs, theta, lambda) + 1e-6 * eye(n, n);
    
    vec f = zeros<vec>(n);

    for (int iter = 0; iter < 20; ++iter) {
        
        vec pi = 1.0 / (1.0 + exp(-f));
        
        vec W_diag = pi % (1 - pi);
        
        mat W = diagmat(W_diag);
        
        vec z = f + (y - pi) / W_diag;
        
        mat L = chol(eye(n, n) + sqrt(W) * K * sqrt(W), "lower");
        
        vec b = sqrt(W) * (z - f);
        
        vec a = solve(trimatl(L), b);
        
        vec delta_f = sqrt(W) * solve(trimatu(L.t()), a);
        
        f += delta_f;
        
        if (norm(delta_f) < 1e-6) break;
        
    }

    vec pi = 1.0 / (1.0 + exp(-f));
    
    vec W_diag = pi % (1 - pi);
    
    mat W = diagmat(W_diag);
    
    mat Sigma_inv = inv(K) + W;
    
    mat Sigma = inv(Sigma_inv);
    
    return {f, Sigma};
}

//Likelihood

double MarginalLikelihood(const mat& X, const vec& y, const vector<mat>& As, const vector<mat>& Fs, vec theta, double lambda) {
    
    uword n = X.n_rows;
    
    auto [f_map, Sigma] = LaplaceApprox(X, y, As, Fs, theta, lambda);
    
    mat K = KernelMatrix(X, X, As, Fs, As, Fs, theta, lambda) + 1e-6 * eye(n, n);
    
    vec pi = 1.0 / (1.0 + exp(-f_map));
    
    vec W_diag = pi % (1 - pi);
    
    mat W = diagmat(W_diag);
    
    mat L = chol(eye(n, n) + sqrt(W) * K * sqrt(W), "lower");

    double log_det = 2.0 * sum(log(L.diag()));
    
    double term1 = -0.5 * as_scalar(f_map.t() * inv(K) * f_map);
    
    double term2 = accu(y % log(pi) + (1-y) % log(1 - pi));
    
    double term3 = -0.5 * log_det;
    
    return term1 + term2 + term3;
}

//MCMC

tuple<vec, vec> update_theta(const mat& X, const vec& y, const vector<mat>& As, const vector<mat>& Fs, vec current_theta, double current_lambda, vec step_size) {
    
    uword n = current_theta.n_rows;
    
    vec new_theta = current_theta;
    
    vec last_accept_prob = ones<vec>(n) * 0.0;

    for (int i = 0; i < n; ++i) {
        
        double current_ll = MarginalLikelihood(X, y, As, Fs, current_theta, current_lambda) + log(current_theta(i));
        
        double prop = log(current_theta(i)) + randn() * step_size(i);
        
        new_theta(i) = exp(prop);
        
        if (new_theta(i) < 0.0001 || new_theta(i) > 10.0) {
            
            std::cout << "reject_new_theta(i): out of prior bounds" << std::endl;
            
            new_theta(i) = current_theta(i); // revert change
            
            continue;
        }
        
        double prop_ll = MarginalLikelihood(X, y, As, Fs, new_theta, current_lambda) + log(new_theta(i));

        double log_ratio = prop_ll - current_ll;
        
        double log_accept_prob = std::min(0.0, log_ratio);
        
        last_accept_prob(i) = exp(log_accept_prob);

        if (randu() < last_accept_prob(i)) {
            
            current_theta(i) = new_theta(i);
            
        } else {
            new_theta(i) = current_theta(i);
        }
    }
    return {current_theta, last_accept_prob};
}

tuple<double, double> update_lambda(const mat& X, const vec& y, const vector<mat>& As, const vector<mat>& Fs, vec current_theta, double current_lambda, vec step_size) {
    
    double current_ll = MarginalLikelihood(X, y, As, Fs, current_theta, current_lambda) + log(current_lambda);
    
    double prop = log(current_lambda) + randn() * step_size(2);
    
    double lambda_prop = exp(prop);
    
    if (lambda_prop < 0.0001 || lambda_prop > 10.0) {
        
        std::cout << "reject lambda_prop: out of prior bounds" << std::endl;
        
        return std::make_tuple(current_lambda, 0.0); // reject with prob = 0
        
    }
    
    double prop_ll = MarginalLikelihood(X, y, As, Fs, current_theta, lambda_prop) + log(lambda_prop);

    double log_ratio = prop_ll - current_ll;
    
    double log_accept_prob = std::min(0.0, log_ratio);
    
    double accept_prob = exp(log_accept_prob);

    if (randu() < accept_prob) {
        
        current_lambda = lambda_prop;
    }
    return {current_lambda, accept_prob};
}

// Prediction

vec GPC_Predict(const mat &X_train, const vec &y_train, const mat &X_test, const vector<mat> &As_train, const vector<mat> &Fs_train, const vector<mat> &As_test, const vector<mat> &Fs_test, const mat &param_samples, int n_iter) {

    uword n_train = X_train.n_rows;
    
    uword n_test = X_test.n_rows;
    
    mat predictions_y = mat(X_test.n_rows, n_iter, fill::zeros);
    
    for (int iter = 0; iter < n_iter; ++iter) {
        
        std::cout << "prediction: " << iter << std::endl;
        
        vec theta = param_samples.row(iter).subvec(0, 1).t();
        
        double lambda = param_samples(iter, 2);
        
        auto [f_map, Sigma] = LaplaceApprox(X_train, y_train, As_train, Fs_train, theta, lambda);
        
        mat K = KernelMatrix(X_train, X_train, As_train, Fs_train, As_train, Fs_train, theta, lambda) + 1e-6 * eye(n_train, n_train);
        
        mat K_s = KernelMatrix(X_train, X_test, As_train, Fs_train, As_test, Fs_test, theta, lambda);
        
        mat K_ss = KernelMatrix(X_test, X_test, As_test, Fs_test, As_test, Fs_test, theta, lambda) + 1e-6 * eye(n_test, n_test);
        
        vec mu = K_s.t() * solve(K, f_map);
        
        mat v = solve(K, K_s);
        
        vec var = K_ss.diag() - sum(v % K_s, 0).t();
        
        vec pred_prob = normcdf(mu / sqrt(1 + var));
        
        predictions_y.col(iter) = pred_prob;
        
    }
    
    vec Pred_Y = arma::mean(predictions_y, 1);
    
    return Pred_Y;
}



int main() {
    
    auto start = chrono::high_resolution_clock::now();
    
    // Load data

    mat X;
    
    // change the path to load data/ ///////////////////////////////////////////////////////////////////
    X.load("/Users/zhangyue/Library/Mobile Documents/com~apple~CloudDocs/ZhangY/GP_omics/OTU_table_normalised_alldata.csv", csv_ascii);
    // /////////////////////////////////////////////////////////////////////////////////////////////////

    mat X_all = X.cols(1, 742);
    
    vec y_all = X.col(743);
    
    X_all = X_all.rows(1, 88);
    
    y_all = y_all.rows(1, 88);
    
    uword total_samples = X_all.n_rows;
    
    // change the path to load data/ ///////////////////////////////////////////////////////////////////
    string adj_folder = "/Users/zhangyue/Library/Mobile Documents/com~apple~CloudDocs/ZhangY/GP_omics/adjacency_matrices_zeropadding";
    
    string feature_folder = "/Users/zhangyue/Library/Mobile Documents/com~apple~CloudDocs/ZhangY/GP_omics/adjacency_matrices_zeropadding/node_features";
    // ////////////////////////////////////////////////////////////////////////////////////////////////

    vector<mat> adjacency_matrices = load_adjacency_matrices(adj_folder);
    
    vector<vec> node_features = load_node_features(feature_folder);

    if (adjacency_matrices.size() != node_features.size()) {
        
        cerr << "Error: Number of adjacency matrices (" << adjacency_matrices.size()
        
             << ") != number of feature files (" << node_features.size() << ")" << endl;
        exit(1);
    }

    // Build As and Fs (Fs is reshaped from vec to mat)
    
    vector<mat> As = adjacency_matrices;
    
    vector<mat> Fs;

    for (size_t i = 0; i < adjacency_matrices.size(); ++i) {
        
        vec feat = node_features[i];
        
        if (adjacency_matrices[i].n_rows != feat.n_elem) {
            
            cerr << "Error: Graph " << i << " has " << adjacency_matrices[i].n_rows
            
                 << " nodes but " << feat.n_elem << " features" << endl;
            
            exit(1);
        }
        Fs.push_back(reshape(feat, feat.n_elem, 1));
    }
    
    
    
    // Split data

    //int train_size = 70;
    //int test_size = 30;
    
    int n_splits = 20;
    
    vec test_accuracies(n_splits, fill::zeros);

    for (int i = 0; i < n_splits; i++) {
        
        // label = 0 indices: 10–19, 50–59
        
        uvec label0_pool = join_vert(regspace<uvec>(10, 19), regspace<uvec>(50, 59));
        
    
        label0_pool = shuffle(label0_pool);
        
        uvec label0_train = label0_pool.head(15);
        
        uvec label0_test = label0_pool.tail(label0_pool.n_elem - 15);

        // label = 1 indices: all others
        
        uvec all_indices = regspace<uvec>(0, 87);
        
        uvec label1_pool = all_indices;
        

        label1_pool.shed_rows(regspace<uvec>(10, 19));
        label1_pool.shed_rows(regspace<uvec>(40, 49));
        
        label1_pool = shuffle(label1_pool);
        
        uvec label1_train = label1_pool.head(15);
        
        uvec label1_test = label1_pool.tail(15);

       
        uvec train_idx = join_vert(label0_train, label1_train);
        uvec test_idx = join_vert(label0_test, label1_test);

        mat X_train = X_all.rows(train_idx);
        vec y_train = y_all.rows(train_idx);
        mat X_test = X_all.rows(test_idx);
        vec y_test = y_all.rows(test_idx);

        vector<mat> As_train, As_test;
        vector<mat> Fs_train, Fs_test;

        for (auto idx : train_idx) {
            As_train.push_back(As[idx]);
            Fs_train.push_back(Fs[idx]);
        }
        for (auto idx : test_idx) {
            As_test.push_back(As[idx]);
            Fs_test.push_back(Fs[idx]);
        }

        // change the path to save results  /////////////////////////////////////////////////////
        string base_path = "/Users/zhangyue/Library/Mobile Documents/com~apple~CloudDocs/ZhangY/GP_omics/results/split_" + to_string(i);
        // ///////////////////////////////////////////////////////////////////////////////////
        
        X_train.save(base_path + "_X_train_graph.csv", csv_ascii);
        y_train.save(base_path + "_Y_train_graph.csv", csv_ascii);
        X_test.save(base_path + "_X_test_graph.csv", csv_ascii);
        y_test.save(base_path + "_Y_test_graph.csv", csv_ascii);
        
        // Initial for MCMC
        
        int n_iter = 10000;
        
        double target_acceptance_rate = 0.234;
        
        vec step_size = ones<vec>(3) * 0.1;

        vec current_theta = ones<vec>(2) * 0.1;
        
        double current_lambda = 1.0;

        mat param_samples(n_iter, 3, fill::zeros);

        for (int iter = 0; iter < n_iter; ++iter) {
            
            std::cout << "MCMC Iteration: " << iter << std::endl;

            auto theta_result = update_theta(X_train, y_train, As_train, Fs_train, current_theta, current_lambda, step_size);

            current_theta = get<0>(theta_result);

            param_samples(iter, 0) = current_theta(0);
            
            param_samples(iter, 1) = current_theta(1);
            
            step_size(0) = arma::as_scalar(exp(log(step_size(0)) + pow(1.0 / (iter + 1), 0.7) * (std::get<1>(theta_result)(0) - target_acceptance_rate)));
            
            step_size(1) = arma::as_scalar(exp(log(step_size(1)) + pow(1.0 / (iter + 1), 0.7) * (std::get<1>(theta_result)(1) - target_acceptance_rate)));

            
            auto lambda_result = update_lambda(X_train, y_train, As_train, Fs_train, current_theta, current_lambda, step_size);

            current_lambda = get<0>(lambda_result);
            
            param_samples(iter, 2) = current_lambda;
            
            step_size(2) = exp(log(step_size(2)) + pow(1.0 / (iter + 1), 0.7) * (std::get<1>(lambda_result) - target_acceptance_rate));
            
            std::cout << "step_size: " << step_size << std::endl;
            
            
        }
        
        param_samples.save(base_path + "_param_samples_graph.csv", csv_ascii);


        vec test_prob = GPC_Predict(X_train, y_train, X_test, As_train, Fs_train, As_test, Fs_test, param_samples, n_iter);

        vec test_label = conv_to<vec>::from(test_prob > 0.5);
        
        test_label.save(base_path + "_Pred_Y_mean_graph_RBF_100.csv", csv_ascii);
        
        
        
        double test_acc = accu(test_label == y_test) / double(y_test.n_elem);
        
        cout << "Test Accuracy: " << test_acc * 100 << "%" << endl;

        cout << "Finished split " << i + 1 << "/" << n_splits << endl;
    }
    
    test_accuracies.save("/Users/zhangyue/Library/Mobile Documents/com~apple~CloudDocs/ZhangY/GP_omics/results/Test_Accuracy_all_splits.csv", csv_ascii);

    

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Total execution time: " << duration.count() << " s" << endl;

    return 0;
}
