
#include <iostream>
#include <armadillo>
#include <random>
#include <limits>  


arma::mat simulate_gbm_paths(double S0, double mu, double sigma, double T, int N, int num_paths) {
    double dt = T / N;
    arma::vec t = arma::linspace(0, T, N + 1);

    arma::mat paths(num_paths, N + 1);  // Paths as rows
    paths.col(0).fill(S0);       // Initialize first column with S0

    // Random number generation (using C++11 random engine)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    arma::mat Z = arma::mat(num_paths, N).randn(); // Generate matrix of standard normal random numbers

    // Monte Carlo simulation
    for (int i = 0; i < num_paths; i++) {
        for (int j = 0; j < N; j++) {
            double dW = sqrt(dt) * Z(i, j);
            paths(i, j + 1) = paths(i, j) * exp((mu - 0.5 * sigma * sigma) * dt + sigma * dW);
        }
    }

    return paths;
}



arma::mat put_payoff(const arma::mat& paths, double K) {
    arma::mat payoffs(paths.n_rows, paths.n_cols);

    for (arma::uword col = 0; col < paths.n_cols; ++col) {
        payoffs.col(col) = max(K - paths.col(col), arma::vec(paths.n_rows).fill(0.0)); // Element-wise max
    }

    return payoffs;
}


arma::mat create_basis(const arma::vec& x) {
    int n = x.n_elem; // Number of elements in x
    arma::mat B(n, 3);     // Basis matrix with n rows and 3 columns

    // Fill the basis matrix
    B.col(0).fill(1);            // First column with ones
    B.col(1) = x;                // Second column with x values
    B.col(2) = x % x;           // Third column with x^2 values (element-wise squaring)

    return B;
}

arma::mat RegCoeff(const arma::mat& S, const arma::mat& payoffs, const arma::vec& df, std::function<arma::mat(const arma::vec&)> basis, double Nb, double Nr) {

    arma::mat f = arma::zeros<arma::mat>(Nb, Nr - 1); // Initialize coefficients matrix

    arma::vec v = payoffs.col(Nr - 1);         // Start with the last payoff column

    arma::mat ones_v = arma::ones<arma::mat>(v.n_elem, 1);

    for (int i = Nr - 2; i >= 0; --i) {  // Backward induction (from Nr-2 to 0)

        arma::uvec index = arma::find(payoffs.col(i) > 0);  // Indices of in-the-money paths

		

        arma::vec s(index.n_elem);  // Initialize with correct size


        for (arma::uword j = 0; j < index.n_elem; ++j) {
            s(j) = S(index(j), i + 1); // Fill 's' with the corresponding S values for that time slice
        }
      

        v = v % (ones_v*df(i + 1));     // Discounted option values at t_i


       arma::mat A = basis(s);       // Subset of basis matrix for ITM paths

        // Regression coefficients using Armadillo's linear solver, using the 'solve' function directly to avoid matrix inversion
        f.col(i) = solve(A, v(index), arma::solve_opts::fast);


        arma::vec c = A * f.col(i);        // Continuation values

        for (arma::uword j = 0; j < index.n_elem; ++j) {
            if (payoffs(index(j), i) >= c(j)) {
                v(index(j)) = payoffs(index(j), i); // Update if payoff > continuation value
            }
        }

       
    }

    return f;
}


double LongstaffSchwartzPricer(const arma::mat& S, const arma::mat& payoffs, const arma::vec& df, std::function<arma::mat(const arma::vec&)> basis, const arma::mat& RG, double Nr, double NSim, double alpha) {
    arma::mat f = arma::zeros<arma::mat>(RG.n_rows, RG.n_cols); // Initialize coefficients matrix

    arma::vec v = payoffs.col(Nr - 1);         // Start with the last payoff column

    arma::mat ones_v = arma::ones<arma::mat>(v.n_elem, 1);


    for (int i = Nr - 2; i >= 0; --i) {  // Backward induction (from Nr-2 to 0)

        arma::uvec index = arma::find(payoffs.col(i) > 0);  // Indices of in-the-money paths

        arma::vec s(index.n_elem);  // Initialize with correct size

        for (arma::uword j = 0; j < index.n_elem; ++j) {
            s(j) = S(index(j), i + 1); // Fill 's' with the corresponding S values for that time slice
        }

     
        v = v % (ones_v * df(i + 1));     // Discounted option values at t_i

        arma::mat A = basis(s);       // Subset of basis matrix for ITM paths


        arma::vec c = A * RG.col(i);        // Continuation values

        
        for (arma::uword j = 0; j < index.n_elem; ++j) {
            if (payoffs(index(j), i) >= c(j)) {
                v(index(j)) = payoffs(index(j), i); // Update if payoff > continuation value
            }
        }

    }

	return arma::mean(v) * df(0);
}

int main() {
	double Nr = 13;
	double NSim = 10000;
	double S0 = 100;
	double K = 100;

	double T = 0.25;
	double dt = T / Nr;
	double r = 0.06;
	double d = 0.0;
	double sigma = 0.15;
	double type = -1;


    arma::vec discount_factor = exp(-r * dt) * arma::ones<arma::vec>(Nr, 1);


	arma::mat path1 = simulate_gbm_paths(S0, r - d, sigma, T, Nr, NSim);
	arma::mat path2 = simulate_gbm_paths(S0, r - d, sigma, T, Nr, NSim/10);

    // Display some results (e.g., the first 5 paths)
	path1.rows(0, 4).print("First 5 paths:");

	arma::mat payoff1 = put_payoff(path1, K);
	arma::mat payoff2 = put_payoff(path2, K);


	arma::mat RG = RegCoeff(path2, payoff2, discount_factor, create_basis, 3, Nr);

    double price = LongstaffSchwartzPricer(path1, payoff1, discount_factor, create_basis, RG, Nr, NSim, 0.99);

    std::cout << price << std::endl;


}