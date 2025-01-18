/*
 * Wilson loop in the QWZ chern insulator on the square lattice
 * Author: Nabil Atlam
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>


using Eigen::Matrix2cd;
using Eigen::Vector2d;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::SelfAdjointEigenSolver; // To diagonalize the 2 x 2 Bloch Hamiltonian
using Eigen::ComplexEigenSolver;     // To diagonalize the Wilson loop operator
using namespace std;

#define COMPLEX0 complex<double>{0.0, 0.0}
#define COMPLEX1 complex<double>{1.0, 0.0}
#define COMPLEXI complex<double>{0.0, 1.0}

#define NUM_BANDS 2
#define PROJ_SUBSPACE 1

typedef Matrix<complex<double>, NUM_BANDS, PROJ_SUBSPACE> Utype;

/*
 * Definition of the Hamiltonian, will use Eigen library to do matrix calculations
 */
inline Matrix2cd Hamiltonian(const Vector2d& k, const double M) {
    const double kx{k(0)}; const double ky{k(1)};
    const complex Mz{COMPLEX1 * (M + cos(kx) + cos(ky))};
    return Matrix2cd{{Mz, COMPLEX1 * sin(kx) - COMPLEXI * sin(ky)}, {COMPLEX1 * sin(kx) + COMPLEXI * sin(ky), -Mz}};
}


// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main(int argc, char* argv[]) {
    std::cout << "Wilson loop spectrum in the Qi-Wu-Zhang model. \n";

    if (argc != 2) {
        throw std::invalid_argument("Wrong number of arguments. Please provide one argument; the M parameter in the QWZ Hamiltonian");
    }

    /*
     * WILSON LOOP CALCULATION. This is a simple exercise since it is a two-band system
     */

    // Model Parameters
    double M{std::stod(argv[1])};
    //double M = -1.0;
    std::cout << "Mass parameter: "  << M << std::endl;

    // Kpoint grid Box [0, 2pi]^2
    constexpr int Nk{100};
    Matrix<Eigen::Vector<double, 2>, Dynamic, Dynamic> kpoints(Nk, Nk);
    for (int i = 0; i < Nk; ++i) {
        for (int j = 0; j < Nk; ++j) {
            kpoints(i, j) = {2.0 * M_PI * static_cast<double>(i) / Nk, 2.0 * M_PI * static_cast<double>(j) / Nk};
        } // j

    } // i

    // Container to store the result of the calculation
    std::vector<complex<double>> wilson_loop_spectrum(Nk);
    std::vector<double> wilson_loop_phases(Nk);

    Utype U_cache; Utype U;
    SelfAdjointEigenSolver<Matrix2cd> es_Ham;
    ComplexEigenSolver<Utype> es_Wilson;

    for (int n = 0; n < Nk; ++n) {
        // Initialize the Wilson loop along the ky cycle to the identity matrix
        auto Wilson_loop{COMPLEX1};

        // the kpoint (n, 0)
        auto k0{kpoints(n, 0)};
        es_Ham.compute(Hamiltonian(k0, M));
        U_cache = es_Ham.eigenvectors().block(0, 0, NUM_BANDS, PROJ_SUBSPACE);

        // loop over points m = 1 ---> m = Nk - 1
        for (int m = 1; m < Nk; ++m) {
            auto k{kpoints(n, m)};
            // First, spectrum of the Bloch Hamiltonian
            es_Ham.compute(Hamiltonian(k, M));
            U = es_Ham.eigenvectors().block(0, 0, NUM_BANDS, PROJ_SUBSPACE);

            Wilson_loop = Wilson_loop * U_cache.adjoint() * U;
            U_cache = std::move(U);

        } // m

        // Link (Nk - 1, 0)
        es_Ham.compute(Hamiltonian(k0, M));
        U = es_Ham.eigenvectors().block(0, 0, NUM_BANDS, PROJ_SUBSPACE);
        Wilson_loop = Wilson_loop * U_cache.adjoint() * U;

        // Finally, compute the spectrum at n
        wilson_loop_spectrum[n] = Wilson_loop;


    } // n

    // Next, we need to take the logarithm of the result
    for (int n = 0; n < Nk; ++n) {
     wilson_loop_phases[n] = std::arg(wilson_loop_spectrum[n]);
    }


    // Next, the logarithm has a branch cut and monodromy of 2 pi . So, we need to map all phases to the fundamental domain

    double prev_phase{wilson_loop_phases[0]};
    for (int n = 1; n < Nk; n++) {
     const double this_phase = wilson_loop_phases[n];
     double delta = this_phase - prev_phase;
     if (delta > M_PI) {delta -= 2 * M_PI;}
     else if (delta < -M_PI) {delta += 2 * M_PI;}
     wilson_loop_phases[n] = prev_phase + delta;
     prev_phase = wilson_loop_phases[n];
    }




    // print the spectra
    std::ofstream out("Wilson_Loop_Phases");   // OUTPUT FILE STREAM
    for (int n = 0; n < Nk; n++) {
        out << std::left
                << std::fixed
                << std::setw(30)
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << wilson_loop_phases[n]
                << std::endl;
    }

    // Print the kpoints used in the calculation

    std::ofstream outk("Kpoints");   // OUTPUT FILE STREAM
    for (int n = 0; n < Nk; n++) {
        for (int m = 0; m < Nk; m++) {
            outk << std::left
                    << std::fixed
                    << std::setw(30)
                    << std::setprecision(std::numeric_limits<double>::max_digits10)
                    << kpoints(n, m)(0)
                    << std::setw(40)
                    << kpoints(n, m)(1)
                    << std::endl;
        }
    }

    double res{0};
    for (int n = 0; n < Nk - 1; n++) {
           res += wilson_loop_phases[n + 1] - wilson_loop_phases[n];
       }
    const double chern_number = res / (2.0 * M_PI);
    std::cout << "Chern number  =>   " << chern_number << std::endl;





    return 0;
}
