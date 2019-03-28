/*
                                ╔═════════════════════════════════════════════╗
                                ║             .-') _                          ║
                                ║            (  OO) )                         ║
                                ║          ,(_)----. .---. .-----.            ║
                                ║          |       |/_   |/  -.   \           ║
                                ║          '--.   /  |   |'-' _'  |           ║
                                ║          (_/   /   |   |   |_  <            ║
                                ║           /   /___ |   |.-.  |  |           ║
                                ║          |        ||   |\ `-'   /           ║
                                ║          `--------'`---' `----''     ©2019  ║
                                ╚═════════════════════════════════════════════╝
 * This is an implementation of the mean shift algorithm given in
 * http://faculty.ucmerced.edu/mcarreira-perpinan/papers/mean-shift-review.pdf
 * We decided against using the matrix form
 */

#ifndef DEEPLEARNING_MEANSHIFT_H
#define DEEPLEARNING_MEANSHIFT_H
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include "../maths/utils.h"
#include "../maths/stats.h"
#include "../maths/numerical_analysis.h"

namespace machine_learning {
    namespace clustering {
        class Gaussian_mean_shift{
        private:
            std::vector<std::vector<double>> points;
            std::vector<int> labels;
            double sigma_squared;
            double p_xn_given_x(int n, std::vector<double> x);
        public:
            Gaussian_mean_shift(const std::vector<std::vector<double>> &points, const double &sigma){
                this->points = points;
                this->sigma_squared = pow(sigma, 2);
                this->labels = std::vector<int>(points.size(), 0);
            }
            void fit(double tol, double epsilon);

            int get_label(int index){
                return labels[index];
            }
        };
    }
}

double machine_learning::clustering::Gaussian_mean_shift::p_xn_given_x(int n, std::vector<double> x){
    double denominator = 0;
    for (int i = 0; i < points.size(); i++){
        denominator += exp(-maths::distance_squared(x, this->points[i]) / (2 * this->sigma_squared));
    }
    return exp(-maths::distance_squared(x, points[n]) / (2 * this->sigma_squared)) / denominator;
}

void machine_learning::clustering::Gaussian_mean_shift::fit(double tol, double epsilon) {
    int N = this->points.size();
    std::vector<std::vector<double>> z(N);
    std::vector<double> x;
    std::vector<double> p(N, 0);
    for (int i = 0; i < N; i++){
        x = this->points[i];
        std::vector<double> x_old;
        do{
            std::vector<double> temp(this->points[0].size(), 0);
            x_old = x;
#pragma omp parallel for
            for (int j = 0; j < N; j++){
                p[j] = this->p_xn_given_x(j, x);
                for (int k = 0; k < this->points[0].size(); k++){
                    temp[k] += p[j] * points[j][k];
                }
            }
            x = temp;
            // std::cout << "distance to old node: " << maths::distance(x, x_old)  << std::endl ;
        } while (maths::distance(x, x_old) > tol);
        z[i] = x;
        std::cout << "z[" << i << "] = ";
        for (int i = 0; i < x.size(); i++){
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }
    // efficient connected-components algorithm
    int K = 0;
    bool assigned = false;
    std::vector<std::vector<double>> c;
    c.push_back(z[0]);
    this->labels[0] = 0;
    for (int i = 1; i < N; i++){
        assigned = false;
        for (int k = 0; k < K; k++){
            if (maths::distance(z[i], c[k]) < epsilon){
                this->labels[i] = k;
                assigned = true;
                break;
            }
        }
        if (!assigned){
            K++;
            c.push_back(z[i]);
            this->labels[i] = K;
        }
    }
}

/*
double kernel_density(std::vector<double> x,
        double (*func)(double),
        std::vector<std::vector<double>> points,
        double sigma_squared){
    double result = 0;
    u_long N = points.size();
    for (int i = 0; i < N; i++){
        result += func(maths::distance_squared(x, points[i]) / sigma_squared) / N;
    }
    return result;
}
*/

/*
std::vector<double> f(std::vector<double> x,
        std::vector<std::vector<double>> points,
        double (*func)(double),
        double sigma_squared,
        bool is_Gaussian){
    double scalar = 0;
    std::vector<double> result_vector = std::vector<double>(points[0].size(), 0);
    if (is_Gaussian){
        for (int i = 0; i < points.size(); i++) {
            scalar = p_xn_given_x(i, x, points, sigma_squared);
            for (int j = 0; j < points[0].size(); j++){
                result_vector[j] += scalar * points[i][j];
            }
        }
        return result_vector;
    } else {
        double denominator = 0;
        for (int i = 0; i < points.size(); i++){
            denominator += maths::differentiate(func, maths::distance_squared(x, points[i])/sigma_squared);
        }

        for (int i = 0; i < points.size(); i++) {
            scalar = maths::differentiate(func, maths::distance_squared(x, points[i])/sigma_squared) / denominator;
            for (int j = 0; j < points[0].size(); j++){
                result_vector[j] += scalar * points[i][j];
            }
        }
        return result_vector;
    }
}
 */
#endif //DEEPLEARNING_MEANSHIFT_H
