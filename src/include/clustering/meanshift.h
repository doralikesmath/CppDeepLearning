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
        namespace mean_shift{
            class Gaussian_mean_shift{
            public:
                std::vector<std::vector<double>> points;
                double sigma_squared;
                bool is_Gaussian;
            };

            class Gaussian_blurring_mean_shift{
            public:
                std::vector<std::vector<double>> points;
                double sigma_squared;
                bool is_Gaussian;
            };
        }


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

        double p_xn_given_x(int n,
                std::vector<double> x,
                std::vector<std::vector<double>> points,
                double sigma_squared){
            double denominator = 0;
            for (int i = 0; i < points.size(); i++){
                denominator += exp(-maths::distance_squared(x, points[i]) / (2 * sigma_squared));
            }
            return exp(-maths::distance_squared(x, points[n]) / (2 * sigma_squared)) / denominator;
        }

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
    }
}
#endif //DEEPLEARNING_MEANSHIFT_H
