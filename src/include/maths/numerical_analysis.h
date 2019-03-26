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
 */
#include <cmath>
#include <vector>

#ifndef DEEPLEARNING_NUMERICAL_ANALYSIS_H
#define DEEPLEARNING_NUMERICAL_ANALYSIS_H

namespace maths{
    double integrate(double a, double b, int n, double (*func)(double)){
        // a very naive approach to numerically integrate functions
        // the Tanh-Sinh Quadrature algorithm will be implemented later
        // integrate func over [a, b] using n intervals
        double x = a;
        const double dx = (b - a) / (double) n;
        double val = 0.5 * func(x);
        x += dx;
        for (int i = 1; i < n; i++, x += dx){
            val += func(x);
        }
        val += 0.5 * func(x);
        return val * dx;
    }

    double differentiate(double (*func)(double), double x){
        double epsilon = std::numeric_limits<double>::epsilon();
        double sqrt_epsilon = std::sqrt(epsilon);
        if (x == 0){
            return (func(epsilon) - func(-epsilon)) / (2 * epsilon);
        } else {
            return (func(x * (1 + sqrt_epsilon)) - func(x * (1 - sqrt_epsilon))) / (2 * x * sqrt_epsilon);
        }
    }
};

#endif //DEEPLEARNING_NUMERICAL_ANALYSIS_H
