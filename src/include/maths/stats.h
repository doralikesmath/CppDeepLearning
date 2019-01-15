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

#ifndef DEEPLEARNING_STATS_H
#define DEEPLEARNING_STATS_H

namespace stats{
    template <typename Type>
    double mean(const std::vector<Type> &vec){
        double result = (double)vec[0];
        for (long i = 1; i < vec.size(); i++){
            result = (i * result + vec[i]) / (i + 1);
        }
        return result;
    }

    template <typename Type>
    double variance(const std::vector<Type> &vec){
        double m = mean(vec);
        double result = pow(vec[0] - m, 2);
        for (long i = 1; i < vec.size(); i++){
            result = (i * result + pow(vec[i] - m, 2)) / (i + 1);
        }
        return result;
    }

    template <typename Type>
    double standard_deviation(const std::vector<Type> &vec){
        return sqrt(variance(vec));
    }
}
#endif //DEEPLEARNING_STATS_H
