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

#ifndef DEEPLEARNING_LOSSES_H
#define DEEPLEARNING_LOSSES_H
#include <vector>
#include <cmath>

namespace loss{
    template <typename Type>
    double squared_loss(const std::vector<Type> &forecast, const std::vector<Type> &actual){
        assert(forecast.size() == actual.size());
        double loss = 0.0;
        for (long i = 0; i < forecast.size(); i++){
            loss += std::pow(forecast[i] - actual[i], 2);
        }
        return loss;
    }

    template <typename Type>
    double mean_squared_error(const std::vector<Type> &forecast, const std::vector<Type> &actual){
        return squared_loss(forecast, actual) / forecast.size();
    }

    template <typename Type>
    double absolute_error(const std::vector<Type> &forecast, const std::vector<Type> &actual) {
        assert(forecast.size() == actual.size());
        double loss = 0.0;
        for (long i = 0; i < forecast.size(); i++) {
            loss += std::abs(forecast[i] - actual[i]);
        }
        return loss;
    }

    template <typename Type>
    double mean_absolute_error(const std::vector<Type> &forecast, const std::vector<Type> &actual){
        return absolute_error(forecast, actual) / forecast.size();
    }

}
#endif //DEEPLEARNING_LOSSES_H
