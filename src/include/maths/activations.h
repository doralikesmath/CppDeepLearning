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
#include <vector>
#include <cmath>
#include <string>
#include "utils.h"

#ifndef DEEPLEARNING_ACTIVATIONS_H
#define DEEPLEARNING_ACTIVATIONS_H

namespace activations{
    double sigmoid(double x){
        return 1 / (1 + exp(-x));
    }

    double sigmoid_derivative(double x){
        double a = sigmoid(x);
        return a * (1 - a);
    }

    double tanh(double x){
        return (exp(x) - 1) / (exp(x) + 1);
    }

    double tanh_derivative(double x){
        return 1 - std::pow(tanh(x), 2);
    }

    double hard_tanh(double x){
        return maths::max(maths::min(x, 1.0), -1.0);
    }

    double hard_tanh_derivative(double x){
        return ((-1 <= x) and (x <= 1))? 1 : 0;
    }

    double ReLU(double x){
        return maths::max(x, 0.0);
    }

    double ReLU_derivative(double x){
        return (x >= 0) ? 1 : 0;
    }

    double activation(double x, std::string type){
        if (type == "ReLU" or type == "relu"){
            return ReLU(x);
        } else if (type == "hard_tanh" or type == "hard tanh"){
            return hard_tanh(x);
        } else if (type == "sigmoid"){
            return sigmoid(x);
        } else{
            return x;
        }
    }

    double activation_derivative(double x, std::string type){
        if (type == "ReLU" or type == "relu"){
            return ReLU_derivative(x);
        } else if (type == "hard_tanh" or type == "hard tanh"){
            return hard_tanh_derivative(x);
        } else if (type == "sigmoid"){
            return sigmoid_derivative(x);
        } else{
            return x;
        }
    }

}
#endif //DEEPLEARNING_ACTIVATIONS_H
