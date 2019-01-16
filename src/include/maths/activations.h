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
#include "utils.h"

#ifndef DEEPLEARNING_ACTIVATIONS_H
#define DEEPLEARNING_ACTIVATIONS_H

namespace activations{
    double sigmoid(double x){
        return 1 / (1 + exp(x));
    }

    double tanh(double x){
        return (exp(x) - 1) / (exp(x) + 1);
    }

    double hard_tanh(double x){
        return maths::max(maths::min(x, 1.0), -1.0);
    }

    double ReLU(double x){
        return maths::max(x, 0.0);
    }
}
#endif //DEEPLEARNING_ACTIVATIONS_H
