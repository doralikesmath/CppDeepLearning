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
#include <cassert>
#include <cmath>
#include "../maths/matrix.h"
#include "../maths/utils.h"
#include "../maths/activations.h"

#ifndef DEEPLEARNING_NETWORK_H
#define DEEPLEARNING_NETWORK_H

namespace NeuralNetwork{
    struct Layer{
        LinAlg::Matrix<double> weights;
        std::vector <double> values;
    };
    class Sequential{
        std::vector<LinAlg::Matrix<double>> layers;
    };
}
#endif //DEEPLEARNING_NETWORK_H
