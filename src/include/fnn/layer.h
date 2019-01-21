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
#include <random>
#include <string>
#include "../maths/matrix.h"
#include "../maths/utils.h"
#include "../maths/activations.h"

#ifndef DEEPLEARNING_LAYER_H
#define DEEPLEARNING_LAYER_H
namespace NeuralNetwork{
    class Layer{
        bool editable = true;
    public:
        std::string activation;
        LinAlg::Matrix<double> weights;
        std::vector <double> values;
        LinAlg::Matrix<double> derivatives;
        // constructor
        Layer(long row, long col, double lower_bound, double upper_bound, const std::string &activation = "identity");
        void lock();
        void unlock();
        void print();
        void activate();
    };
}

void NeuralNetwork::Layer::print() {
    // std::cout << "The current weights: " << std::endl;
    std::cout << this->weights << std::endl;
}


NeuralNetwork::Layer::Layer(long row, long col, double lower_bound, double upper_bound, const std::string &activation):
        weights(row, col, lower_bound, upper_bound),
        derivatives(row, col, 1){
    this->values = std::vector<double> (row, 0);
    this->activation = activation;
}

void NeuralNetwork::Layer::activate() {
    if (this->activation == "relu" || this->activation == "ReLU"){
        for (int i = 0; i < this->values.size(); i++){
            this->values[i] = activations::ReLU(this->values[i]);
        }
    } else if (this->activation == "sigmoid") {
        for (int i = 0; i < this->values.size(); i++) {
            this->values[i] = activations::sigmoid(this->values[i]);
        }
    } else if (this->activation == "tanh") {
        for (int i = 0; i < this->values.size(); i++) {
            this->values[i] = activations::tanh(this->values[i]);
        }
    } else if (this->activation == "softmax") {
        double sum_exp = 0;
        for (int i = 0; i < this->values.size(); i++) {
            sum_exp += exp(this->values[i]);
        }
        for (int i = 0; i < this->values.size(); i++) {
            this->values[i] /= sum_exp;
        }
    }
}

#endif //DEEPLEARNING_LAYER_H
