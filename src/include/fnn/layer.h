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
        std::vector <double> pre_activate;
        std::vector <double> post_activate;
        LinAlg::Matrix<double> weight_derivatives;
        std::vector <double> value_derivatives;
        // constructor
        Layer(long row, long col, double lower_bound, double upper_bound, const std::string &activation = "identity");
        void lock();
        void unlock();
        void print();
        void activate();
        double activate_derivative(int i);
    };
}

void NeuralNetwork::Layer::print() {
    // std::cout << "The current weights: " << std::endl;
    std::cout << this->weights << std::endl;
}


NeuralNetwork::Layer::Layer(long row, long col, double lower_bound, double upper_bound, const std::string &activation):
        weights(row, col, lower_bound, upper_bound),
        weight_derivatives(row, col, 1){
    this->pre_activate = std::vector<double> (row, 0);
    this->post_activate = std::vector<double> (row, 0);
    this->value_derivatives = std::vector<double> (row, 1);
    this->weights = LinAlg::Matrix<double> (row, col, lower_bound, upper_bound);
    this->activation = activation;
}

void NeuralNetwork::Layer::activate() {
    if (this->activation == "relu" || this->activation == "ReLU"){
#pragma omp parallel for
        for (int i = 0; i < this->pre_activate.size(); i++){
            this->post_activate[i] = activations::ReLU(this->pre_activate[i]);
        }
    } else if (this->activation == "sigmoid") {
#pragma omp parallel for
        for (int i = 0; i < this->pre_activate.size(); i++) {
            this->post_activate[i] = activations::sigmoid(this->pre_activate[i]);
        }
    } else if (this->activation == "tanh") {
#pragma omp parallel for
        for (int i = 0; i < this->pre_activate.size(); i++) {
            this->post_activate[i] = activations::tanh(this->pre_activate[i]);
        }
    } else if (this->activation == "softmax") {
        double sum_exp = 0;
#pragma omp parallel for
        for (int i = 0; i < this->pre_activate.size(); i++) {
            sum_exp += exp(this->pre_activate[i]);
        }
#pragma omp parallel for
        for (int i = 0; i < this->pre_activate.size(); i++) {
            this->post_activate[i] = exp(this->pre_activate[i]) / sum_exp;
        }
    } else{
        this->post_activate = this->pre_activate;  // the default activation is the identity map
    }
}

double NeuralNetwork::Layer::activate_derivative(int i) {
    if (this->activation == "relu" || this->activation == "ReLU"){
        return (this->pre_activate[i] > 0)? 1 : 0;
    } else if (this->activation == "sigmoid") {
        return this->post_activate[i] * (1 - this->post_activate[i]);
    } else if (this->activation == "tanh") {
        return 1 - pow(this->post_activate[i], 2);
    } else if (this->activation == "hard_tanh") {
        return (-1 <= this->post_activate[i] && this->post_activate[i] <= 1)? 1 : 0;
    } else{
        return 1;  // the default activation is the identity map
    }
}

#endif //DEEPLEARNING_LAYER_H
