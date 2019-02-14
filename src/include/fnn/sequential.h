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

 * This is a class of very simple sequential neural network where the input of each neuron is just a single entry.
 * We will develope a tensor module for dealing with more complex neural networks later.
 */
#include <vector>
#include <cassert>
#include <cmath>
#include <ctime>
#include <random>
#include <string>
#include "../maths/matrix.h"
#include "../maths/utils.h"
#include "../maths/activations.h"
#include "layer.h"
#ifndef DEEPLEARNING_SEQUENTIAL_H
#define DEEPLEARNING_SEQUENTIAL_H

namespace NeuralNetwork{
    class Sequential{
        bool verbose;
        bool continuously_backup;
        bool compiled = false;
        double learning_rate;
        int number_of_epochs;
        std::vector<int> params;
        int number_of_layers;
        std::string loss;

        void forward_pass();
        void back_propagate(std::vector<double> expected);
        void update_weights();
    public:
        std::vector<Layer> layers;
        std::vector<std::string> activations;
        Sequential();
        Sequential(std::vector<int> params);
        void add_layer(int number_of_neuron, std::string activation = "identity");
        void compile(int number_of_epochs = 100000, double learning_rate = 0.5, std::string loss = "squared_error");
        template <typename Type>
        void fit(std::vector<std::vector<Type>> X, std::vector<std::vector<Type>> y);
        void summarize();
        void print_weights(int layer);
        void print_values(int layer);
        template <typename Type>
        std::vector<double> predict(std::vector<Type> input);
        void save_model(std::string filename);
        void load_model(std::string filename);
    };
}

NeuralNetwork::Sequential::Sequential() {
}

NeuralNetwork::Sequential::Sequential(std::vector<int> params) {
    this->number_of_layers = params.size();
    for (int i = 0; i < params.size() - 1; i++){
        this->layers.push_back(Layer(params[i], params[i+1], 0, 0.5));
    }
    layers.push_back(Layer(params[params.size() - 1], 1, 0, 0.5));
    this->compiled = true;
}


void NeuralNetwork::Sequential::add_layer(int number_of_neuron, std::string activation) {
    this->params.push_back(number_of_neuron);
    this->activations.push_back(activation);
}

void NeuralNetwork::Sequential::compile(int number_of_epochs, double learning_rate, std::string loss) {
    this->loss = loss;
    this->learning_rate = learning_rate;
    this->number_of_epochs = number_of_epochs;
    this->number_of_layers = this->params.size();
    for (int i = 0; i < params.size() - 1; i++){
        this->layers.push_back(Layer(this->params[i], this->params[i+1], 0, 0.5, this->activations[i]));
    }
    layers.push_back(Layer(this->params[this->number_of_layers - 1], 1, 0, 0.5,
            this->activations[this->number_of_layers-1]));
    this->compiled = true;
}

void NeuralNetwork::Sequential::forward_pass() {
    this->layers[0].activate();
// #pragma omp parallel for
    for (int i = 1; i < this->number_of_layers; i++){
        this->layers[i].pre_activate = this->layers[i-1].weights.transpose() * this->layers[i-1].post_activate;
        this->layers[i].activate();
    }
}

void NeuralNetwork::Sequential::update_weights() {
// #pragma omp parallel for
    for (int i = 0; i < this->number_of_layers; i++){
        for (int j = 0; j < this->layers[i].weights.shape[0]; j++){
            for (int k = 0; k < this -> layers[i].weights.shape[1]; k++){
                this->layers[i].weights(j, k) -= this->learning_rate * this->layers[i].weight_derivatives(j, k);
            }
        }
    }
}

void NeuralNetwork::Sequential::back_propagate(std::vector<double> expected) {
    double temp;
    int n = this->number_of_layers - 1;
    if (this->loss == "squared_error"){
#pragma omp parallel for
        for (int i = 0; i < this->layers[n].value_derivatives.size(); i++){
            this->layers[n].value_derivatives[i] =
                    (this->layers[this->number_of_layers - 1].post_activate[i] - expected[i]) *
                    this->layers[n].activate_derivative(i);
        }
    }

// #pragma omp parallel for
    for (int i = n - 1; i >= 0; i--){
// #pragma omp parallel for
        for (int j = 0; j < this->layers[i].value_derivatives.size(); j++){
            temp = 0;
            // the dot product of 2 vectors, really
// #pragma omp parallel for
            for (int k = 0; k < this->layers[i+1].value_derivatives.size(); k++){
                temp += this->layers[i].weights(j, k) * this->layers[i+1].value_derivatives[k];
            }
            this->layers[i].value_derivatives[j] = temp * this->layers[i].activate_derivative(j);
        }
    }

// #pragma omp parallel for
    for (int i = 0; i < n; i++){
        for (int j = 0; j < this->layers[i].weights.shape[0]; j++){
            for (int k = 0; k < this->layers[i].weights.shape[1]; k++){
                this->layers[i].weight_derivatives(j, k) = this->layers[i+1].value_derivatives[k] *
                        this->layers[i].post_activate[j];
            }
        }
    }

    this->update_weights();
}

template <typename Type>
void NeuralNetwork::Sequential::fit(std::vector<std::vector<Type>> X, std::vector<std::vector<Type>> y) {
    std::cout << "================================ Training Started ==============================" << std::endl;
    clock_t t;
    t = clock();
    double avg_loss = 0;
    long count = 0;

// #pragma omp parallel for
    for (int i = 0; i < this->number_of_epochs; i++){
        for (int j = 0; j< X.size(); j++){
            count++;
            this->layers[0].pre_activate = X[j];
            this->forward_pass();
            std::vector<double> predicted = this->layers[this->number_of_layers - 1].post_activate;
            double loss = 0;
            for (int k = 0; k < y[j].size(); k++){
                loss += pow(predicted[k] - y[j][k], 2);
            }
            avg_loss = avg_loss * ((count - 1) / count) + loss * loss / count;
            this->back_propagate(y[j]);
        }
    }
    t = clock() - t;
    std::cout << "Number of epochs: " << count << "." << std::endl;
    std::cout << "Training time: " << ((float)t)/CLOCKS_PER_SEC << " seconds." << std::endl;
    std::cout << "Average loss: " << avg_loss <<  "." << std::endl;
    std::cout << "=============================== Training Finished ==============================" << std::endl;
}

template <typename Type>
std::vector<double> NeuralNetwork::Sequential::predict(std::vector<Type> input) {
    this->layers[0].pre_activate = input;
    this->forward_pass();
    return this->layers[this->number_of_layers - 1].post_activate;
}

void NeuralNetwork::Sequential::print_weights(int layer) {
    std::cout << "The weights in layer " << layer << " : " << std::endl;
    std::cout << this->layers[layer].weights << std::endl;
}

void NeuralNetwork::Sequential::print_values(int layer) {
    std::cout << "The pre-activate values in layers " << layer << " : " << std::endl;
    for (int i = 0; i < this->layers[layer].pre_activate.size(); i++){
        std::cout << this->layers[layer].pre_activate[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "The post-activate values in layers " << layer << " : " << std::endl;
    for (int i = 0; i < this->layers[layer].post_activate.size(); i++){
        std::cout << this->layers[layer].post_activate[i] << " ";
    }
    std::cout << std::endl;
}

void NeuralNetwork::Sequential::summarize() {
    if (this->compiled) {
        std::cout << "==================================== Summarize =================================" << std::endl;
        std::cout << "Number of layers: " << this->number_of_layers << std::endl;
        std::cout << "================================================================================" << std::endl;
        long param = 0;
        for (int i = 0; i < this->number_of_layers; i++) {
            std::cout << "Layer " << i << " - Size: " << this->layers[i].weights.shape[0] << " x "
                      << this->layers[i].weights.shape[1]
                      << " - Activation: " << this->layers[i].activation << std::endl;
            param += this->layers[i].weights.shape[0] * this->layers[i].weights.shape[1];
        }
        param -= this->layers[this->number_of_layers-1].weights.shape[0] * this->layers[this->number_of_layers-1].weights.shape[1];
        std::cout << "================================================================================" << std::endl;
        std::cout << "Total number of parameters: " << param << std::endl;
    } else{
        std::cout << "The model needs to be compiled first." << std::endl;
    }
}

#endif //DEEPLEARNING_NETWORK_H
