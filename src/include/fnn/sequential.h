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
        int number_of_epochs;
        std::vector<int> params;
        int number_of_layers;
    public:
        std::vector<Layer> layers;
        std::vector<std::string> activations;
        Sequential();
        Sequential(std::vector<int> params);
        Sequential(int input_size);
        void add_layer(int number_of_neuron, std::string activation = "identity");
        void compile();
        void forward_pass();
        void back_propagation();
        void fit();
        void summarize();
        void print_weights(int layer);
        void print_values(int layer);
        std::vector<double> predict();
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

void NeuralNetwork::Sequential::compile() {
    this->number_of_layers = this->params.size();
    for (int i = 0; i < params.size() - 1; i++){
        this->layers.push_back(Layer(this->params[i], this->params[i+1], 0, 0.5, this->activations[i]));
    }
    layers.push_back(Layer(this->params[this->number_of_layers - 1], 1, 0, 0.5, this->activations[this->number_of_layers-1]));
    this->compiled = true;
}

void NeuralNetwork::Sequential::forward_pass() {
    for (int i = 1; i< this->number_of_layers; i++){
        this->layers[i].values = this->layers[i-1].weights.transpose() * this->layers[i-1].values;
        this->layers[i].activate();
    }
}

void NeuralNetwork::Sequential::print_weights(int layer) {
    std::cout << "============================================" << std::endl;
    std::cout << "The weights in layer " << layer << " : " << std::endl;
    std::cout << this->layers[layer].weights << std::endl;
}

void NeuralNetwork::Sequential::print_values(int layer) {
    std::cout << "============================================" << std::endl;
    std::cout << "The values in layers " << layer << " : " << std::endl;
    for (int i = 0; i < this->layers[layer].values.size(); i++){
        std::cout << this->layers[layer].values[i] << " ";
    }
    std::cout << std::endl;
}

void NeuralNetwork::Sequential::back_propagation() {

}

void NeuralNetwork::Sequential::fit() {

}

void NeuralNetwork::Sequential::summarize() {
    if (this->compiled) {
        std::cout << "================ Summarize ================" << std::endl;
        std::cout << "Number of layers: " << this->number_of_layers << std::endl;
        std::cout << "============================================" << std::endl;
        long param = 0;
        for (int i = 0; i < this->number_of_layers; i++) {
            std::cout << "Layer " << i << " - Size: " << this->layers[i].weights.shape[0] << " x "
                      << this->layers[i].weights.shape[1]
                      << " - Activation: " << this->layers[i].activation << std::endl;
            param += this->layers[i].weights.shape[0] * this->layers[i].weights.shape[1];
        }
        param -= this->layers[this->number_of_layers-1].weights.shape[0] * this->layers[this->number_of_layers-1].weights.shape[1];
        std::cout << "============================================" << std::endl;
        std::cout << "Total number of parameters: " << param << std::endl;
    } else{
        std::cout << "The model needs to be compiled first." << std::endl;
    }
}


#endif //DEEPLEARNING_NETWORK_H
