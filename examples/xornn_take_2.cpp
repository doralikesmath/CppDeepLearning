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

#include "../src/include/fnn/sequential.h"
#include "../src/include/maths/matrix.h"
#include <vector>
#include <iostream>

int main(){
    std::vector<std::vector<double>> TRAINING_SET = {{0 ,0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<std::vector <double >> LABELS = {{0}, {1}, {1}, {0}};

    NeuralNetwork::Sequential nn;
    nn.add_layer(2);
    nn.add_layer(4, "sigmoid");
    nn.add_layer(1);
    nn.compile(2000000, 0.3, "squared_error");
    nn.summarize();
    nn.fit(TRAINING_SET, LABELS);
    std::vector<double> results = nn.predict(std::vector<double>({0, 0}));
    std::cout << "0 XOR 0 is " << results[0] << std::endl;
    results = nn.predict(std::vector<double>({0, 1}));
    std::cout << "0 XOR 1 is " << results[0] << std::endl;
    results = nn.predict(std::vector<double>({1, 0}));
    std::cout << "1 XOR 0 is " << results[0] << std::endl;
    results = nn.predict(std::vector<double>({1, 1}));
    std::cout << "1 XOR 1 is " << results[0] << std::endl;
    return 0;
}