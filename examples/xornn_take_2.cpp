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
    NeuralNetwork::Sequential nn;
    nn.add_layer(2, "sigmoid");
    nn.add_layer(10, "tanh");
    nn.add_layer(3);
    nn.add_layer(1);
    nn.compile();
    nn.layers[0].pre_activate = {1, 1};
    nn.summarize();
    nn.print_values(1);
    nn.forward_pass();
    nn.print_values(1);
    return 0;
}