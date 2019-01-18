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

int main(){
    NeuralNetwork::Sequential nn;
    nn.add_layer(2, "sigmoid");
    nn.add_layer(10, "tanh");
    nn.add_layer(3);
    nn.add_layer(1);
    nn.compile();
    nn.summarize();
    return 0;
}