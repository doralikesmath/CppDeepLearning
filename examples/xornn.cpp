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

 * We will implement a very simple feed-forward neural network for the XOR function without using any library beside
 * the standard C++ ones.
 * +---+---+---------+
 * | X | Y | X XOR Y |
 * +---+---+---------+
 * | 0 | 0 |    0    |
 * | 0 | 1 |    1    |
 * | 1 | 0 |    1    |
 * | 1 | 1 |    0    |
 * +---+---+---------+
 */

#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include "../src/include/maths/activations.h"
/*
 * Our network will have 2 input nodes, 1 hidden layers with 3 neurons, activation function is just the sigmoid,
 * and an output neuron with activation is the identity function.
 * There will be 9 weights values:
 * weights[0:2] : from input node 0 to the hidden layers
 * weights[3:6] : from input node 1 to the hidden layers
 * weights[7:9] : from the hidden layer to the output node
 * values[0:1] : the input values
 * values[2:4] : the hidden values
 * values[5] : the output value
 */
const std::vector<std::vector<int>> TRAINING_SET = {{0 ,0}, {1, 0}, {0, 1}, {1, 1}};
const std::vector <double > LABELS = {0, 1, 1, 0};
std::vector<double> values =  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
std::vector<double> weights;
std::vector<double> derivatives = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};


void forward_pass(){
    values[2] = activations::sigmoid(values[0] * weights[0] + values[1] * weights[3]);
    values[3] = activations::sigmoid(values[0] * weights[1] + values[1] * weights[4]);
    values[4] = activations::sigmoid(values[0] * weights[2] + values[1] * weights[5]);
    values[5] = values[2] * weights[6] + values[3] * weights[7] + values[4] * weights[8];
}

void backward_pass(double expected){
    double temp = values[5] - expected;
    derivatives[0] = temp * weights[6] * values[2] * (1 - values[2]) * values[0];
    derivatives[1] = temp * weights[7] * values[3] * (1 - values[3]) * values[0];
    derivatives[2] = temp * weights[8] * values[4] * (1 - values[4]) * values[0];
    derivatives[3] = temp * weights[6] * values[2] * (1 - values[2]) * values[1];
    derivatives[4] = temp * weights[7] * values[3] * (1 - values[3]) * values[1];
    derivatives[5] = temp * weights[8] * values[4] * (1 - values[4]) * values[1];
    derivatives[6] = temp * values[2];
    derivatives[7] = temp * values[3];
    derivatives[8] = temp * values[4];
}

void fit(double learning_rate, double normalization, int epochs){
    std::cout << "================================ Training Started ==============================" << std::endl;
    clock_t t;
    t = clock();
    double avg_loss = 0;
    long count = 0;
    #pragma omp parallel for
    for (int i = 0; i < epochs; i++){
        /*
        if ((i + 1) % 1000 == 0) {
            std::cout << "=============================== Epoch no " << i + 1 << " ==============================="
                      << std::endl;
        }*/
        #pragma omp parallel for
        for (int j = 0; j < 4; j++){
            count ++;
            values[0] = TRAINING_SET[j][0];
            values[1] = TRAINING_SET[j][1];
            double t_expected = (double)LABELS[j];
            forward_pass();
            double loss = abs(- t_expected + values[5]);
            /*
            if ((i+1) % 1000 == 0 ) {
                std::cout << values[0] << " XOR " << values[1] << " : expected = " << t_expected << " - prediction = "
                          << values[5] << std::endl;
            }*/
            avg_loss = avg_loss * ((count - 1) / count) + loss * loss / count;

            backward_pass(t_expected);
            // updating the weights
            #pragma omp parallel for
            for (int k = 0; k < 9; k++){
                weights[k] = weights[k] - learning_rate * derivatives[k];
            }
        }
        /*
        if ((i + 1) % 1000 == 0){
            std::cout << "The average loss is " << avg_loss << std::endl;
        }
        */
    }
    t = clock() - t;
    std::cout << "Number of epochs: " << count << "." << std::endl;
    std::cout << "Training time: " << ((float)t)/CLOCKS_PER_SEC << " seconds." << std::endl;
    std::cout << "Average loss: " << avg_loss <<  "." << std::endl;
    std::cout << "=============================== Training Finished ==============================" << std::endl;
}

double predict(int a, int b){
    double x = activations::sigmoid(a * weights[0] + b * weights[3]);
    double y = activations::sigmoid(a * weights[1] + b * weights[4]);
    double z = activations::sigmoid(a * weights[2] + b * weights[5]);
    return x * weights[6] + y * weights[7] + z * weights[8];
}

int main(){
    std::uniform_real_distribution<double> unif(0, 0.5);
    std::default_random_engine re;
    for (int i = 0; i < 9; i++){
        weights.push_back(unif(re));
    }

    fit(0.3, 0.05, 2000000);

    std::cout << "0 XOR 0 is " << predict(0, 0) << std::endl;
    std::cout << "0 XOR 1 is " << predict(0, 1) << std::endl;
    std::cout << "1 XOR 0 is " << predict(1, 0) << std::endl;
    std::cout << "1 XOR 1 is " << predict(1, 1) << std::endl;
    return 0;
}