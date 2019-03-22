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
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef DEEPLEARNING_STATS_H
#define DEEPLEARNING_STATS_H

namespace statistics{
    template <typename Type>
    double mean(const std::vector<Type> &vec){
        double result = (double)vec[0];
        for (long i = 1; i < vec.size(); i++){
            result = (i * result + vec[i]) / (i + 1);
        }
        return result;
    }

    template <typename Type>
    double variance(const std::vector<Type> &vec){
        double m = mean(vec);
        double result = pow(vec[0] - m, 2);
        for (long i = 1; i < vec.size(); i++){
            result = (i * result + pow(vec[i] - m, 2)) / (i + 1);
        }
        return result;
    }

    template <typename Type>
    double standard_deviation(const std::vector<Type> &vec){
        return sqrt(variance(vec));
    }

    int random_from_discrete_prob_space(std::vector<double> PROBABILITIES){
        // we first check if PROBABILITIES is indeed a probability space
        double sum = 0;
        std::vector <double> MODIFIED_PROBABILITIES = {0};

        for (int i = 0; i < PROBABILITIES.size(); i++){
            // assert(PROBABILITIES[i] >= 0);
            sum += PROBABILITIES[i];
            MODIFIED_PROBABILITIES.push_back(sum);
        }

        // assert(1 - ZERO_THRESHOLD <= sum && sum <= 1 + ZERO_THRESHOLD);
        if (sum == 0){
            return 0;
        }
        std::random_device r;
        std::default_random_engine e(r());
        std::uniform_real_distribution<double> uniform_dist (0, 1);
        double rand = uniform_dist(e);
        /*
        while (rand == 1){
            rand = uniform_dist(e);
        }
        */
        for (int i = 0; i < MODIFIED_PROBABILITIES.size(); i++) {
            if ((rand >= MODIFIED_PROBABILITIES[i]) && (rand < MODIFIED_PROBABILITIES[i + 1])) {
                return i;
            }
        }
        return PROBABILITIES.size() - 1;
    }

    std::vector<int> random_indices_below_a_number(int result_length, int upper_cap){
        /*
         * This function will randomly generated a random vector of length result_length,
         * such that each entry is in [0, upper_cap] and is unique
         * :params:
         * result_length: length of the desired vector
         * upper_cap: the upper cap of the entries
         */
        assert(result_length <= upper_cap);
        std::vector<int> results(upper_cap);

        for (int  i = 0; i < upper_cap; i++){
            results[i] = i;
        }
        std::random_device r;
        std::default_random_engine e(r());
        std::shuffle(results.begin(), results.end(), e);
        return std::vector<int> (results.begin(), results.begin() + result_length);
    }
}
#endif //DEEPLEARNING_STATS_H
