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

#ifndef DEEPLEARNING_KMEAN_H
#define DEEPLEARNING_KMEAN_H
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include "../maths/utils.h"
#include "../maths/stats.h"

namespace machine_learning{
    namespace clustering{
        std::vector<std::vector<double>> k_mean_clustering(std::vector<std::vector<double>> points, int k, bool kpp=false){

            for (int i = 1; i < points.size(); i++){
                assert(points[0].size() == points[i].size());
            }

            std::vector<std::vector<double>> new_centers(k);
            std::vector<std::vector<double>> old_centers(k);
            std::vector<int> counter(k);
            std::vector<int> labels(points.size());

            // initialize the centers
            std::random_device r;
            std::default_random_engine e(r());
            std::uniform_int_distribution<int> uniform_dist(0, k);
            if (!kpp){
                for (int i = 0; i < k; i++){
                    new_centers[i] = points[uniform_dist(e)];
                }
            } else {
                // the k-mean++ initialization
                new_centers[0] = points[uniform_dist(e)];
                for (int i = 1; i < k; i++){
                    std::vector<double> distances(points.size());
                    double sum = 0;
                    for (int j = 0; j < points.size(); j++){
                        distances[j] = maths::distance_squared(new_centers[i-1], points[j]);
                        sum += distances[j];
                    }
                    for (int j = 0; j < points.size(); j++){
                        distances[j] /= sum;
                    }
                    new_centers[i] = points[statistics::random_from_discrete_prob_space(distances)];
                }
            }

            while (new_centers!= old_centers){
                // clustering with old_centers
                // reset counter
                counter = std::vector<int>(k, 0);
                for (int i = 0; i < points.size(); i++){
                    double min_distance = maths::distance(points[i], new_centers[0]);
                    labels[i] = 0;
                    for (int j = 1; j < k; j++){
                        double temp = maths::distance(points[i], new_centers[j]);
                        if (temp < min_distance){
                            min_distance = temp;
                            labels[i] = j;
                        }
                    }
                    counter[labels[i]] ++;
                    old_centers = new_centers;
                }

                // reset the centers
                new_centers = std::vector<std::vector<double>> (k, std::vector<double>(points[0].size(), 0));
                // choosing new centers
                for (int i = 0; i < points.size(); i++){
                    for (int j = 0; j < points[i].size(); j++){
                        new_centers[labels[i]][j] += points[i][j] / counter[labels[i]];
                    }
                }
            }
            return new_centers;
        }
    }
}

#endif //DEEPLEARNING_KMEAN_H
