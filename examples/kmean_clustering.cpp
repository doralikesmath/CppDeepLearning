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

#include "../src/include/clustering/kmean.h"
#include <vector>
#include <iostream>
#include <random>


int main(){

    int no_of_points = 10000;
    int dimension = 2;
    int no_of_cluster = 3;
    int no_of_test_points = 10;

    std::vector<std::vector<double>> points(no_of_points, std::vector<double>(dimension, 0));
    std::vector<std::vector<double>> test_points(no_of_test_points, std::vector<double>(dimension, 0));
    // generate a random set of points
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<double> uniform_dist(0, 100);

    for (int i = 0; i < no_of_points; i++){
        for (int j = 0; j < dimension; j++){
            points[i][j] = uniform_dist(e);
        }
    }

    for (int i = 0; i < no_of_test_points; i++){
        for (int j = 0; j < dimension; j++){
            test_points[i][j] = uniform_dist(e);
        }
    }

    std::cout << "================ k-mean ================" << std::endl;
    machine_learning::clustering::k_mean_clustering cluster(points);
    cluster.fit(no_of_cluster, false);
    for (int i = 0; i < no_of_cluster; i++){
        for (int j = 0; j < dimension; j++){
            std::cout << cluster.centers[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "========================================" << std::endl;

    for (int i = 0; i < no_of_test_points; i++){
        std::cout << "(" << test_points[i][0]
                  << ", " << test_points[i][1]
                  << "): cluster no " << cluster.predict(test_points[i])
                  << std::endl;
    }

    std::cout << "=============== k-mean++ ===============" << std::endl;
    cluster.fit(no_of_cluster, true);
    for (int i = 0; i < no_of_cluster; i++){
        for (int j = 0; j < dimension; j++){
            std::cout << cluster.centers[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "========================================" << std::endl;

    for (int i = 0; i < no_of_test_points; i++){
        std::cout << "(" << test_points[i][0]
                  << ", " << test_points[i][1]
                  << "): cluster no " << cluster.predict(test_points[i])
                  << std::endl;
    }

    return 0;
}