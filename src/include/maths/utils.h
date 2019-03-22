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
#include <cmath>
#include <cassert>

#ifndef DEEPLEARNING_UTILS_H
#define DEEPLEARNING_UTILS_H

namespace maths{
    template <typename Type>
    Type max(const std::vector<Type> &vec){
        Type max = vec[0];
        for (long i = 1; i < vec.size(); i++){
            if (max < vec[i])
                max = vec[i];
        }
        return max;
    }

    template <typename Type>
    Type max(const Type &a, const Type &b){
        return a > b ? a : b;
    }

    template <typename Type>
    Type min(const std::vector<Type> &vec){
        Type min = vec[0];
        for (long i = 1; i < vec.size(); i++){
            if (min > vec[i])
                min = vec[i];
        }
        return min;
    }

    template <typename Type>
    Type min(const Type &a, const Type &b){
        return a < b ? a : b;
    }

    template <typename Type>
    long argmax(const std::vector<Type> &vec){
        long max_index = 0;
        Type max = vec[0];
        for (long i = 1; i < vec.size(); i++){
            if (vec[i] > max){
                max = vec[i];
                max_index = i;
            }
        }
        return max_index;
    };

    template <typename Type>
    long argmin(const std::vector<Type> &vec){
        long min_index = 0;
        Type min = vec[0];
        for (long i = 1; i < vec.size(); i++){
            if (vec[i] < min){
                min = vec[i];
                min_index = i;
            }
        }
        return min_index;
    };

    template <typename Type>
    std::vector<double> softmax(const std::vector<Type> &vec){
        double sum = 0;
        std::vector<double> result(vec.size());
        for (long i = 0; i < vec.size(); i++){
            sum += exp(vec[i]);
        }
        for (long i = 0; i < vec.size(); i++){
            result[i] = exp(vec[i]) / sum;
        }
        return result;
    }

    int sign(double x){
        if (x == 0){
            return 0;
        } else
            return x > 0 ? 1 : -1;
    }

    template <typename Type>
    double l2_norm_squared(const std::vector<Type> &vec){
        double norm = 0;
        for (long i = 0; i < vec.size(); i++){
            norm += std::pow(vec[i], 2);
        }
        return norm;
    }

    template <typename Type>
    double l2_norm(const std::vector<Type> &vec){
        return sqrt(l2_norm_squared(vec));
    }

    template <typename Type>
    double distance_squared(const std::vector<Type> &x, const std::vector<Type> &y){
        assert(x.size() == y.size());
        double result = 0;
        for (int i = 0; i < x.size(); i++){
            result += std::pow(x[i] - y[i], 2);
        }
        return result;
    }

    template <typename Type>
    double distance(const std::vector<Type> &x, const std::vector<Type> &y){
        assert(x.size() == y.size());
        return std::sqrt(distance_squared(x, y));
    }

    std::vector<bool> one_hot_encoding(u_long size, u_long label){
        assert(label < size);
        std::vector<bool> result(size, false);
        result[label] = true;
        return result;
    }
}

#endif //DEEPLEARNING_UTILS_H
