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
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../include/matrix.h"

// verify if the input data is indeed a matrix
bool is_matrix(const std::vector < std::vector<double> > &M){
    int m = M.size();
    int n = M[0].size();
    for (int i = 0; i < m; i++){
        if (M[i].size() != n){
            return false;
        }
    }
    return true;
}

// the constructor function
LinAlg::Matrix::Matrix(const std::vector < std::vector<double> > &M){
    assert(::is_matrix(M) && "The input is not a matrix.");
    this->M = M;
    this->shape[0] = M.size();
    this->shape[1] = M[0].size();
}

// returns the transpose of the matrix
LinAlg::Matrix LinAlg::Matrix::T(){
    std::vector <std::vector <double> > N;
    for (int i = 0; i < this->shape[1]; i++){
        std::vector <double> temp_row;
        for (int j = 0; j < this->shape[0]; j++){
            temp_row.push_back(this->M[j][i]);
        }
        N.push_back(temp_row);
    }
    return LinAlg::Matrix(N);
}

void LinAlg::Matrix::print(){
    long m = this->shape[0];
    long n = this->shape[1];

    short fixed_width = 5;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n - 1; j++){
            int w = (int)log10(this->M[i][j]) + 3;
            if (w > fixed_width){
                fixed_width = w;
            }
        }
    }

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n - 1; j++){
            std::cout << std::setw(fixed_width) << this->M[i][j] << ",";
        }
        std::cout << std::setw(fixed_width) << this->M[i][n-1] << std::endl;
    }
}

LinAlg::Matrix LinAlg::operator+ (const LinAlg::Matrix &first_matrix, const LinAlg::Matrix &second_matrix){
    assert(first_matrix.shape[0] == second_matrix.shape[0] and first_matrix.shape[1] == second_matrix.shape[1]);
    std::vector<std::vector<double>> result = first_matrix.M;
    for (int i = 0; i < first_matrix.shape[0]; i++){
        for (int j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] += second_matrix.M[i][j];
        }
    }
    return LinAlg::Matrix(result);
}

LinAlg::Matrix LinAlg::operator- (const LinAlg::Matrix &first_matrix, const LinAlg::Matrix &second_matrix){
    assert(first_matrix.shape[0] == second_matrix.shape[0] and first_matrix.shape[1] == second_matrix.shape[1]);
    std::vector<std::vector<double>> result = first_matrix.M;
    for (int i = 0; i < first_matrix.shape[0]; i++){
        for (int j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] -= second_matrix.M[i][j];
        }
    }
    return LinAlg::Matrix(result);
}

LinAlg::Matrix LinAlg::operator* (const LinAlg::Matrix &first_matrix, const double &c){
    std::vector<std::vector<double>> result = first_matrix.M;
    for (int i = 0; i < first_matrix.shape[0]; i++){
        for (int j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] *= c;
        }
    }
    return LinAlg::Matrix(result);
}

LinAlg::Matrix LinAlg::operator* (const double &c, const LinAlg::Matrix &first_matrix){
    std::vector<std::vector<double>> result = first_matrix.M;
    for (int i = 0; i < first_matrix.shape[0]; i++){
        for (int j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] *= c;
        }
    }
    return LinAlg::Matrix(result);
}