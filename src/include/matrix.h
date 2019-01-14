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
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef DEEPLEARNING_MATRIX_H
#define DEEPLEARNING_MATRIX_H
namespace LinAlg{
    template <typename Type>
    class Matrix{
        public:
            std::vector < std::vector<Type> > M;
            long shape[2];
            Matrix(const std::vector < std::vector <Type> > &M);
            Matrix T();
            void print();
            double det();
            Matrix inverse();
            Type operator () (const long &row, const long &column);
    };
    template <typename Type>
    Matrix <Type> operator + (const Matrix <Type> &first_matrix, const Matrix <Type> &second_matrix);
    template <typename Type>
    Matrix <Type> operator - (const Matrix <Type> &first_matrix, const Matrix <Type> &second_matrix);
    template <typename Type>
    Matrix <Type> operator * (const Matrix <Type> &first_matrix, const double &c);
    template <typename Type>
    Matrix <Type> operator * (const double &c, const Matrix <Type> &first_matrix);
}

// verify if the input data is indeed a matrix
template <typename Type>
bool is_matrix(const std::vector < std::vector<Type> > &M){
    long m = M.size();
    long n = M[0].size();
    for (long i = 0; i < m; i++){
        if (M[i].size() != n){
            return false;
        }
    }
    return true;
}

// the constructor function
template <typename Type>
LinAlg::Matrix<Type>::Matrix(const std::vector<std::vector<Type>> &M){
    assert(::is_matrix(M) && "The input is not a matrix.");
    this->M = M;
    this->shape[0] = M.size();
    this->shape[1] = M[0].size();
}

// returns the transpose of the matrix
template <typename Type>
LinAlg::Matrix<Type> LinAlg::Matrix<Type>::T(){
    std::vector <std::vector <Type> > N;
    for (int i = 0; i < this->shape[1]; i++){
        std::vector <Type> temp_row;
        for (int j = 0; j < this->shape[0]; j++){
            temp_row.push_back(this->M[j][i]);
        }
        N.push_back(temp_row);
    }
    return LinAlg::Matrix<Type>(N);
}

// print out the matrix with some padding space
template <typename Type>
void LinAlg::Matrix<Type>::print(){
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

// overloading the + operator
template <typename Type>
LinAlg::Matrix<Type> LinAlg::operator+ (const LinAlg::Matrix<Type> &first_matrix, const LinAlg::Matrix<Type> &second_matrix){
    assert(first_matrix.shape[0] == second_matrix.shape[0] and first_matrix.shape[1] == second_matrix.shape[1]);
    std::vector<std::vector<Type>> result = first_matrix.M;
    for (int i = 0; i < first_matrix.shape[0]; i++){
        for (int j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] += second_matrix.M[i][j];
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// overloading the - operator
template <typename Type>
LinAlg::Matrix<Type> LinAlg::operator- (const LinAlg::Matrix<Type> &first_matrix, const LinAlg::Matrix<Type> &second_matrix){
    assert(first_matrix.shape[0] == second_matrix.shape[0] and first_matrix.shape[1] == second_matrix.shape[1]);
    std::vector<std::vector<Type>> result = first_matrix.M;
    for (int i = 0; i < first_matrix.shape[0]; i++){
        for (int j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] -= second_matrix.M[i][j];
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// overloading the scalar multiplication of the right
template <typename Type>
LinAlg::Matrix<Type> LinAlg::operator* (const LinAlg::Matrix<Type> &first_matrix, const double &c){
    std::vector<std::vector<Type>> result = first_matrix.M;
    for (int i = 0; i < first_matrix.shape[0]; i++){
        for (int j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] *= c;
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// overloading the scalar multiplication on the left
template <typename Type>
LinAlg::Matrix<Type> LinAlg::operator* (const double &c, const LinAlg::Matrix<Type> &first_matrix){
    std::vector<std::vector<Type>> result = first_matrix.M;
    for (int i = 0; i < first_matrix.shape[0]; i++){
        for (int j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] *= c;
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// access individual entry
template <typename Type>
Type LinAlg::Matrix<Type>::operator()(const long &row, const long &column){
    return this->M[row][column];
}
#endif //DEEPLEARNING_MATRIX_H
