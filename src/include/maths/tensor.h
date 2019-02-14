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
#include <cassert>
#include <random>

#ifndef DEEPLEARNING_MATRIX_H
#define DEEPLEARNING_MATRIX_H
namespace LinAlg{
    template <typename Type>
    classTensor{
    public:
        std::vector < std::vector<Type> > M;
        long shape[2];

        //constructors
       Tensor(const std::vector < std::vector <Type> > &M);
       Tensor(const long &number_of_rows, const long &number_of_cols, const Type &init_value);
       Tensor(const long &number_of_rows, const long &number_of_cols, const Type &lower_bound, const Type &upper_bound);
       Tensor(const char &c, const long &dimension);

        // the transpose
       Tensor transpose();

        // the determinant
        double det();

        // the inverseTensor
       Tensor inverse();

        // access individual element (i, j)
        inline Type operator () (const long &row, const long &column) const;
        inline Type & operator () (const long &row, const long &column);

        // access the row (i)
        inline std::vector <Type> operator() (const long &row) const;
        inline std::vector <Type> &operator() (const long &row);

        // apply a function to every element of theTensor
       Tensor apply_function(Type (*function)(Type));
       Tensor reshape(const long &number_of_rows, const long &number_of_cols);

        // print theTensor
        friend std::ostream &operator << (std::ostream &os, constTensor<Type> &M) {
            long m = M.shape[0];
            long n = M.shape[1];

            short fixed_width = 5;
#pragma omp parallel for
            for (long i = 0; i < m; i++){
#pragma omp parallel for
                for (long j = 0; j < n - 1; j++){
                    short w = (short)log10(M.M[i][j]) + (short)3;
                    if (w > fixed_width){
                        fixed_width = w;
                    }
                }
            }

            for (long i = 0; i < m; i++){
                for (long j = 0; j < n - 1; j++){
                    std::cout << std::setw(fixed_width) << M.M[i][j] << ",";
                }
                std::cout << std::setw(fixed_width) << M.M[i][n-1] << std::endl;
            }
            return os;
        }
    };

    template <typename Type>
   Tensor <Type> operator + (constTensor <Type> &first_matrix, constTensor <Type> &second_matrix);
    template <typename Type>
   Tensor <Type> operator - (constTensor <Type> &first_matrix, constTensor <Type> &second_matrix);
    template <typename Type>
   Tensor <Type> operator * (constTensor <Type> &first_matrix, const double &c);
    template <typename Type>
   Tensor <Type> operator * (const double &c, constTensor <Type> &first_matrix);
    template <typename Type>
   Tensor <Type> operator * (constTensor <Type> &first_matrix, constTensor <Type> &second_matrix);
    template <typename Type>
    std::vector <Type> operator * (constTensor <Type> &first_matrix, const std::vector <Type> &second_matrix);
    template <typename Type>
   Tensor <Type> hadamard_product (constTensor <Type> &first_matrix, constTensor <Type> &second_matrix);
}

// verify if the input data is indeed aTensor
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

// the constructor functions
template <typename Type>
LinAlg::Matrix<Type>::Matrix(const std::vector<std::vector<Type>> &M){
    assert(::is_matrix(M) && "The input is not aTensor.");
    this->M = M;
    this->shape[0] = M.size();
    this->shape[1] = M[0].size();
}

template <typename Type>
LinAlg::Matrix<Type>::Matrix(const long &number_of_rows, const long &number_of_cols, const Type &init_value){
    this->M =  std::vector<std::vector<Type>>(number_of_rows, std::vector<Type> (number_of_cols, init_value));
    this->shape[0] = number_of_rows;
    this->shape[1] = number_of_cols;
}

template <typename Type>
LinAlg::Matrix<Type>::Matrix(const char &c, const long &dimension){
    this->M =  std::vector<std::vector<Type>>(dimension, std::vector<Type> (dimension, 0));
    this->shape[0] = dimension;
    this->shape[1] = dimension;
    if ((c == 'I') || (c == 'i')){
#pragma omp parallel for
        for (long i = 0; i < dimension; i++){
            this->M[i][i] = 1;
        }
    }
}

template <typename Type>
LinAlg::Matrix<Type>::Matrix(const long &number_of_rows, const long &number_of_cols, const Type &lower_bound, const Type &upper_bound){
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;
    this->M =  std::vector<std::vector<Type>>(number_of_rows, std::vector<Type> (number_of_cols));
    for (long i = 0; i < number_of_rows; i++){
        for (long j = 0; j < number_of_cols; j++){
            this->M[i][j] = unif(re);
        }
    }
    this->shape[0] = number_of_rows;
    this->shape[1] = number_of_cols;
}

// returns the transpose of theTensor
template <typename Type>
LinAlg::Matrix<Type> LinAlg::Matrix<Type>::transpose(){
    std::vector <std::vector <Type> > N;
#pragma omp parallel for
    for (long i = 0; i < this->shape[1]; i++){
        std::vector <Type> temp_row;
#pragma omp parallel for
        for (long j = 0; j < this->shape[0]; j++){
            temp_row.push_back(this->M[j][i]);
        }
        N.push_back(temp_row);
    }
    return LinAlg::Matrix<Type>(N);
}

// overloading the + operator
template <typename Type>
LinAlg::Matrix<Type> LinAlg::operator+ (const LinAlg::Matrix<Type> &first_matrix, const LinAlg::Matrix<Type> &second_matrix){
    assert(first_matrix.shape[0] == second_matrix.shape[0] and first_matrix.shape[1] == second_matrix.shape[1]);
    std::vector<std::vector<Type>> result = first_matrix.M;
#pragma omp parallel for
    for (long i = 0; i < first_matrix.shape[0]; i++){
#pragma omp parallel for
        for (long j = 0; j < first_matrix.shape[1]; j++){
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
#pragma omp parallel for
    for (long i = 0; i < first_matrix.shape[0]; i++){
#pragma omp parallel for
        for (long j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] -= second_matrix.M[i][j];
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// Hadamard product of two matrices
template <typename Type>
LinAlg::Matrix<Type> LinAlg::hadamard_product(const LinAlg::Matrix<Type> &first_matrix, const LinAlg::Matrix<Type> &second_matrix){
    assert(first_matrix.shape[0] == second_matrix.shape[0] and first_matrix.shape[1] == second_matrix.shape[1]);
    std::vector<std::vector<Type>> result = first_matrix.M;
#pragma omp parallel for
    for (long i = 0; i < first_matrix.shape[0]; i++){
#pragma omp parallel for
        for (long j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] *= second_matrix.M[i][j];
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// overloading the scalar multiplication of the right
template <typename Type>
LinAlg::Matrix<Type> LinAlg::operator* (const LinAlg::Matrix<Type> &first_matrix, const double &c){
    std::vector<std::vector<Type>> result = first_matrix.M;
#pragma omp parallel for
    for (long i = 0; i < first_matrix.shape[0]; i++){
#pragma omp parallel for
        for (long j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] *= c;
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// overloading the scalar multiplication on the left
template <typename Type>
LinAlg::Matrix<Type> LinAlg::operator* (const double &c, const LinAlg::Matrix<Type> &first_matrix){
    std::vector<std::vector<Type>> result = first_matrix.M;
#pragma omp parallel for
    for (long i = 0; i < first_matrix.shape[0]; i++){
#pragma omp parallel for
        for (long j = 0; j < first_matrix.shape[1]; j++){
            result[i][j] *= c;
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// overloading the * operator between matrices
template <typename Type>
LinAlg::Matrix<Type> LinAlg::operator* (const LinAlg::Matrix<Type> &first_matrix, const LinAlg::Matrix<Type> &second_matrix){
    assert(first_matrix.shape[1] == second_matrix.shape[0]);
    std::vector<std::vector<Type>> result(first_matrix.shape[0], std::vector<Type> (second_matrix.shape[1], 0));
#pragma omp parallel for
    for (long i = 0; i < first_matrix.shape[0]; i++){
#pragma omp parallel for
        for (long j = 0; j < second_matrix.shape[1]; j++){
#pragma omp parallel for
            for (long k = 0; k < first_matrix.shape[1]; k++){
                result[i][j] += first_matrix.M[i][k] * second_matrix.M[k][j];
            }
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// overloading the * operator between aTensor and a vector
template <typename Type>
std::vector<Type> LinAlg::operator* (const LinAlg::Matrix<Type> &first_matrix, const std::vector<Type> &second_matrix){
    assert(first_matrix.shape[1] == second_matrix.size());
    std::vector<Type> result(first_matrix.shape[0]);
#pragma omp parallel for
    for (long i = 0; i < first_matrix.shape[0]; i++){
#pragma omp parallel for
        for (long k = 0; k < first_matrix.shape[1]; k++){
            result[i] += first_matrix.M[i][k] * second_matrix[k];
        }
    }
    return result;
}

// access individual entry
template <typename Type>
inline Type LinAlg::Matrix<Type>::operator()(const long &row, const long &column) const{
    return (this->M[row][column]);
}

// access individual entry
template <typename Type>
inline Type & LinAlg::Matrix<Type>::operator()(const long &row, const long &column){
    return this->M[row][column];
}

// access individual row
template <typename Type>
inline std::vector<Type> LinAlg::Matrix<Type>::operator() (const long &row) const{
    return this->M[row];
}

// access individual row
template <typename Type>
inline std::vector<Type> & LinAlg::Matrix<Type>::operator() (const long &row){
    return this->M[row];
}

// apply a function to every entry of theTensor
template <typename Type>
LinAlg::Matrix<Type> LinAlg::Matrix<Type>::apply_function(Type (*function)(Type)){
    std::vector<std::vector<Type>> result = this->M;
#pragma omp parallel for
    for (long i = 0; i < this->shape[0]; i++){
#pragma omp parallel for
        for (long j = 0; j < this->shape[1]; j++){
            result[i][j] = function(result[i][j]);
        }
    }
    return LinAlg::Matrix<Type>(result);
}

// reshape aTensor
template <typename Type>
LinAlg::Matrix<Type> LinAlg::Matrix<Type>::reshape(const long &number_of_rows, const long &number_of_cols){
    assert(this->shape[0] * this->shape[1] == number_of_cols * number_of_rows);
    std::vector <std::vector<Type>> result(number_of_rows, std::vector<Type>(number_of_cols));
#pragma omp parallel for
    for (long i = 0; i < number_of_rows; i++){
#pragma omp parallel for
        for (long j = 0; j < number_of_cols; j++){
            long k = i * number_of_cols + j;
            long x = k / this->shape[1];
            long y = k % this->shape[1];
            result[i][j] = this->M[x][y];
        }
    }
    return LinAlg::Matrix<Type>(result);
}

#endif //DEEPLEARNING_MATRIX_H