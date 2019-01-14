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

#ifndef DEEPLEARNING_MATRIX_H
#define DEEPLEARNING_MATRIX_H
namespace LinAlg{
    class Matrix{
        public:
            std::vector < std::vector<double> > M;
            long shape[2];
            Matrix(const std::vector < std::vector<double> > &M);
            Matrix T();
            void print();

    };
    Matrix operator + (const Matrix &first_matrix, const Matrix &second_matrix);
    Matrix operator - (const Matrix &first_matrix, const Matrix &second_matrix);
    Matrix operator * (const LinAlg::Matrix &first_matrix, const double &c);
    Matrix operator * (const double &c, const LinAlg::Matrix &first_matrix);
}
#endif //DEEPLEARNING_MATRIX_H
