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
#include "src/include/matrix.h"

int main(){
    LinAlg::Matrix<int> M1({{12, 220, 23091}, {259, 598, 209}, {2002, 72, 562}});
    LinAlg::Matrix<int> M2({{585, 300, 829},{52, 621, 1},{2, 7023, 498}});
    std::cout << "M1 + M2 = \n";
    (M1 + M2).print();
    std::cout << "M1 - M2 = \n";
    (M1 - M2).print();
    std::cout << "M1 is a " << M1.shape[0] << "x" << M1.shape[1] << " matrix." << std::endl;
    M1.print();
    LinAlg::Matrix<int> N = M1.T();
    std::cout << "M1^T is a " << N.shape[0] << "x" << N.shape[1] << " matrix." << std::endl;
    N.print();
    std::cout << "M1 * 2 = " << std::endl;
    (M1 * 2).print();
    std::cout << "0.2 * M1 = " << std::endl;
    (0.2 * M1).print();
    std::cout << "Notice that the above multiplication converts the result to a matrix with integer entries." << std::endl;
    std::cout << "2 * M1 * 3 = " << std::endl;
    (2 * M1 * 3).print();
    std::cout << "M1[0][0] = " << M1(0, 0) << std::endl;
    return 0;
}