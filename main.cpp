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
#include "src/include/maths/matrix.h"
#include "src/include/maths/utils.h"
#include "src/include/maths/losses.h"

int main(){
    std::vector <double> vec = {1, 2, 3, 4, 5};
    std::cout << "The l2 norm of the vector is " << maths::l2_norm(vec) << std::endl;
    std::vector <double> sm = maths::softmax(vec);
    for (int i = 0; i < sm.size(); i++){
        std::cout << "sm[" << i << "] = " << sm[i] << std::endl;
    }
    LinAlg::Matrix<int> M({{1,2,3}, {2,4,5}});
    std::cout << M;
    std::cout << "Squared error = " << loss::squared_loss(M(0), M(1)) << std::endl;
    std::cout << "Mean squared error = " << loss::mean_squared_error(M(0), M(1)) << std::endl;
    std::cout << M.reshape(6, 1).transpose();
    return 0;
}