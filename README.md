# Deep-Learning
Deep Learning from scratch with C++ with some support for parallel computing using OpenMP.

## To complie with Clang
1. Clone this repository

        git clone git@github.com:doralikesmath/Deep-Learning.git

2. Create the `Makefile` using `CMake`

        cmake .. -D CMAKE_CXX_COMPILER=clang++ - DCMAKE_C_COMPILER=clang
        
3. Compile the test program

        make

## To complie on MacOS with OpenMP
1. Install the lastest version of `gcc`, since the default `clang` compiler on MacOS does not support the compiling flag `-fopenmp` we need for OpenMP.

        brew install gcc

2. Clone this repository
    
        git clone git@github.com:doralikesmath/Deep-Learning.git
        
3. Create the `build` folder

        cd Deep-Learning
        mkdir build
        cd build

4. Create the `Makefile` using `CMake`

        cmake -D CMAKE_C_COMPILER=/usr/local/bin/gcc-8 -D CMAKE_CXX_COMPILER=/usr/local/bin/g++-8 ..
        
5. Finally, compile the test program

        make
