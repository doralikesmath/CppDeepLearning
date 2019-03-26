# CppDeepLearning
Deep Learning from scratch with C++ with some support for parallel computing using OpenMP.
## 1. Design
I like the clean and clear designs of Keras and Scikit-learn, hence I am copying much of theirs here.
## 2. Algorithms
### 2.1 Neural Networks
1. Fully-connected feed-forward network

### 2.2. Machine Learning
#### 2.2.1. Clustering
1. k-mean and k-mean++

## 3. Compiling
### 3.1. To complie with Clang
1. Clone this repository

        git clone git@github.com:doralikesmath/Deep-Learning.git
        
2. Create the `build` folder

        cd Deep-Learning
        mkdir build
        cd build

3. Create the `Makefile` using `CMake`

        cmake .. -D CMAKE_CXX_COMPILER=clang++ - DCMAKE_C_COMPILER=clang
        
4. Compile the test program

        make

### 3.2. To complie on MacOS with OpenMP
As this moment, my implementation of OpenMP is broken so please don't try this yet.

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
