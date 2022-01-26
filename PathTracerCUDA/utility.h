#pragma once

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include <random>

// Usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
const float Infinity = std::numeric_limits<float>::infinity();
const float Pi = 3.1415926535897932385f;

// Utility Functions
__host__ __device__ inline float DegreesToRadians(float degrees) {
    return degrees * (CUDART_PI_F/ 180.0f);
}

__host__ __device__ inline float RadiansToDegrees(float radians) {
    return radians * (180.0f / CUDART_PI_F);
}


std::random_device                  rand_dev;
std::mt19937                        generator(rand_dev());
std::uniform_real_distribution<float>  distr(0.0f, 1.0f);

inline float randFloat() {
    return distr(generator);
}