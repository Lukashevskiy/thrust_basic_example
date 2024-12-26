#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include "cublas_utils.h"

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Solve SLAU Ax = B using LU decomposition
void solve_slae(float *d_A, float *d_B, float *d_X, int n, int batch_size) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int *d_P; // Pivot indices
    int *d_info; // Info array for errors
    CUDA_CHECK(cudaMalloc(&d_P, n * batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, batch_size * sizeof(int)));

    // Perform LU decomposition
    CUBLAS_CHECK(cublasSgetrfBatched(handle, n, &d_A, n, d_P, d_info, batch_size));

    // Solve the system using the LU factorization
    const float **d_A_array = (const float **)&d_A; // Pointer to array of matrix pointers
    float **d_B_array = &d_B;
    CUBLAS_CHECK(cublasSgetrsBatched(handle, CUBLAS_OP_N, n, 1, d_A_array, n, d_P, d_B_array, n, d_info, batch_size));

    // Copy result from d_B to d_X
    CUDA_CHECK(cudaMemcpy(d_X, d_B, n * batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

    // Clean up
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_info));
    CUBLAS_CHECK(cublasDestroy(handle));
}

// Print matrix A(nr_rows_A, nr_cols_A) stored in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
    for (int i = 0; i < nr_rows_A; ++i) {
        for (int j = 0; j < nr_cols_A; ++j) {
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Define matrix and vector dimensions
    int n = 3; // Matrix size (n x n)
    int batch_size = 1; // Number of matrices (batch size)

    // Host memory allocation
    float *h_A = (float *)malloc(n * n * sizeof(float));
    float *h_B = (float *)malloc(n * sizeof(float)); // Right-hand side vector
    float *h_X = (float *)malloc(n * sizeof(float)); // Solution vector

    // Device memory allocation
    float *d_A, *d_B, *d_X;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, n * sizeof(float)));

    // Initialize A and B
    GPU_fill_rand(d_A, n, n);
    GPU_fill_rand(d_B, n, 1);

    // Copy A and B to host for debugging
    CUDA_CHECK(cudaMemcpy(h_A, d_A, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B, d_B, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print matrices
    std::cout << "Matrix A:" << std::endl;
    print_matrix(h_A, n, n);
    std::cout << "Vector B:" << std::endl;
    print_matrix(h_B, n, 1);

    // Solve SLAU Ax = B
    solve_slae(d_A, d_B, d_X, n, batch_size);

    // Copy result to host
    CUDA_CHECK(cudaMemcpy(h_X, d_X, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print solution
    std::cout << "Solution X:" << std::endl;
    print_matrix(h_X, n, 1);

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_X));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_X);

    return 0;
}
