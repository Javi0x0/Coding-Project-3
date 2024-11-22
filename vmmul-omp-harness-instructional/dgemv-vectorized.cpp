#include <immintrin.h> // For AVX intrinsics
const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

void my_dgemv(int n, double* A, double* x, double* y) {
    for (int i = 0; i < n; i++) {
        __m256d temp_vec = _mm256_setzero_pd(); // Initialize a vector with zeros
        int j;

        // Vectorized loop using AVX (process 4 doubles at a time)
        for (j = 0; j <= n - 4; j += 4) {
            __m256d a_vec = _mm256_loadu_pd(&A[i * n + j]); // Load 4 elements of A
            __m256d x_vec = _mm256_loadu_pd(&x[j]);         // Load 4 elements of x
            __m256d prod_vec = _mm256_mul_pd(a_vec, x_vec); // Multiply A and x
            temp_vec = _mm256_add_pd(temp_vec, prod_vec);   // Accumulate the result
        }

        // Horizontal addition of the AVX vector (reduce to a scalar)
        double temp_array[4];
        _mm256_storeu_pd(temp_array, temp_vec);
        double temp = temp_array[0] + temp_array[1] + temp_array[2] + temp_array[3];

        // Handle remaining elements (for elements not divisible by 4)
        for (; j < n; j++) {
            temp += A[i * n + j] * x[j];
        }

        // Add to y[i]
        y[i] += temp;
    }
}
