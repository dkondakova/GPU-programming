#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

const int blockSize = 512;

const int N = 1e8;
const float PI = 3.14159265359f;

template <typename T>
__global__ void initArraySin(T* arr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    arr[idx] = sin((idx % 360) * PI / 180.0);
  }
}

template <typename T>
__global__ void initArraySinf(T* arr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    arr[idx] = sinf((idx % 360) * PI / 180.0);
  }
}

template <typename T>
__global__ void initArray__Sinf(T* arr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    arr[idx] = __sinf((idx % 360) * PI / 180.0);
  }
}

template <typename T>
void calculateError(const char* func_name, const char* type_name) {
  T* d_arr;
  cudaMalloc((void**)&d_arr, N * sizeof(T));
  
  dim3 BS(blockSize);
  dim3 GS((N + blockSize - 1) / blockSize);
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (func_name == "sin") {
    initArraySin<T><<<GS, BS>>>(d_arr, N);
  } else if (func_name == "sinf") {
    initArraySinf<T><<<GS, BS>>>(d_arr, N);
  } else if (func_name == "__sinf") {
    initArray__Sinf<T><<<GS, BS>>>(d_arr, N);
  } else {
    std::cout << "Unknown function." << std::endl;
  }
  
  cudaDeviceSynchronize();
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  
  T* arr = new T[N];
  cudaMemcpy(arr, d_arr, N * sizeof(T), cudaMemcpyDeviceToHost);
  
  double error = 0.0;
  for (int i = 0; i < N; ++i) {
    error += abs(sin((i % 360) * PI / 180.0) - arr[i]);
  }
  error /= N;
  
  std::cout << "(" << type_name << ", " << func_name << ") " <<
    "Error: " << error << ". " <<
    "Time: " << duration << " microseconds. " << std::endl;
  
  delete[] arr;
  cudaFree(d_arr);
}

int main() {
  calculateError<float>("sin", "float");
  calculateError<float>("sinf", "float");
  calculateError<float>("__sinf", "float");
  
  calculateError<double>("sin", "double");
  calculateError<double>("sinf", "double");
  calculateError<double>("__sinf", "double");
}
