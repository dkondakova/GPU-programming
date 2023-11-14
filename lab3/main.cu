#include <iostream>
#include <omp.h>
#include <png.h>
#include <vector>
#include <cstring>
#include <chrono>

#define CHANNELS 3
#define BLUR_RADIUS 6
#define KERNEL_SIZE (2 * BLUR_RADIUS + 1)
#define BLUR_FACTOR (1.0f / ((2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1)))
#define BLOCK_SIZE 32
#define SHARED_BLOCK_SIZE (BLOCK_SIZE + 2 * BLUR_RADIUS)

bool loadImage(const char *filename, png_bytep **row_pointers, int &width, int &height) {
    FILE *file = nullptr;
    png_structp png = nullptr;
    png_infop info = nullptr;

    try {
        file = fopen(filename, "rb");
        if (!file) throw std::runtime_error("Не удалось открыть файл " + std::string(filename));

        png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png) throw std::runtime_error("Ошибка при создании png_struct");

        info = png_create_info_struct(png);
        if (!info) throw std::runtime_error("Ошибка при создании png_info");

        png_init_io(png, file);
        png_read_info(png, info);

        width = png_get_image_width(png, info);
        height = png_get_image_height(png, info);

        if (png_get_color_type(png, info) != PNG_COLOR_TYPE_RGB) throw std::runtime_error("Тип изображения не RGB");

        *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
        for (int i = 0; i < height; ++i) {
            (*row_pointers)[i] = (png_byte *) malloc(sizeof(png_byte) * width * CHANNELS);
        }

        png_read_image(png, (*row_pointers));

        png_destroy_read_struct(&png, &info, nullptr);
        fclose(file);

        return true;
    } catch (const std::exception &e) {
        std::cerr << "Ошибка при чтении изображения: " << e.what() << std::endl;

        if (file) fclose(file);
        if (png) png_destroy_read_struct(&png, nullptr, nullptr);
        if (info) png_destroy_read_struct(nullptr, &info, nullptr);

        return false;
    }
}

bool saveImage(const char *filename, png_bytep **row_pointers, const int &width, const int &height) {
    FILE *file = nullptr;
    png_structp png = nullptr;
    png_infop info = nullptr;

    try {
        file = fopen(filename, "wb");
        if (!file) throw std::runtime_error("Не удалось открыть или создать файл файл " + std::string(filename));

        png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png) throw std::runtime_error("Ошибка при создании png_struct");

        info = png_create_info_struct(png);
        if (!info) throw std::runtime_error("Ошибка при создании png_info");

        png_init_io(png, file);

        png_set_IHDR(png, info, width, height, 8,
                     PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);

        png_write_image(png, (*row_pointers));
        png_write_end(png, info);

        png_destroy_write_struct(&png, &info);
        fclose(file);

        return true;
    } catch (const std::exception &e) {
        std::cerr << "Ошибка при записи изображения: " << e.what() << std::endl;

        if (file) fclose(file);
        if (png) png_destroy_read_struct(&png, nullptr, nullptr);

        return false;
    }
}

__global__ void applyBlurGlobal(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int startRow) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy) {
            for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx) {
                int neighborRow = row + dy + startRow;
                int neighborCol = col + dx;

                if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
                    int neighborIdx = (neighborRow * width + neighborCol) * CHANNELS;
                    pixelValue.x += inputImage[neighborIdx];
                    pixelValue.y += inputImage[neighborIdx + 1];
                    pixelValue.z += inputImage[neighborIdx + 2];
                } else {
                    pixelValue.x += 255.0f;
                    pixelValue.y += 255.0f;
                    pixelValue.z += 255.0f;
                }
            }
        }

        int idx = (row * width + col) * CHANNELS;

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * BLUR_FACTOR);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * BLUR_FACTOR);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * BLUR_FACTOR);
    }
}
__global__ static void applyBlurShared_fast_v2(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int startRow) {
    __shared__ float3 sharedData[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

    int destCol = (threadIdx.y * BLOCK_SIZE + threadIdx.x) / SHARED_BLOCK_SIZE;
    int destRow = (threadIdx.y * BLOCK_SIZE + threadIdx.x) % SHARED_BLOCK_SIZE;
    int srcCol = blockIdx.y * BLOCK_SIZE + destCol - KERNEL_SIZE / 2 + startRow;
    int srcRow = blockIdx.x * BLOCK_SIZE + destRow - KERNEL_SIZE / 2;

    if (srcCol >= 0 && srcCol < height && srcRow >= 0 && srcRow < width) {
        int srcIdx = (srcCol * width + srcRow) * CHANNELS;
        sharedData[destCol][destRow] = make_float3(inputImage[srcIdx], inputImage[srcIdx + 1], inputImage[srcIdx + 2]);
    } else {
        sharedData[destCol][destRow] = make_float3(255, 255, 255);
    }

    destCol = (threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE) / SHARED_BLOCK_SIZE;
    destRow = (threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE) % SHARED_BLOCK_SIZE;
    srcCol = blockIdx.y * BLOCK_SIZE + destCol - KERNEL_SIZE / 2 + startRow;
    srcRow = blockIdx.x * BLOCK_SIZE + destRow - KERNEL_SIZE / 2;

    if (destCol < SHARED_BLOCK_SIZE) {
        if (srcCol >= 0 && srcCol < height && srcRow >= 0 && srcRow < width) {
            int srcIdx = (srcCol * width + srcRow) * CHANNELS;
            sharedData[destCol][destRow] = make_float3(inputImage[srcIdx], inputImage[srcIdx + 1], inputImage[srcIdx + 2]);
        } else {
            sharedData[destCol][destRow] = make_float3(255, 255, 255);
        }
    }

    __syncthreads();

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < height && col < width) {
        float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy) {
            for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx) {
                int sharedRow = threadIdx.y + dy + BLUR_RADIUS;
                int sharedCol = threadIdx.x + dx + BLUR_RADIUS;

                pixelValue.x += sharedData[sharedRow][sharedCol].x;
                pixelValue.y += sharedData[sharedRow][sharedCol].y;
                pixelValue.z += sharedData[sharedRow][sharedCol].z;
            }
        }

        int idx = (row * width + col) * CHANNELS;

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * BLUR_FACTOR);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * BLUR_FACTOR);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * BLUR_FACTOR);
    }
}

texture<unsigned char, 1, cudaReadModeElementType> texRef;

__global__ void applyBlurTexture(unsigned char *outputImage, int width, int height, int startRow) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy) {
            for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx) {
                int neighborRow = row + dy + startRow;
                int neighborCol = col + dx;

                if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
                    int neighborIdx = (neighborRow * width + neighborCol) * CHANNELS;
                    pixelValue.x += tex1Dfetch(texRef, neighborIdx);
                    pixelValue.y += tex1Dfetch(texRef, neighborIdx + 1);
                    pixelValue.z += tex1Dfetch(texRef, neighborIdx + 2);
                } else {
                    pixelValue.x += 255.0f;
                    pixelValue.y += 255.0f;
                    pixelValue.z += 255.0f;
                }
            }
        }

        int idx = (row * width + col) * CHANNELS;

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * BLUR_FACTOR);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * BLUR_FACTOR);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * BLUR_FACTOR);
    }
}

void applyBlur(const unsigned char *inputImage, unsigned char *outputImage, size_t imageSize, int width, int height, const std::string& type) {
    auto start_time_all = std::chrono::high_resolution_clock::now();

    int nGPUs;
    cudaGetDeviceCount(&nGPUs);
    std::cout << "Кол-во GPU: " << nGPUs << std::endl;
    if (nGPUs >= 1) {
        omp_set_num_threads(nGPUs);
    }
    unsigned char outputImage_local[nGPUs][imageSize / nGPUs];
    #pragma omp parallel
    {

        int cpu_thread_id = omp_get_thread_num();
        int num_cpu_threads = omp_get_num_threads();
        cudaSetDevice(cpu_thread_id % num_cpu_threads); //set device

        unsigned char *d_inputImage, *d_outputImage;
        cudaMalloc((void **) &d_inputImage, imageSize);
        cudaMalloc((void **) &d_outputImage, imageSize / num_cpu_threads);
        cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

        if (type == "texture") {
            cudaBindTexture(0, texRef, d_inputImage, imageSize);
        }

        dim3 GS(width / BLOCK_SIZE, height / BLOCK_SIZE / num_cpu_threads, 1);
        dim3 BS(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 BS_s(BLOCK_SIZE + 2 * BLUR_RADIUS, BLOCK_SIZE + 2 * BLUR_RADIUS, 1);

        int startRow = cpu_thread_id * height / num_cpu_threads;

        auto start_time = std::chrono::high_resolution_clock::now();

        if (type == "global") {
            applyBlurGlobal<<<GS,BS>>>(d_inputImage, d_outputImage, width, height, startRow);
        } else if (type == "shared") {
            //applyBlurShared<<<GS,BS_s>>>(d_inputImage, d_outputImage, width, height);
            applyBlurShared_fast_v2<<<GS,BS>>>(d_inputImage, d_outputImage, width, height, startRow);
        } else if (type == "texture") {
            applyBlurTexture<<<GS,BS>>>(d_outputImage, width, height, startRow);
        }

        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        std::cout << "Time (" << type << "): " << duration << "us." << std::endl;

        if (type == "texture") {
            cudaUnbindTexture(texRef);
        }

        cudaMemcpy(outputImage_local[cpu_thread_id], d_outputImage, imageSize / num_cpu_threads, cudaMemcpyDeviceToHost);

        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
    }

    for (int i = 0; i < nGPUs; ++i) {
        size_t offset = i * (imageSize / nGPUs);

        std::memcpy(outputImage + offset, outputImage_local[i], imageSize / nGPUs);
    }

    auto end_time_all = std::chrono::high_resolution_clock::now();      
    auto duration_all = std::chrono::duration_cast<std::chrono::microseconds>(end_time_all - start_time_all).count();

    std::cout << "Time with data copying (" << type <<  "): " << duration_all << "us." << std::endl;
}

int main() {
    const char *inputFileName = "input1024.png";
    int width, height;
    png_bytep *row_pointers;

    if (!loadImage(inputFileName, &row_pointers, width, height)) return 1;

    size_t imageSize = width * height * CHANNELS * sizeof(unsigned char);
    auto *inputImage = new unsigned char[imageSize];
    auto *outputImage = new unsigned char[imageSize];

    for (int i = 0; i < height; i++) {
        memcpy(&inputImage[i * width * CHANNELS * sizeof(unsigned char)],
               row_pointers[i],
               width * CHANNELS * sizeof(unsigned char));
    }

    std::vector<std::string> types = {"global", "shared", "texture"};

    for (const auto& type : types) {
        applyBlur(inputImage, outputImage, imageSize, width, height, type);

        for (int i = 0; i < height; i++) {
            memcpy(row_pointers[i],
                   &outputImage[i * width * CHANNELS * sizeof(unsigned char)],
                   width * CHANNELS * sizeof(unsigned char));
        }

        std::string outputFileName = "output/" + type + ".png";
        if (!saveImage(outputFileName.c_str(), &row_pointers, width, height)) return 1;
    }

    delete[] inputImage;
    delete[] outputImage;

    for (int i = 0; i < height; ++i) {
        delete[] row_pointers[i];
    }
    delete[] row_pointers;

    return 0;
}
