#include <iostream>
#include <curand_kernel.h>
#include <png.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>

#include <curand_kernel.h>

#define CHANNELS 3
#define BLOCK_SIZE 16
#define POINT_RADIUS 10
#define N 131072
#define K 5

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

bool saveImage(const char *filename, png_bytep **row_pointers, const int &width, const int &height, int colorType) {
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
                     colorType, PNG_INTERLACE_NONE,
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

void drawCircle(unsigned char *image, int width, int height, int index, int radius) {
    int x = index % width;
    int y = index / width;

    if (x < 0 || x >= width || y < 0 || y >= height) {
        std::cerr << "Координаты за пределами изображениня." << std::endl;
        return;
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int dx = j - x;
            int dy = i - y;
            int distanceSquared = dx * dx + dy * dy;

            if (((radius - 2) * (radius - 2) <= distanceSquared) && (distanceSquared <= radius * radius)) {
                // Вычисление индекса пикселя в массиве
                int new_index = (i * width + j) * 3;

                // Задание цвета точки внутри окружности
                image[new_index] = 255;     // red
                image[new_index + 1] = 255; // green
                image[new_index + 2] = 0;   // blue
            }
        }
    }
}

void drawPoint(unsigned char *image, int width, int height, int index) {
    int x = index % width;
    int y = index / width;

    if (x < 0 || x >= width || y < 0 || y >= height) {
        std::cerr << "Координаты за пределами изображениня." << std::endl;
        return;
    }

    for (int i = -POINT_RADIUS; i <= POINT_RADIUS; ++i) {
        for (int j = -POINT_RADIUS; j <= POINT_RADIUS; ++j) {
            int new_x = x + i;
            int new_y = y + j;

            if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                int new_index = (new_y * width + new_x) * 3;

                image[new_index] = 255;     // red
                image[new_index + 1] = 255; // green
                image[new_index + 2] = 0;   // blue
            }
        }
    }
}

__global__ void sobelFilter(const unsigned char *inputImage, unsigned char *outputImage, int width, int height, int *whitePoints, int *numWhitePoints) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int GxR = 0, GyR = 0;
        int GxG = 0, GyG = 0;
        int GxB = 0, GyB = 0;

        // Ядро фильтра Собеля для обнаружения горизонтальных границ
        int sobelX[3][3] = {{-1, 0, 1},
                            {-2, 0, 2},
                            {-1, 0, 1}};

        // Ядро фильтра Собеля для обнаружения вертикальных границ
        int sobelY[3][3] = {{-1, -2, -1},
                            {0, 0, 0},
                            {1, 2, 1}};

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int neighborX = x + dx;
                int neighborY = y + dy;

                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    int pixelIndex = (neighborY * width + neighborX) * CHANNELS; // Индекс для каждого канала
                    GxR += sobelX[dx + 1][dy + 1] * inputImage[pixelIndex];
                    GyR += sobelY[dx + 1][dy + 1] * inputImage[pixelIndex];
                    GxG += sobelX[dx + 1][dy + 1] * inputImage[pixelIndex + 1];
                    GyG += sobelY[dx + 1][dy + 1] * inputImage[pixelIndex + 1];
                    GxB += sobelX[dx + 1][dy + 1] * inputImage[pixelIndex + 2];
                    GyB += sobelY[dx + 1][dy + 1] * inputImage[pixelIndex + 2];
                }
            }
        }

        // Общий градиент для каждого канала
        int gradientR = std::sqrt((float) (GxR * GxR + GyR * GyR));
        int gradientG = std::sqrt((float) (GxG * GxG + GyG * GyG));
        int gradientB = std::sqrt((float) (GxB * GxB + GyB * GyB));

        int index = y * width + x;
        // Запись результата в чёрно-белое изображение
        if (((float)(gradientR + gradientG + gradientB) / CHANNELS / 255.0f) > 0.95f) {
            int currentCount = atomicAdd(numWhitePoints, 1);
            whitePoints[currentCount] = index;
            outputImage[index] = 255;
        } else {
            outputImage[index] = 0;
        }
    }
}

int* applyBlur(const unsigned char *inputImage, unsigned char *outputImage,
        const size_t &inputImageSize, const size_t &outputImageSize,
        int width, int height, int &numWhitePoints) {
    // Выделение памяти на устройстве для входного и выходого (с применённым фильтром Собеля) изображений
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void **) &d_inputImage, inputImageSize);
    cudaMalloc((void **) &d_outputImage, outputImageSize);
    cudaMemcpy(d_inputImage, inputImage, inputImageSize, cudaMemcpyHostToDevice);

    // Выделение памяти на устройстве для массива белых точек и переменной для хранения их числа
    int *d_whitePoints;
    int *d_numWhitePoints;
    cudaMalloc((void**)&d_whitePoints, sizeof(int) * width * height);
    cudaMalloc((void**)&d_numWhitePoints, sizeof(int));
    cudaMemset(d_numWhitePoints, 0, sizeof(int));


    dim3 GS((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 BS(BLOCK_SIZE, BLOCK_SIZE, 1);

    sobelFilter<<<GS,BS>>>(d_inputImage, d_outputImage, width, height, d_whitePoints, d_numWhitePoints);

    cudaDeviceSynchronize();


    // Копирование результатов обратно на хост
    cudaMemcpy(&numWhitePoints, d_numWhitePoints, sizeof(int), cudaMemcpyDeviceToHost);
    int *whitePoints = new int[numWhitePoints];
    cudaMemcpy(whitePoints, d_whitePoints, sizeof(int) * numWhitePoints, cudaMemcpyDeviceToHost);

    // Копирование результатов обратно на хост
    cudaMemcpy(outputImage, d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);


    cudaFree(d_whitePoints);
    cudaFree(d_numWhitePoints);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return whitePoints;
}

__device__ void leastSquaresCircle(const int *points, int numPoints, int width, int& centerX, int& centerY, int& radius) {
    centerX = 0;
    centerY = 0;
    radius = 0;

    for (int iter = 0; iter < 3; ++iter) {
        float sumX = 0.0, sumY = 0.0, sumDist = 0.0;
        for (int i = 0; i < numPoints; ++i) {
            float dx = (points[i] % width) - centerX;
            float dy = (points[i] / width) - centerY;
            float dist = sqrt(dx * dx + dy * dy);

            sumX += dx * (dist - radius) / dist;
            sumY += dy * (dist - radius) / dist;
            sumDist += (dist - radius) * (dist - radius);
        }

        centerX += sumX / numPoints;
        centerY += sumY / numPoints;
        radius = sqrt(sumDist / numPoints);
    }
}

__global__ void processPoints(const int* whitePoints, int* array, int* pointsArray, int numWhitePoints, int width, int height, float* results) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < N) {
        curandState state;
        curand_init(threadId, 0, 0, &state);

        // Выбор K случайных точек для каждого потока
        int selectedPoints[K];
        for (int & selectedPoint : selectedPoints) {
            int randomIndex = curand(&state) % numWhitePoints;
            selectedPoint = whitePoints[randomIndex];
        }

        // Нахождение окружности методом наименьших квадратов
        int centerX, centerY, radius;
        leastSquaresCircle(selectedPoints, K, width, centerX, centerY, radius);

        // Оценка результатов и добавление в results
        float error = 0.0;
        for (int selectedPoint : selectedPoints) {
            int dx = (selectedPoint % width) - centerX;
            int dy = (selectedPoint / width) - centerY;
            float distance = std::sqrt((float) (dx * dx + dy * dy));
            error += abs(distance - radius);
        }

        results[threadId] = error;

        array[threadId * 2] = centerY * width + centerX;
        array[threadId * 2 + 1] = radius;

        for (int i = 0; i < K; ++i) {
            pointsArray[threadId * K + i] = selectedPoints[i];
        }
    }
}

int* applyProcessPoints(const int* whitePoints, int numWhitePoints,
                        int width, int height,
                        int &centerIndex, int &radius) {
    int* d_whitePoints;
    cudaMalloc((void**)&d_whitePoints, sizeof(int) * numWhitePoints);
    cudaMemcpy(d_whitePoints, whitePoints, sizeof(int) * numWhitePoints, cudaMemcpyHostToDevice);

    float* d_results;
    cudaMalloc((void**)&d_results, sizeof(float) * N);

    int* d_array;
    cudaMalloc((void**)&d_array, sizeof(int) * N * 2);

    int* d_pointsArray;
    cudaMalloc((void**)&d_pointsArray, sizeof(int) * N * K);


    dim3 GS(N / 256);
    dim3 BS(256);

    processPoints<<<GS,BS>>>(d_whitePoints, d_array, d_pointsArray, numWhitePoints, width, height, d_results);

    cudaDeviceSynchronize();


    auto* h_results = new float[N];
    cudaMemcpy(h_results, d_results, sizeof(float) * N, cudaMemcpyDeviceToHost);

    int bestResultIndex = 0;
    float bestResult = h_results[0];
    for (int i = 1; i < N; ++i) {
        if (h_results[i] < bestResult) {
            bestResultIndex = i;
            bestResult = h_results[i];
        }
    }

    int* h_array = new int[2];
    cudaMemcpy(h_array, d_array + bestResultIndex * 2, sizeof(int) * 2, cudaMemcpyDeviceToHost);

    centerIndex = h_array[0];
    radius = h_array[1];

    int* h_pointsArray = new int[K];
    cudaMemcpy(h_pointsArray, d_pointsArray + bestResultIndex * K, sizeof(int) * K, cudaMemcpyDeviceToHost);


    cudaFree(d_whitePoints);

    delete[] h_results;
    cudaFree(d_results);

    delete[] h_array;
    cudaFree(d_array);

    return h_pointsArray;
}

int main() {
    const char *inputFileName = "input.png";
    int width, height;
    png_bytep *input_row_pointers;

    if (!loadImage(inputFileName, &input_row_pointers, width, height)) return 1;

    std::cout << width << " " << height << std::endl;

    size_t inputImageSize = width * height * CHANNELS * sizeof(unsigned char);
    size_t outputImageSize = width * height * sizeof(unsigned char);
    auto *inputImage = new unsigned char[inputImageSize];
    auto *outputImage = new unsigned char[outputImageSize];

    for (int i = 0; i < height; ++i) {
        memcpy(&inputImage[i * width * CHANNELS * sizeof(unsigned char)],
               input_row_pointers[i],
               width * CHANNELS * sizeof(unsigned char));
    }

    int numWhitePoints = 0;
    int *whitePoints = applyBlur(inputImage, outputImage, inputImageSize, outputImageSize, width, height, numWhitePoints);

    int centerIndex, radius;
    int *points = applyProcessPoints(whitePoints, numWhitePoints, width, height, centerIndex, radius);

    // =====================

    drawCircle(inputImage, width, height, centerIndex, radius);

    for (int i = 0; i < K; i++) {
        drawPoint(inputImage, width, height, points[i]);
    }

    for (int i = 0; i < height; ++i) {
        memcpy(input_row_pointers[i],
               &inputImage[i * width * CHANNELS * sizeof(unsigned char)],
               width * CHANNELS * sizeof(unsigned char));
    }

    std::string inputFileName2 = "result.png";
    if (!saveImage(inputFileName2.c_str(), &input_row_pointers, width, height, PNG_COLOR_TYPE_RGB)) return 1;

    // =====================

    auto *output_row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (int i = 0; i < height; ++i) {
        output_row_pointers[i] = (png_byte *) malloc(sizeof(png_byte) * width * CHANNELS);
    }

    for (int i = 0; i < height; ++i) {
        memcpy(output_row_pointers[i],
               &outputImage[i * width * sizeof(unsigned char)],
               width * sizeof(unsigned char));
    }

    std::string outputFileName = "output.png";
    if (!saveImage(outputFileName.c_str(), &output_row_pointers, width, height, PNG_COLOR_TYPE_GRAY)) return 1;

    // =====================

    delete[] inputImage;
    delete[] outputImage;

    for (int i = 0; i < height; ++i) {
        delete[] output_row_pointers[i];
    }
    delete[] output_row_pointers;

    for (int i = 0; i < height; ++i) {
        delete[] input_row_pointers[i];
    }
    delete[] input_row_pointers;

    return 0;
}
