#include <iostream>
#include <vector>
#include <chrono>
#include <png.h>

#define BLOCK_SIZE 16
#define CHANNELS 3
#define BLUR_RADIUS 8
#define BLUR_FACTOR (1.0f / ((2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1)))

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

__global__ void applyBlurGlobal(unsigned char *inputImage, unsigned char *outputImage, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = (row * width + col) * CHANNELS;

        float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy) {
            for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx) {
                int neighborRow = row + dy;
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

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * BLUR_FACTOR);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * BLUR_FACTOR);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * BLUR_FACTOR);
    }
}

__global__ void applyBlurShared(unsigned char *inputImage, unsigned char *outputImage, int width, int height) {
    __shared__ float3 sharedData[BLOCK_SIZE + 2 * BLUR_RADIUS][BLOCK_SIZE + 2 * BLUR_RADIUS];

    int col = blockIdx.x * blockDim.x + threadIdx.x - (2 * blockIdx.x + 1) * BLUR_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y - (2 * blockIdx.y + 1) * BLUR_RADIUS;

    int idx = (row * width + col) * CHANNELS;
    sharedData[threadIdx.y][threadIdx.x] = (col >= 0 && col < width && row >= 0 && row < height) ?
	     make_float3(inputImage[idx], inputImage[idx + 1], inputImage[idx + 2]) :
	     make_float3(255.0f, 255.0f, 255.0f);

    __syncthreads();

    if (col >= 0 && col < width && row >= 0 && row < height && 
		    threadIdx.x >= BLUR_RADIUS && threadIdx.x < (blockDim.x - BLUR_RADIUS) &&
		    threadIdx.y >= BLUR_RADIUS && threadIdx.y < (blockDim.y - BLUR_RADIUS)) {

        float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy) {
            for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx) {
                int sharedRow = threadIdx.y + dy;
                int sharedCol = threadIdx.x + dx;

		if (sharedRow >= 0 && sharedRow < BLOCK_SIZE + 2 * BLUR_RADIUS && 
				sharedCol >= 0 && sharedCol < BLOCK_SIZE + 2 * BLUR_RADIUS) {
                    pixelValue.x += static_cast<float>(sharedData[sharedRow][sharedCol].x);
                    pixelValue.y += static_cast<float>(sharedData[sharedRow][sharedCol].y);
                    pixelValue.z += static_cast<float>(sharedData[sharedRow][sharedCol].z);
		}
            }
        }

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * BLUR_FACTOR);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * BLUR_FACTOR);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * BLUR_FACTOR);
    }
}

texture<unsigned char, 1, cudaReadModeElementType> texRef;

__global__ void applyBlurTexture(unsigned char *outputImage, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = (row * width + col) * CHANNELS;

	float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy) {
            for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx) {
                int neighborRow = row + dy;
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

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * BLUR_FACTOR);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * BLUR_FACTOR);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * BLUR_FACTOR);
    }
}

void applyBlur(const unsigned char *inputImage, unsigned char *outputImage, size_t imageSize, int width, int height, const std::string& type) {
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void **) &d_inputImage, imageSize);
    cudaMalloc((void **) &d_outputImage, imageSize);
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    if (type == "texture") {
        //size_t offset = 0;
        //texRef.addressMode[0] = cudaAddressModeWrap;
        //texRef.addressMode[1] = cudaAddressModeWrap;
        cudaBindTexture(0, texRef, d_inputImage, imageSize);
    }
    
    dim3 GS(width / BLOCK_SIZE, height / BLOCK_SIZE, 1);
    dim3 BS(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 BS_s(BLOCK_SIZE + 2 * BLUR_RADIUS, BLOCK_SIZE + 2 * BLUR_RADIUS, 1);

    //int deviceID = 0; // Идентификатор устройства
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, deviceID);
    //int smCount = deviceProp.multiProcessorCount; // Количество SM на устройстве
    //int maxBlocksPerSM; // Количство блоков, которые можно запустить на SM с учетом ограничений ресурсов
    //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, applyBlurTexture, BS.x * BS.y, 0);
    //int maxBlocks = maxBlocksPerSM * smCount; // Максимальное количество блоков, которое можно запустить

    //std::cout << "Количество SM на устройстве: " << smCount << std::endl;
    //std::cout << "Количество блоков, которые можно запустить на SM: " << maxBlocksPerSM << std::endl;
    //std::cout << "Максимальное количество блоков, которое можно запустить: " << maxBlocks << std::endl;

    //cudaFuncAttributes kernelAttributes;
    //cudaFuncGetAttributes(&kernelAttributes, applyBlurTexture);
    //size_t sharedMemoryBytes = kernelAttributes.sharedSizeBytes;

    //std::cout << "Разделяемая память, требуемая ядром: " <<  sharedMemoryBytes << " байт" << std::endl;;

    auto start_time = std::chrono::high_resolution_clock::now();

    if (type == "global") {
        applyBlurGlobal<<<GS,BS>>>(d_inputImage, d_outputImage, width, height);
    } else if (type == "shared") {
        applyBlurShared<<<GS,BS_s>>>(d_inputImage, d_outputImage, width, height);
    } else if (type == "texture") {
        applyBlurTexture<<<GS, BS>>>(d_outputImage, width, height);
    }

    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::cout << "Time (" << type <<  "): " << duration << "us." << std::endl;

    if (type == "texture") {
        cudaUnbindTexture(texRef);
    }

    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
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
