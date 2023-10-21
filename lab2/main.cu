#include <iostream>
#include <png.h>

#define BLUR_RADIUS 7

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
            (*row_pointers)[i] = (png_byte *) malloc(sizeof(png_byte) * width * 3);
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

bool saveImage(const char *filename, png_bytep **row_pointers, const int width, const int height) {
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

__global__ void applyBlurGlobal(unsigned char *inputImage, 
		unsigned char *outputImage, 
		int width, int height, int blurRadius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int channels = 3;
        int idx = (row * width + col) * channels;

        float blurFactor = 1.0 / ((2 * blurRadius + 1) * (2 * blurRadius + 1));

        float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -blurRadius; dy <= blurRadius; ++dy) {
            for (int dx = -blurRadius; dx <= blurRadius; ++dx) {
                int neighborRow = row + dy;
                int neighborCol = col + dx;

                if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
                    int neighborIdx = (neighborRow * width + neighborCol) * channels;
                    pixelValue.x += inputImage[neighborIdx];
                    pixelValue.y += inputImage[neighborIdx + 1];
                    pixelValue.z += inputImage[neighborIdx + 2];
                }
            }
        }

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * blurFactor);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * blurFactor);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * blurFactor);
    }
}

__global__ void applyBlurShared(unsigned char *inputImage, 
		unsigned char *outputImage, 
		int width, int height, int blurRadius) {
    __shared__ float3 sharedData[16][16];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int channels = 3;
        int idx = (row * width + col) * channels;

	sharedData[threadIdx.y][threadIdx.x] = make_float3(inputImage[idx],
                                                  inputImage[idx + 1],
                                                  inputImage[idx + 2]);
    }

    __syncthreads();

    if (col < width && row < height) {
        int channels = 3;
        int idx = (row * width + col) * channels;

        float blurFactor = 1.0 / ((2 * blurRadius + 1) * (2 * blurRadius + 1));

        int sharedX = threadIdx.x;
        int sharedY = threadIdx.y;

        float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -blurRadius; dy <= blurRadius; ++dy) {
            for (int dx = -blurRadius; dx <= blurRadius; ++dx) {
                int sharedRow = sharedY + dy;
                int sharedCol = sharedX + dx;

		if (sharedRow >= 0 && sharedRow < 16 && sharedCol >= 0 && sharedCol < 16) {
                    pixelValue.x += sharedData[sharedRow][sharedCol].x;
                    pixelValue.y += sharedData[sharedRow][sharedCol].y;
                    pixelValue.z += sharedData[sharedRow][sharedCol].z;
		}
            }
        }

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * blurFactor);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * blurFactor);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * blurFactor);
    }
}

texture<unsigned char, 1, cudaReadModeElementType> texRef;

__global__ void applyBlurTexture(unsigned char *outputImage, int width, int height, int blurRadius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int channels = 3;
        int idx = (row * width + col) * channels;

        float blurFactor = 1.0 / ((2 * blurRadius + 1) * (2 * blurRadius + 1));

        float3 pixelValue = make_float3(0.0f, 0.0f, 0.0f);
        for (int dy = -blurRadius; dy <= blurRadius; ++dy) {
            for (int dx = -blurRadius; dx <= blurRadius; ++dx) {
                int neighborRow = row + dy;
                int neighborCol = col + dx;

                if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
                    int neighborIdx = (neighborRow * width + neighborCol) * channels;
                    pixelValue.x += tex1Dfetch(texRef, neighborIdx);
                    pixelValue.y += tex1Dfetch(texRef, neighborIdx + 1);
                    pixelValue.z += tex1Dfetch(texRef, neighborIdx + 2);
                }
            }
        }

        outputImage[idx] = static_cast<unsigned char>(pixelValue.x * blurFactor);
        outputImage[idx + 1] = static_cast<unsigned char>(pixelValue.y * blurFactor);
        outputImage[idx + 2] = static_cast<unsigned char>(pixelValue.z * blurFactor);
    }
}
 
int main() {
    int blurRadius = 7;

    const char *inputFileName = "input.png";
    const char *outputFileName = "output.png";
    int width, height;

    png_bytep *row_pointers;
    if (!loadImage(inputFileName, &row_pointers, width, height)) return 1;

    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void **) &d_inputImage, imageSize);
    cudaMalloc((void **) &d_outputImage, imageSize);
    
    unsigned char *inputImage = new unsigned char[imageSize]; 
    unsigned char *outputImage = new unsigned char[imageSize];

    for (int i = 0; i < height; i++) {
        memcpy(&inputImage[i * width * 3 * sizeof(unsigned char)],
               row_pointers[i],
               width * 3 * sizeof(unsigned char));
    }

    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 GS(32, 32, 1);
    dim3 BS(16, 16, 1);

    //applyBlurShared<<<GS,BS>>>(d_inputImage, d_outputImage, width, height, blurRadius);

    //applyBlurShared<<<GS,BS>>>(d_inputImage, d_outputImage, width, height, blurRadius);

    /*cudaArray *cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&cuArray, &channelDesc, imageSize);
    cudaMemcpyToArray(cuArray, 0, 0, inputImage, imageSize, cudaMemcpyHostToDevice);
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = false;
    cudaBindTextureToArray(texRef, cuArray, channelDesc);
    applyBlurTexture<<<GS, BS>>>(d_outputImage, width, height, blurRadius);
    cudaUnbindTexture(texRef);
	*/
    size_t offset = 0;
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    //texRef.filterMode = cudaFilterModeLinear;
    //texRef.normalized = false;
    cudaBindTexture(&offset, texRef, d_inputImage, imageSize);
    applyBlurTexture<<<GS, BS>>>(d_outputImage, width, height, blurRadius);
    cudaUnbindTexture(texRef);
    
    std::cout << static_cast<int>(inputImage[0]) << std::endl;

    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; i++) {
        memcpy(row_pointers[i],
               &outputImage[i * width * 3 * sizeof(unsigned char)],
               width * 3 * sizeof(unsigned char));
    }

    if (!saveImage(outputFileName, &row_pointers, width, height)) return 1;

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    delete[] inputImage;
    delete[] outputImage;

    for (int i = 0; i < height; ++i) {
        delete[] row_pointers[i];
    }
    delete[] row_pointers;

    return 0;
}
