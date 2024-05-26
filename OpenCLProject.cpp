#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <windows.h>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
using namespace std;
using namespace chrono;

// https://github.com/michel-meneses/great-opencl-examples
// https://habr.com/ru/articles/261323/


ofstream csvout;
const int MAX_SOURCE_SIZE = 10000;

struct Images {
    vector<const char*> imagesNames = { "different_sizes/300x300.png", "different_sizes/400x400.png",
                                       "different_sizes/500x500.png", "different_sizes/600x600.png",
                                       "different_sizes/950x950.png", "different_sizes/2400x2400.png"
                                      };
    vector <unsigned char*> imagesArray;
    vector <int> sizes;
    vector <int> channels;
};

struct Data {
    int width;
    int height;
    int channels;
    int kernelSize;

    Data(int width_, int height_, int channels_, int kernelSize_) {
        width = width_;
        height = height_;
        channels = channels_;
        kernelSize = kernelSize_;
    }
};


float gaussFunction(float x, float y, float sigma) {
    return exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
}

float* generatingGaussKernel(float sigma, int kernelSize=20) {
    float* kernel = new float[kernelSize * kernelSize];

    float sum = 0;
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i * kernelSize + j] = gaussFunction(i - kernelSize / 2, j - kernelSize / 2, sigma);
            sum += kernel[i * kernelSize + j];
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

cl_kernel formatCLKernel(string fileName, string kernelName, cl_context& context,
                         cl_device_id& deviceID, cl_int& ret) {
    FILE* fp = fopen(fileName.c_str(), "r");

    char* kernelCode = (char*)malloc(MAX_SOURCE_SIZE);
    size_t kernelSize = fread(kernelCode, 1, MAX_SOURCE_SIZE, fp);

    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelCode, (const size_t*)&kernelSize, &ret);
    cl_uint num_devices = 1;
    ret = clBuildProgram(program, num_devices, &deviceID, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &ret);

    return kernel;
}

unsigned char* gaussBlurOpenCL(unsigned char* image, int width, int height,
    int channels, float sigma, int kernelSize=20) {

    float* gaussKernel = generatingGaussKernel(sigma);
    unsigned char* newImage = image;
    int data[] = { width, height, channels, kernelSize };

    cl_uint retNumPlatforms, retNumDevices;
    cl_platform_id platformID;
    cl_device_id deviceID;
    cl_int ret;
    cl_context context;
    cl_command_queue commandQueue;

    /* get available platforms */
    ret = clGetPlatformIDs(1, &platformID, &retNumPlatforms);
    /* get available devices */
    ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &retNumDevices);
    /* create context */
    context = clCreateContext(NULL, retNumDevices, &deviceID, NULL, NULL, &ret);
    /* create command */
    commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &ret);

    cl_kernel kernel = formatCLKernel("gaussKernel.cl", "gaussBlur", context, deviceID, ret);

    cl_mem imageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * channels * width * height, NULL, &ret);
    cl_mem kernelBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * kernelSize * kernelSize, NULL, &ret);
    cl_mem dataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 4, NULL, &ret);

    cl_int ret2;

    // image buffer
    ret2 = clEnqueueWriteBuffer(commandQueue, imageBuffer, CL_TRUE, 0, sizeof(unsigned char) * width * height * channels, newImage, 0, NULL, NULL);
    ret2 = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
    // kernel buffer
    ret2 = clEnqueueWriteBuffer(commandQueue, kernelBuffer, CL_TRUE, 0, sizeof(float) * kernelSize * kernelSize, gaussKernel, 0, NULL, NULL);
    ret2 = clSetKernelArg(kernel, 1, sizeof(cl_mem), &kernelBuffer);
    // data buffer
    ret2 = clEnqueueWriteBuffer(commandQueue, dataBuffer, CL_TRUE, 0, sizeof(int) * 4, data, 0, NULL, NULL);
    ret2 = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dataBuffer);

    int numWorkItems = width * height;
    size_t globalWorkSize[1] = { numWorkItems };
    ret2 = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    ret2 = clEnqueueReadBuffer(commandQueue, imageBuffer, CL_TRUE, 0, sizeof(unsigned char) * 3 * width * height, newImage, 0, NULL, NULL);

    return newImage;
}


unsigned char* mosaicFilterOpenCL(unsigned char* image, int width, int height,
                                  int channels, int blockSize) {
    unsigned char* newImage = image;

    int data[] = { width, height, channels, blockSize };

    cl_uint retNumPlatforms, retNumDevices;
    cl_platform_id platformID;
    cl_device_id deviceID;
    cl_int ret;
    cl_context context;
    cl_command_queue commandQueue;

    
    ret = clGetPlatformIDs(1, &platformID, &retNumPlatforms);
    ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &retNumDevices);
    context = clCreateContext(NULL, retNumDevices, &deviceID, NULL, NULL, &ret);
    commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &ret);

    cl_kernel kernel = formatCLKernel("mosaicKernel.cl", "mosaicFilter", context, deviceID, ret);

    cl_mem imageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * width * height * channels, NULL, &ret);
    cl_mem dataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 4, NULL, &ret);

    cl_int ret2;

    // image buffer
    ret2 = clEnqueueWriteBuffer(commandQueue, imageBuffer, CL_TRUE, 0, sizeof(unsigned char) * width * height * channels, newImage, 0, NULL, NULL);
    ret2 = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
    // data buffer
    ret2 = clEnqueueWriteBuffer(commandQueue, dataBuffer, CL_TRUE, 0, sizeof(int) * 4, data, 0, NULL, NULL);
    ret2 = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dataBuffer);

    int numWorkItems = width * height;
    size_t globalWorkSize[1] = { numWorkItems };
    ret2 = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    ret2 = clEnqueueReadBuffer(commandQueue, imageBuffer, CL_TRUE, 0, sizeof(unsigned char) *  width * height * 3, newImage, 0, NULL, NULL);

    return newImage;
}

double checkTimeGauss(unsigned char* (*function)(unsigned char*, int, int, int, float),
    unsigned char* image, int width, int height, int channels, float sigma) {
    auto begin = steady_clock::now();
    function(image, width, height, channels, sigma);
    auto end = steady_clock::now();

    return duration_cast<microseconds> (end - begin).count() / 10000000.0;
}

double checkTimeMosaic(unsigned char* (*function)(unsigned char*, int, int, int, int),
    unsigned char* image, int width, int height, int channels, int blockSize) {
    auto begin = steady_clock::now();
    function(image, width, height, channels, blockSize);
    auto end = steady_clock::now();

    return duration_cast<microseconds> (end - begin).count() / 10000000.0;
}

void showTimeGaussOpenCLOneImage(unsigned char* image, int width, int height, int channels, float sigma,
                                 int kernelSize=20) {
    double sum = 0.0;
    int attemptsNumber = 1;
    for (int attempt = 0; attempt < attemptsNumber; attempt++) {
        sum += checkTimeGauss(reinterpret_cast<unsigned char* (*)(unsigned char*, int, int, int, float)>(&gaussBlurOpenCL),
                              image, width, height, channels, sigma);
    }

    csvout << ","  << sum / attemptsNumber;
}

void showTimeMosaicOpenCLOneImage(unsigned char* image, int width, int height, int channels, int blockSize=7) {
    double sum = 0.0;
    int attemptsNumber = 100;
    for (int attempt = 0; attempt < attemptsNumber; attempt++) {
        sum += checkTimeMosaic(reinterpret_cast<unsigned char* (*)(unsigned char*, int, int, int, int)>(&mosaicFilterOpenCL),
            image, width, height, channels, blockSize);
    }

    csvout << ","  << sum / attemptsNumber;
}

void showTimeGauss(Images images, float sigma=7.2) {
    csvout << "Gauss";
    for (int i = 0; i < 6; i++) {
        showTimeGaussOpenCLOneImage(images.imagesArray[i], images.sizes[i], images.sizes[i],
                                    images.channels[i], sigma);
    }
    csvout << "\n";
}

void showTimeMosaic(Images images, int blockSize=7) {
    csvout << "Mosaic";
    for (int i = 0; i < 6; i++) {
        showTimeMosaicOpenCLOneImage(images.imagesArray[i], images.sizes[i], images.sizes[i],
                                     images.channels[i], blockSize);
    }
    csvout << "\n";
}

Images getImages() {
    Images images;
    for (int i = 0; i < 6; i++) {
        int w, h, ch;
        unsigned char* image = stbi_load(images.imagesNames[i], &w, &h, &ch, 0);

        images.imagesArray.push_back(image);
        images.sizes.push_back(w);
        images.channels.push_back(ch);
    }

    return images;
}

void gaussProcess(Images images) {
    const char* imageName = "image.png";
    cout << "Received: " << imageName << endl;

    int width, height, channels;
    unsigned char* image = stbi_load(imageName, &width, &height, &channels, 0);
    cout << "width: " << width << "\n" << "height: " << height << "\n" << "channels: " << channels << endl;

    cout << endl;
    unsigned char* gaussImage = gaussBlurOpenCL(image, width, height, channels, 7.2);
    stbi_write_png("C:/Users/Legion/source/repos/OpenCLProject/different_sizes/Blured image.png", width, height, channels, gaussImage, width * channels);
    showTimeGauss(images);
}

void mosaicProcess(Images images) {
    const char* imageName = "image.png";

    int width, height, channels;
    unsigned char* image = stbi_load(imageName, &width, &height, &channels, 0);

    unsigned char* mosaicImage = mosaicFilterOpenCL(image, width, height, channels, 7);
    stbi_write_png("C:/Users/Legion/source/repos/OpenCLProject/different_sizes/Mosaic image.png", width, height, channels, mosaicImage, width * channels);

    showTimeMosaic(images);
}

int main() {
    csvout.open("time.csv");
    csvout << "300,400,500,600,950,2400\n";

    Images images = getImages();
    gaussProcess(images);               
    mosaicProcess(images);                                                                                                                                                                                               
    csvout.close();
}