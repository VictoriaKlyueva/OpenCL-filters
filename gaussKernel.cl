__kernel void gaussBlur(__global unsigned char* newImage, __global float* gaussKernel, __global int* data) {
    int gid = get_global_id(0);

    __private int width, height, channels, kernelSize = data[0], data[1], data[2], data[3];
    __private float red = 0, green = 0, blue = 0;
    __private int x = gid % width, y = gid / width;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            if ((x + i - kernelSize / 2 >= 0 && x + i - kernelSize / 2 < width) && 
                ((y + j - kernelSize / 2) >= 0 && (y + j - kernelSize / 2) < height)) {
                red += gaussKernel[i * kernelSize + j] * 
                       newImage[((y + j - kernelSize / 2) * width + (x + i - kernelSize / 2)) * channels];
                green += gaussKernel[i * kernelSize + j] * 
                         newImage[((y + j - kernelSize / 2) * width + (x + i - kernelSize / 2)) * channels + 1];
                blue += gaussKernel[i * kernelSize + j] * 
                        newImage[((y + j - kernelSize / 2) * width + (x + i - kernelSize / 2)) * channels + 2];
            }

        }
    }
    newImage[(y * width + x) * channels] = red;
    newImage[(y * width + x) * channels + 1] = green;
    newImage[(y * width + x) * channels + 2] = blue;
}
