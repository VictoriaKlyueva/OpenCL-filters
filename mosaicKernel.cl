__kernel void mosaicFilter(__global unsigned char* newImage, __global int* data) {
    int gid = get_global_id(0);

    __private int width, height, channels, blockSize = data[0], data[1], data[2], data[3];
    __private float red = 0, green = 0, blue = 0;

    for (int x = i; x < min(i + blockSize, height); x++) {
        for (int y = j; y < min(j + blockSize, width); y++) {
            red += image[(x * width + y) * channels];
            green += image[(x * width + y) * channels + 1];
            blue += image[(x * width + y) * channels + 2];
        }
    }

    red /= (blockSize * blockSize);
    green /= (blockSize * blockSize);
    blue /= (blockSize * blockSize);

    for (int x = i; x < min(i + blockSize, height); x++) {
        for (int y = j; y < min(j + blockSize, width); y++) {
            mosaicImage[(x * width + y) * channels] = red;
            mosaicImage[(x * width + y) * channels + 1] = green;
            mosaicImage[(x * width + y) * channels + 2] = blue;
        }
    }
}
