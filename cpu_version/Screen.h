#pragma once

class Screen {
public:
    Screen(int width, int height);
    unsigned char       *buffer;
    int                 width, height, numPixels;
    void setPixel(int x, int y, double r, double g, double b);
    void clear();
};

Screen::Screen(int width, int height) {
    /* Constructor for screen. */
    this->width = width;
    this->height = height;
    this->numPixels = width * height;
}

void Screen::setPixel(int x, int y, double r, double g, double b) {
    /* Sets the color value of a pixel mapping to coords (x,y). */
    // Skip pixels that are out of bounds:
    if (x < 0 || x >= width || y < 0 || y >= height)
        return;
    int pixelIndex = x + width * y;
    // Each pixel has an r, g, and b value - write them to the image.
    int pixelStart = 3 * pixelIndex;
    buffer[pixelStart] = ceil(r * 255);
    buffer[pixelStart + 1] = ceil(g * 255);
    buffer[pixelStart + 2] = ceil(b * 255);
}

void Screen::clear() {
    int i = 0;
    for (int j = 0; j < numPixels; ++j) {
        buffer[i++] = 0;
        buffer[i++] = 0;
        buffer[i++] = 0;
    }
}
