#pragma once
#include <memory>
#include <algorithm>
#include <iostream>
#include "utility.h"
#include "vector3.h"

#define STB_IMAGE_IMPLEMENTATION  
#define STB_IMAGE_WRITE_IMPLEMENTATION  
#include "stb_image.h"
#include "stb_image_write.h"

/// <summary>
/// Representation of RGB image.
/// </summary>
class Image {
public:
    Color3* data;
    float* raw_data;
    int width, height;

    const static int bytes_per_pixel = 3;

    Image(const char* filename) {
        auto components_per_pixel = bytes_per_pixel;
        stbi_ldr_to_hdr_gamma(1.0f);
        raw_data = stbi_loadf(
            filename, &width, &height, &components_per_pixel, 0);

        if (!raw_data) {
            std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
            width = height = 0;
            return;
        }

        data = new Color3[width * height];
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                data[i * width + j] = Color3(
                    raw_data[3 * i * width + j * 3],
                    raw_data[3 * i * width + j * 3 + 1],
                    raw_data[3 * i * width + j * 3 + 2]
                );
            }
        }
        delete[] raw_data;
    }

    ~Image() {
        delete[] data;
    }

    /// <summary>
    /// Image lookup with integer coordinates.
    /// </summary>
    /// <param name="x">Horizontal integer coordinate</param>
    /// <param name="y">Vertical integer coordinate</param>
    /// <returns>Pixel color</returns>
    Color3 Lookup(int x, int y) {
        return data[this->width * y + x];
    }

    /// <summary>
    /// Image lookup with UV coordinates.
    /// </summary>
    /// <param name="u">Horizontal UV coordinate</param>
    /// <param name="v">Vertical UV coordinate</param>
    /// <returns>Pixel color</returns>
    Color3 LookupUV(float u, float v) {
        return data[this->width * int((this->height - 1) * v) + int((this->width - 1) * u)];
    }

};