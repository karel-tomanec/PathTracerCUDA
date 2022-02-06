#pragma once
#include "utility.h"
#include "vector3.h"


/// <summary>
/// An infinitely far away area light source that surrounds the entire scene.
/// </summary>
class EnvironmentMap {
public:

	__device__ EnvironmentMap() = default;

	__device__ EnvironmentMap(Color3* image, int width, int height) : image(image), width(width), height(height) {
	}

	__device__ Color3 Lookup(Vector3 dir) const {
		float u = SphericalPhi(dir) / (2.0f * CUDART_PI_F);
		float v = SphericalTheta(dir) / CUDART_PI_F;
		Color3 c = image[this->width * int((this->height - 1) * v) + int((this->width - 1) * u)];
		c.x = __saturatef(c.x);
		c.y = __saturatef(c.y);
		c.z = __saturatef(c.z);
		return c;
	}

public:
	Color3* image;
	int width;
	int height;
};