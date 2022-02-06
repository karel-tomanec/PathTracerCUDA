#pragma once
#include "utility.h"
#include "vector3.h"

/// <summary>
/// Representation of a light ray.
/// </summary>
class Ray {
public:

	__device__ Ray() = default;

	__device__ Ray(const Vector3& origin, const Vector3& direction) : origin(origin), direction(direction) { }

	__device__ Vector3 At(float t) const
	{
		return origin + t * direction;
	}

public:

	Vector3 origin;
	Vector3 direction;
};