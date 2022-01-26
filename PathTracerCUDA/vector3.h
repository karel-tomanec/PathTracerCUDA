#pragma once

#include <cmath>
#include <stdlib.h>
#include <iostream>
#include "Utility.h"

class Vector3 {
public:

	float x, y, z;

public:

	__host__ __device__ Vector3() = default;

	__host__ __device__ Vector3(float x)
		: x(x), y(x), z(x)
	{}

	__host__ __device__ Vector3(float x, float y, float z)
		: x(x), y(y), z(z)

	{}


	__host__ __device__ float& operator [](int i)
	{
		return (&x)[i];
	}

	__host__ __device__ const float& operator [](int i) const
	{
		return (&x)[i];
	}


	__host__ __device__ Vector3 operator -() const { return Vector3(-x, -y, -z); }

	__host__ __device__ Vector3& operator *=(const float a) {
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}

	__host__ __device__ Vector3& operator *=(const Vector3& v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	__host__ __device__ Vector3& operator /=(const float a) {
		return *this *= 1 / a;
	}

	__host__ __device__ Vector3& operator +=(const Vector3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__host__ __device__ Vector3& operator -=(const Vector3& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

};

__host__ __device__ inline Vector3 operator *(const Vector3& v, float a) {
	return Vector3(v.x * a, v.y * a, v.z * a);
}

__host__ __device__ inline Vector3 operator *(float a, const Vector3& v) {
	return Vector3(v.x * a, v.y * a, v.z * a);
}

__host__ __device__ inline Vector3 operator /(const Vector3& v, float a) {
	a = 1.0 / a;
	return Vector3(v.x * a, v.y * a, v.z * a);
}

__host__ __device__ inline Vector3 operator +(const Vector3& v, const Vector3& u) {
	return Vector3(v.x + u.x, v.y + u.y, v.z + u.z);
}

__host__ __device__ inline Vector3 operator -(const Vector3& v, const Vector3& u) {
	return Vector3(v.x - u.x, v.y - u.y, v.z - u.z);
}

__host__ __device__ inline Vector3 operator *(const Vector3& v, const Vector3& u) {
	return Vector3(v.x * u.x, v.y * u.y, v.z * u.z);
}

__host__ __device__ inline float Average(const Vector3& v) {
	return (v.x + v.y + v.z) / 3.0f;
}

__host__ __device__ inline float SquaredLength(const Vector3& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ inline float Length(const Vector3& v) {
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline Vector3 Normalize(const Vector3& v) {
	return v / Length(v);
}

__host__ __device__ inline float Dot(const Vector3& v, const Vector3& u) {
	return u.x * v.x + u.y * v.y + u.z * v.z;
}

__host__ __device__ inline Vector3 Cross(const Vector3& u, const Vector3& v)
{
	return Vector3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

__host__ __device__ inline Vector3 Reflect(const Vector3& v, const Vector3& n) {
	return 2.0f * Dot(v, n) * n - v;
}

inline std::ostream& operator<<(std::ostream& out, const Vector3& v) {
	return out << v.x << ' ' << v.y << ' ' << v.z;
}

#define RANDVEC3 Vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ Vector3 RandVecInUnitSphere(curandState *local_rand_state) {
	Vector3 p;
    do {
        p = 2.0f * RANDVEC3 - Vector3(1,1,1);
    } while (SquaredLength(p) >= 1.0f);
    return p;
}



__device__ inline Vector3 SphericalDirection(float sintheta, float costheta, float phi) {
	return Vector3(sintheta * cosf(phi), costheta, sintheta * sinf(phi));
}


__device__ inline float SphericalTheta(const Vector3& v) {
	float t = acosf(v.y);
	return t;
}

__device__ inline float SphericalPhi(const Vector3& v) {
	float p = atan2f(v.z, v.x);
	return (p < 0.0f) ? (p + 2.0f * CUDART_PI_F) : p;
	return p;
}

using Point3 = Vector3;
using Color3 = Vector3;
