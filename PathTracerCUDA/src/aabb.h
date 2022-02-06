#pragma once
#include "utility.h"
#include "vector3.h"
#include "ray.h"
#include "hittable.h"

/// <summary>
/// Representation of axis aligned bounding box that is surrounding primitives
/// </summary>
class AABB {
public:

	__host__ __device__ AABB() {}

	__host__ __device__ AABB(const Point3& a, const Point3& b) {
		minimum = a;
		maximum = b;
	}

	/// <summary>
	/// Get intersected point between the ray and the AABB.
	/// </summary>
	/// <param name="ray">ray</param>
	/// <param name="tmin">minimal t</param>
	/// <param name="tmax">maximal t</param>
	/// <returns></returns>
	__device__ bool Hit(const Ray& ray, float tmin, float tmax) const {
		for (int i = 0; i < 3; ++i)
		{
			float invD = 1.0f / ray.direction[i];
			float t0 = (minimum[i] - ray.origin[i]) * invD;
			float t1 = (maximum[i] - ray.origin[i]) * invD;
			if (invD < 0.0f) {
				float tmp = t0;
				t0 = t1;
				t1 = tmp;
			}
			tmin = t0 > tmin ? t0 : tmin;
			tmax = t1 < tmax ? t1 : tmax;
			if (tmax <= tmin) return false;
		}
		return true;
	}


public:
	Point3 minimum;
	Point3 maximum;
};

/// <summary>
/// Surround two input boxes with a larger one.
/// </summary>
/// <param name="box0">box 0</param>
/// <param name="box1">box 1</param>
/// <returns>larger box</returns>
AABB MergeBoxes(AABB& box0, AABB& box1) {
	Point3 small(std::min(box0.minimum.x, box1.minimum.x),
		std::min(box0.minimum.y, box1.minimum.y),
		std::min(box0.minimum.z, box1.minimum.z));

	Point3 big(std::max(box0.maximum.x, box1.maximum.x),
		std::max(box0.maximum.y, box1.maximum.y),
		std::max(box0.maximum.z, box1.maximum.z));

	return AABB(small, big);
}
