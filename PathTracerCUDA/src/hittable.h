#pragma once
#include "utility.h"
#include "vector3.h"
#include "ray.h"
#include "aabb.h"

class Material;

class HitRecord {
public:
	
	Point3 point;
	Vector3 normal;
	float t = FLT_MAX;
	int matId;
	int hit = 0;

};

class Hittable {
public:

	AABB box;

	__device__ virtual bool Hit(const Ray& r, float tmax, float tmin, HitRecord& rec) const = 0;

};

inline bool BoxCompare(const AABB& a, const AABB& b, int axis) {

	return 0.5f * (a.minimum[axis] + a.maximum[axis]) < 0.5f * (b.minimum[axis] + b.maximum[axis]);
}